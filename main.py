from __future__ import annotations

import argparse
import contextlib
import io
import json
import litellm
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pexpect
from litellm import completion
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

litellm.suppress_debug_info = True

CONFIG_PATH = Path(__file__).with_name("config.json")
PROMPT_SENTINEL = "__TERMINUS2_PROMPT__> "
MAX_OUTPUT_BYTES = 10_000
SYSTEM_PROMPT = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{
 "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
 "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
 "commands": [
 {
 "keystrokes": "ls -la\\n",
 "duration": 0.1
 },
 {
 "keystrokes": "cd project\\n",
 "duration": 0.1
 }
 ],
 "task_complete": true
}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)
- "final_message": Optional user-facing completion summary shown when task completion is confirmed

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait for the command to complete before the next command will be executed (defaults to 1.0 if not present)

IMPORTANT: The text inside "keystrokes" will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:
- Most bash commands should end with a newline (\\n) to cause them to execute
- For special key sequences, use tmux-style escape sequences:
 - C-c for Ctrl+C
 - C-d for Ctrl+D

It is better to set a smaller duration than a longer duration. It is always possible to wait again if the prior output has not finished, by running {"keystrokes": "", "duration": 10.0} on subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see intermediate result status.
"""


@dataclass
class Config:
    model: str
    api_base: str
    api_key: str | None = None
    verbosity: int = 1
    temperature: float = 0.7
    max_turns: int = 50


@dataclass
class Command:
    keystrokes: str
    duration: float


@dataclass
class ParsedResponse:
    analysis: str
    plan: str
    commands: list[Command]
    task_complete: bool
    final_message: str | None


def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    data = json.loads(path.read_text())
    return Config(
        model=data["model"],
        api_base=data["api_base"],
        api_key=data.get("api_key"),
        verbosity=int(data.get("verbosity", 1)),
        temperature=float(data.get("temperature", 0.7)),
        max_turns=int(data.get("max_turns", 50)),
    )


def limit_output_length(output: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
    if len(output.encode("utf-8")) <= max_bytes:
        return output

    portion_size = max_bytes // 2
    output_bytes = output.encode("utf-8")
    first_portion = output_bytes[:portion_size].decode("utf-8", errors="ignore")
    last_portion = output_bytes[-portion_size:].decode("utf-8", errors="ignore")
    omitted_bytes = (
        len(output_bytes)
        - len(first_portion.encode("utf-8"))
        - len(last_portion.encode("utf-8"))
    )
    return (
        f"{first_portion}\n[... output limited to {max_bytes} bytes; "
        f"{omitted_bytes} interior bytes omitted ...]\n{last_portion}"
    )


def extract_json_content(response: str) -> str:
    json_start = -1
    json_end = -1
    brace_count = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(response):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if brace_count == 0:
                json_start = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and json_start != -1:
                json_end = i + 1
                break

    if json_start == -1 or json_end == -1:
        raise ValueError("No valid JSON object found in model response")

    return response[json_start:json_end]


def parse_response(text: str) -> ParsedResponse:
    json_payload = extract_json_content(text)
    data = json.loads(json_payload)

    for field in ("analysis", "plan", "commands"):
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in model response")

    if not isinstance(data["commands"], list):
        raise ValueError("'commands' must be an array")

    commands: list[Command] = []
    for i, command in enumerate(data["commands"]):
        if not isinstance(command, dict):
            raise ValueError(f"Command {i + 1} must be an object")
        if "keystrokes" not in command:
            raise ValueError(f"Command {i + 1} missing 'keystrokes'")
        keystrokes = command["keystrokes"]
        if not isinstance(keystrokes, str):
            raise ValueError(f"Command {i + 1} 'keystrokes' must be a string")
        duration = float(command.get("duration", 1.0))
        commands.append(Command(keystrokes=keystrokes, duration=min(duration, 60.0)))

    task_complete = data.get("task_complete", False)
    if isinstance(task_complete, str):
        task_complete = task_complete.lower() in {"true", "1", "yes"}
    elif not isinstance(task_complete, bool):
        task_complete = False

    final_message_raw = data.get("final_message")
    if final_message_raw is None:
        final_message = None
    elif isinstance(final_message_raw, str):
        final_message = final_message_raw
    else:
        raise ValueError("'final_message' must be a string when provided")

    return ParsedResponse(
        analysis=str(data["analysis"]),
        plan=str(data["plan"]),
        commands=commands,
        task_complete=task_complete,
        final_message=final_message,
    )


def clean_terminal_output(output: str) -> str:
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", output).replace("\r", "")


def normalize_command_output(output: str, command: Command) -> str:
    cleaned = clean_terminal_output(output)
    command_line = command.keystrokes.strip()
    normalized_lines: list[str] = []

    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in {PROMPT_SENTINEL.strip(), "%", "'", '"'}:
            continue
        if stripped.startswith("%"):
            # zsh prompt/echo artifacts
            continue
        if command_line and stripped == command_line:
            continue
        if stripped.startswith(PROMPT_SENTINEL):
            stripped = stripped.removeprefix(PROMPT_SENTINEL).strip()
            if not stripped:
                continue
        normalized_lines.append(stripped)

    return "\n".join(normalized_lines).strip()


@contextlib.contextmanager
def suppress_stdio_fd() -> Any:
    """Suppress low-level stdout/stderr writes from noisy SDK internals."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull)


def build_prompt(instruction: str, terminal_state: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Task Description:\n{instruction}\n\n"
        f"Current terminal state:\n{terminal_state}"
    )


def call_model(
    cfg: Config,
    prompt: str,
    history: list[dict[str, str]],
    api_key: str,
) -> str:
    model_name = cfg.model
    completion_kwargs: dict[str, Any] = {}
    if cfg.api_base.rstrip("/").endswith("openrouter.ai/api/v1"):
        completion_kwargs["custom_llm_provider"] = "openrouter"
        if model_name.startswith("openrouter/"):
            model_name = model_name.removeprefix("openrouter/")

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with suppress_stdio_fd(), contextlib.redirect_stdout(
                io.StringIO()
            ), contextlib.redirect_stderr(io.StringIO()):
                result = completion(
                    model=model_name,
                    api_base=cfg.api_base,
                    api_key=api_key,
                    temperature=cfg.temperature,
                    messages=history + [{"role": "user", "content": prompt}],
                    **completion_kwargs,
                )
            payload = cast(dict[str, Any], result)
            return str(payload["choices"][0]["message"]["content"])
        except Exception as err:  # noqa: BLE001
            last_error = err
            message = str(err).lower()
            is_retryable = any(
                token in message for token in ("429", "rate", "timeout", "temporarily")
            )
            if is_retryable and attempt < 3:
                time.sleep(2 * attempt)
                continue
            break

    raise RuntimeError(f"Model request failed: {last_error}")


def start_shell() -> pexpect.spawn:
    # Use a clean bash process to avoid zsh line-editor artifacts in captured output.
    child = pexpect.spawn(
        "/bin/bash",
        ["--noprofile", "--norc", "-i"],
        encoding="utf-8",
        timeout=15,
        echo=False,
    )
    child.sendline(f"export PS1='{PROMPT_SENTINEL}'")
    child.expect_exact(PROMPT_SENTINEL)
    return child


def execute_command(child: pexpect.spawn, cmd: Command) -> str:
    # Keep explicit wait commands predictable.
    if cmd.keystrokes == "":
        time.sleep(max(cmd.duration, 0.0))
        return ""

    if cmd.keystrokes.strip() == "C-c":
        child.sendcontrol("c")
    elif cmd.keystrokes.strip() == "C-d":
        child.sendcontrol("d")
    else:
        keystrokes = cmd.keystrokes
        # Use sendline for the common single-command newline case so command
        # execution and prompt matching stay in sync.
        if keystrokes.endswith("\n") and keystrokes.count("\n") == 1:
            child.sendline(keystrokes.rstrip("\n"))
        else:
            child.send(keystrokes)

    timeout = max(2.0, cmd.duration + 2.0)
    try:
        child.expect_exact(PROMPT_SENTINEL, timeout=timeout)
        raw_output = child.before or ""
    except pexpect.TIMEOUT:
        raw_output = child.before or ""
    return normalize_command_output(raw_output, cmd)


def render_response(console: Console, turn: int, parsed: ParsedResponse, verbosity: int) -> None:
    if verbosity >= 3:
        reasoning = f"analysis:\n{parsed.analysis}\n\nplan:\n{parsed.plan}"
        console.print(Panel(reasoning, title=f"Turn {turn} Reasoning", border_style="magenta"))


def render_command_output(
    console: Console,
    command: Command,
    output: str,
    verbosity: int,
    command_index: int,
) -> None:
    _ = command_index
    width = max(90, min(140, console.width - 2))

    def render_labeled_fixed(label: str, label_style: str, content: str) -> None:
        content_width = max(10, width - len(label))
        lines = content.splitlines() or [""]
        first = True

        for raw_line in lines:
            wrapped = textwrap.wrap(
                raw_line,
                width=content_width,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            if not wrapped:
                wrapped = [""]

            for segment in wrapped:
                prefix = label if first else (" " * len(label))
                line = Text(prefix, style=label_style)
                line.append(segment.ljust(content_width), style="white")
                console.print(line)
                first = False

    if command.keystrokes == "":
        input_text = "<wait>"
    elif command.keystrokes.strip() == "":
        input_text = "<enter>"
    else:
        input_text = command.keystrokes
    display_input = input_text.replace("\n", "\\n")
    normalized_output = output.strip() if output else ""
    output_text = normalized_output if normalized_output else "[no output]"

    if verbosity == 0:
        in_prefix = "in: "
        out_prefix = "out: "

        def fit_line(text: str, prefix_len: int) -> str:
            content_width = max(10, width - prefix_len)
            if len(text) > content_width:
                return text[: content_width - 3] + "..."
            return text.ljust(content_width)

        preview = display_input
        response_preview = output_text.replace("\n", " ")
        console.print(Text("─" * width, style="dim"))
        in_line = Text(in_prefix, style="cyan")
        in_line.append(fit_line(preview or "<wait>", len(in_prefix)), style="white")
        console.print(in_line)
        out_line = Text(out_prefix, style="green")
        out_line.append(fit_line(response_preview, len(out_prefix)), style="white")
        console.print(out_line)
        return

    console.print(Text("─" * width, style="dim"))
    render_labeled_fixed("cmd: ", "cyan", display_input)
    render_labeled_fixed("out: ", "green", output_text)


def render_issue_output(console: Console, kind: str, message: str, verbosity: int) -> None:
    # Keep verbosity 0 very compact by hiding parser/model issue chatter.
    if verbosity == 0:
        return

    width = max(90, min(140, console.width - 2))
    content_width = max(10, width - len("details: "))
    details_text = message.replace("\n", " ")
    wrapped = textwrap.wrap(
        details_text,
        width=content_width,
        replace_whitespace=False,
        drop_whitespace=False,
    )
    if not wrapped:
        wrapped = [""]

    console.print(Text("─" * width, style="dim"))
    error_line = Text("error: ", style="red")
    error_line.append(kind, style="white")
    error_line.append(" " * max(0, width - len("error: ") - len(kind)), style="white")
    console.print(error_line)

    for idx, segment in enumerate(wrapped):
        prefix = "details: " if idx == 0 else (" " * len("details: "))
        line = Text(prefix, style="red")
        line.append(segment.ljust(content_width), style="white")
        console.print(line)


def completion_confirmation_message(terminal_output: str) -> str:
    return (
        f"Current terminal state:\n{terminal_output}\n\n"
        "Are you sure you want to mark the task as complete? "
        "This will trigger completion and you won't be able to make further "
        'corrections. If so, include "task_complete": true in your JSON '
        "response again."
    )


def post_run_summary_prompt() -> str:
    return (
        "The task is now complete. Write a concise final message for the user "
        "summarizing what was accomplished. Include key outcomes and any caveats. "
        "Keep it short and user-facing."
    )


def get_post_run_final_message(
    cfg: Config, history: list[dict[str, str]], api_key: str
) -> str | None:
    try:
        response = call_model(cfg, post_run_summary_prompt(), history, api_key).strip()
    except Exception:
        return None
    return response or None


def run_agent(console: Console, instruction: str, cfg: Config, verbosity: int, api_key: str) -> int:
    child = start_shell()
    history: list[dict[str, str]] = []
    pending_completion = False
    pending_final_message: str | None = None
    terminal_state = "Current Terminal Screen:\n(empty)"
    prompt = build_prompt(instruction, terminal_state)
    command_counter = 0

    try:
        for turn in range(1, cfg.max_turns + 1):
            try:
                model_response = call_model(cfg, prompt, history, api_key)
            except Exception as err:
                render_issue_output(console, "model", str(err), verbosity)
                return 1
            history.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": model_response},
                ]
            )

            try:
                parsed = parse_response(model_response)
            except Exception as err:
                prompt = (
                    "Previous response had parsing errors:\n"
                    f"{err}\n\n"
                    "Please fix these issues and return a valid JSON response."
                )
                render_issue_output(console, "parser", str(err), verbosity)
                continue

            render_response(console, turn, parsed, verbosity)

            combined_output_parts: list[str] = []
            for cmd in parsed.commands:
                command_counter += 1
                output = execute_command(child, cmd)
                render_command_output(console, cmd, output, verbosity, command_counter)
                if output:
                    combined_output_parts.append(output)

            terminal_output = "\n".join(combined_output_parts).strip()
            terminal_output = limit_output_length(terminal_output or "[no new output]")

            if parsed.task_complete:
                if parsed.final_message and parsed.final_message.strip():
                    pending_final_message = parsed.final_message.strip()
                if pending_completion:
                    post_run_message = get_post_run_final_message(cfg, history, api_key)
                    done_text = (
                        post_run_message
                        if post_run_message
                        else pending_final_message
                        if pending_final_message
                        else "Task marked complete (double-confirmed)."
                    )
                    console.print(
                        Panel(
                            done_text,
                            title="Done",
                            border_style="green",
                        )
                    )
                    return 0
                pending_completion = True
                prompt = completion_confirmation_message(terminal_output)
            else:
                pending_completion = False
                pending_final_message = None
                prompt = terminal_output

        console.print(
            Panel(
                f"Reached max turns ({cfg.max_turns}) without completion.",
                title="Stopped",
                border_style="yellow",
            )
        )
        return 1
    finally:
        child.close(force=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Terminus-2 wrapper CLI")
    parser.add_argument(
        "instruction",
        nargs="*",
        help="Instruction for the agent. If omitted, interactive prompt is used.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 3],
        default=None,
        help="0: one line per tool call, 1: full tool inputs/responses, 3: + reasoning",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to config.json with model/API settings.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override max turns from config.json for this run.",
    )
    return parser.parse_args(argv)


def resolve_api_key(config_api_key: str | None) -> str | None:
    if config_api_key:
        raw = config_api_key.strip()
        # Allow config values like "$OPENROUTER_API_KEY" or "${OPENROUTER_API_KEY}".
        expanded = os.path.expandvars(raw).strip()
        if expanded and expanded != raw and "$" not in expanded:
            return expanded
        # Also allow a direct literal key in config.json.
        if raw and "$" not in raw:
            return raw

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    # Fallback: read the variable from zsh config in case it was set as a shell
    # variable but not exported into the current process environment.
    try:
        result = subprocess.run(
            [
                "zsh",
                "-ic",
                "source ~/.zshrc >/dev/null 2>&1; printf %s \"$OPENROUTER_API_KEY\"",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None

    candidate = result.stdout.strip()
    return candidate or None


def main() -> None:
    console = Console()
    args = parse_args(sys.argv[1:])

    try:
        config = load_config(args.config)
    except Exception as err:
        console.print(Panel(str(err), title="Config Error", border_style="red"))
        raise SystemExit(1) from err

    if args.max_turns is not None:
        config.max_turns = max(1, args.max_turns)
    if args.verbosity is None:
        args.verbosity = config.verbosity

    instruction = " ".join(args.instruction).strip()
    if not instruction:
        instruction = Prompt.ask("[bold]Enter instruction[/bold]").strip()
    if not instruction:
        console.print(Panel("Instruction is required.", border_style="red"))
        raise SystemExit(1)

    api_key = resolve_api_key(config.api_key)
    if not api_key:
        console.print(
            Panel(
                "API key not found. Set config.json api_key or OPENROUTER_API_KEY.",
                title="Missing API Key",
                border_style="red",
            )
        )
        raise SystemExit(1)

    console.print(
        Panel(
            f"Model: {config.model}\n"
            f"API Base: {config.api_base}\n"
            f"Verbosity: {args.verbosity}\n"
            f"Max Turns: {config.max_turns}",
            title="Terminus-2 Wrapper",
            border_style="cyan",
        )
    )
    raise SystemExit(run_agent(console, instruction, config, args.verbosity, api_key))


if __name__ == "__main__":
    main()
