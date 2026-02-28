from __future__ import annotations

import contextlib
import io
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, cast

import litellm
import pexpect
from final_summary import build_done_text
from litellm import completion

litellm.suppress_debug_info = True

PROMPT_SENTINEL = "__TERMINUS2_PROMPT__> "
MAX_OUTPUT_BYTES = 10_000
SYSTEM_PROMPT = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {{
      "keystrokes": "ls -la\\n",
      "duration": 0.1
    }},
    {{
      "keystrokes": "cd project\\n",
      "duration": 0.1
    }}
  ],
  "task_complete": true
}}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait for the command to complete before the next command will be executed (defaults to 1.0 if not present)

IMPORTANT: The text inside "keystrokes" will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:
- Most bash commands should end with a newline (\\n) to cause them to execute
- For special key sequences, use tmux-style escape sequences:
  - C-c for Ctrl+C
  - C-d for Ctrl+D

The "duration" attribute specifies the number of seconds to wait for the command to complete (default: 1.0) before the next command will be executed. On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds. On commands (e.g., gcc, find, rustc) set a duration of 1.0 seconds. On slow commands (e.g., make, python3 [long running script], wget [file]) set an appropriate duration as you determine necessary.

It is better to set a smaller duration than a longer duration. It is always possible to wait again if the prior output has not finished, by running {{"keystrokes": "", "duration": 10.0}} on subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see intermediate result status.

Important notes:
- Each command's keystrokes are sent exactly as written to the terminal
- Do not include extra whitespace before or after the keystrokes unless it's part of the intended command
- Extra text before or after the JSON will generate warnings but be tolerated
- The JSON must be valid - use proper escaping for quotes and special characters within strings
- Commands array can be empty if you want to wait without taking action

Task Description:
{instruction}

Current terminal state:
{terminal_state}
"""


@dataclass
class ModelConfig:
    model: str
    api_base: str
    api_key: str | None = None
    temperature: float | None = None


@dataclass
class Config:
    active_model_key: str
    active_model: ModelConfig
    verbosity: int = 1
    max_turns: int = 50
    max_wait_seconds: float = 60.0


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


@dataclass
class AgentCallbacks:
    on_reasoning: Callable[[int, ParsedResponse], None] | None = None
    on_command_output: Callable[[Command, str], None] | None = None
    on_issue: Callable[[str, str], None] | None = None
    on_done: Callable[[str], None] | None = None
    on_stopped: Callable[[int], None] | None = None


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

        if char == '"':
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
    missing_fields = [
        field for field in ("analysis", "plan", "commands") if field not in data
    ]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    if not isinstance(data["commands"], list):
        raise ValueError("Field 'commands' must be an array")

    commands: list[Command] = []
    parse_error: str | None = None
    for i, command in enumerate(data["commands"]):
        if not isinstance(command, dict):
            parse_error = f"Command {i + 1} must be an object"
            break

        if "keystrokes" not in command:
            parse_error = f"Command {i + 1} missing required 'keystrokes' field"
            break

        keystrokes = command["keystrokes"]
        if not isinstance(keystrokes, str):
            parse_error = f"Command {i + 1} 'keystrokes' must be a string"
            break

        duration_value = command.get("duration", 1.0)
        duration = (
            float(duration_value) if isinstance(duration_value, (int, float)) else 1.0
        )
        commands.append(Command(keystrokes=keystrokes, duration=max(duration, 0.0)))

    task_complete = _coerce_task_complete(data.get("task_complete", False))
    if parse_error:
        if task_complete:
            commands = []
        else:
            raise ValueError(parse_error)

    final_message = _normalized_final_message(data.get("final_message"))
    return ParsedResponse(
        analysis=str(data["analysis"]),
        plan=str(data["plan"]),
        commands=commands,
        task_complete=task_complete,
        final_message=final_message,
    )


def _coerce_task_complete(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}

    return False


def _normalized_final_message(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        return value

    raise ValueError("'final_message' must be a string when provided")


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


def build_prompt(instruction: str, terminal_state: str, max_wait_seconds: float) -> str:
    del max_wait_seconds
    return SYSTEM_PROMPT.format(instruction=instruction, terminal_state=terminal_state)


def call_model(
    cfg: Config,
    prompt: str,
    history: list[dict[str, str]],
    api_key: str,
) -> str:
    model_name = cfg.active_model.model
    completion_kwargs: dict[str, Any] = {}
    if cfg.active_model.api_base.rstrip("/").endswith("openrouter.ai/api/v1"):
        completion_kwargs["custom_llm_provider"] = "openrouter"
        if model_name.startswith("openrouter/"):
            model_name = model_name.removeprefix("openrouter/")

    # Omit temperature when unset so provider defaults apply.
    if cfg.active_model.temperature is not None:
        completion_kwargs["temperature"] = cfg.active_model.temperature

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with (
                suppress_stdio_fd(),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                result = completion(
                    model=model_name,
                    api_base=cfg.active_model.api_base,
                    api_key=api_key,
                    messages=history + [{"role": "user", "content": prompt}],
                    **completion_kwargs,
                )
            payload = cast(dict[str, Any], result)
            return str(payload["choices"][0]["message"]["content"])
        except Exception as err:  # noqa: BLE001
            last_error = err
            message = str(err).lower()
            if (
                any(
                    token in message
                    for token in ("429", "rate", "timeout", "temporarily")
                )
                and attempt < 2
            ):
                time.sleep(2 * (attempt + 1))
                continue

            break

    raise RuntimeError(f"Model request failed: {last_error}")


def start_shell() -> pexpect.spawn:
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


def execute_command(child: pexpect.spawn, cmd: Command, max_wait_seconds: float) -> str:
    effective_wait = min(max(cmd.duration, 0.0), max(max_wait_seconds, 0.1))
    if cmd.keystrokes == "":
        time.sleep(effective_wait)
        return ""

    if cmd.keystrokes.strip() == "C-c":
        child.sendcontrol("c")
    elif cmd.keystrokes.strip() == "C-d":
        child.sendcontrol("d")
    else:
        keystrokes = cmd.keystrokes
        if keystrokes.endswith("\n") and keystrokes.count("\n") == 1:
            child.sendline(keystrokes.rstrip("\n"))
        else:
            child.send(keystrokes)

    timeout = min(max(cmd.duration, 0.0) + 2.0, max(max_wait_seconds, 0.1))
    try:
        child.expect_exact(PROMPT_SENTINEL, timeout=timeout)
        raw_output = child.before or ""
    except pexpect.TIMEOUT:
        raw_output = child.before or ""

    return normalize_command_output(output=raw_output, command=cmd)


def completion_confirmation_message(terminal_output: str) -> str:
    return (
        f"Current terminal state:\n{terminal_output}\n\n"
        "Are you sure you want to mark the task as complete? "
        "This will trigger your solution to be graded and you won't be able to "
        'make any further corrections. If so, include "task_complete": true '
        "in your JSON response again."
    )


def _append_turn_history(
    history: list[dict[str, str]],
    prompt: str,
    model_response: str,
) -> None:
    history.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": model_response},
        ]
    )


def _execute_turn_commands(
    child: pexpect.spawn,
    parsed: ParsedResponse,
    max_wait_seconds: float,
    callbacks: AgentCallbacks,
) -> str:
    combined_output_parts: list[str] = []
    for cmd in parsed.commands:
        output = execute_command(
            child=child,
            cmd=cmd,
            max_wait_seconds=max_wait_seconds,
        )
        if callbacks.on_command_output:
            callbacks.on_command_output(cmd, output)

        if output:
            combined_output_parts.append(output)

    terminal_output = "\n".join(combined_output_parts).strip()
    return limit_output_length(terminal_output or "[no new output]")


def run_agent(
    instruction: str,
    cfg: Config,
    api_key: str,
    callbacks: AgentCallbacks | None = None,
) -> int:
    callbacks = callbacks or AgentCallbacks()
    child = start_shell()
    history: list[dict[str, str]] = []
    pending_completion = False
    pending_final_message: str | None = None
    prompt = build_prompt(
        instruction=instruction,
        terminal_state="Current Terminal Screen:\n(empty)",
        max_wait_seconds=cfg.max_wait_seconds,
    )

    try:
        for turn in range(1, cfg.max_turns + 1):
            try:
                model_response = call_model(
                    cfg=cfg,
                    prompt=prompt,
                    history=history,
                    api_key=api_key,
                )
            except Exception as err:
                if callbacks.on_issue:
                    callbacks.on_issue("model", str(err))

                return 1
            _append_turn_history(
                history=history,
                prompt=prompt,
                model_response=model_response,
            )

            try:
                parsed = parse_response(model_response)
            except Exception as err:
                prompt = (
                    "Previous response had parsing errors:\n"
                    f"{err}\n\n"
                    "Please fix these issues and provide a proper JSON response."
                )
                if callbacks.on_issue:
                    callbacks.on_issue("parser", str(err))

                continue

            if callbacks.on_reasoning:
                callbacks.on_reasoning(turn, parsed)

            terminal_output = _execute_turn_commands(
                child=child,
                parsed=parsed,
                max_wait_seconds=cfg.max_wait_seconds,
                callbacks=callbacks,
            )

            if parsed.task_complete:
                if parsed.final_message and parsed.final_message.strip():
                    pending_final_message = parsed.final_message.strip()

                if pending_completion:
                    done_text = build_done_text(
                        call_model_fn=call_model,
                        cfg=cfg,
                        history=history,
                        api_key=api_key,
                        pending_final_message=pending_final_message,
                    )
                    if callbacks.on_done:
                        callbacks.on_done(done_text)

                    return 0
                pending_completion = True
                prompt = completion_confirmation_message(terminal_output)
            else:
                pending_completion = False
                pending_final_message = None
                prompt = terminal_output

        if callbacks.on_stopped:
            callbacks.on_stopped(cfg.max_turns)

        return 1
    finally:
        child.close(force=True)
