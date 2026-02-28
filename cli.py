from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from core_agent import AgentCallbacks, Command, Config, ParsedResponse, run_agent

CONFIG_PATH = Path(__file__).with_name("config.json")


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
        expanded = os.path.expandvars(raw).strip()
        if expanded and expanded != raw and "$" not in expanded:
            return expanded
        if raw and "$" not in raw:
            return raw

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    try:
        result = subprocess.run(
            [
                "zsh",
                "-ic",
                'source ~/.zshrc >/dev/null 2>&1; printf %s "$OPENROUTER_API_KEY"',
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None

    candidate = result.stdout.strip()
    return candidate or None


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
        max_wait_seconds=float(data.get("max_wait_seconds", 60.0)),
    )


def _display_width(console: Console) -> int:
    fallback_width = max(20, console.width)
    detected_width = shutil.get_terminal_size(fallback=(fallback_width, 24)).columns
    return max(20, detected_width)


def _render_labeled_fixed(
    console: Console, width: int, label: str, label_style: str, content: str
) -> None:
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


def _fit_line(text: str, width: int, prefix_len: int) -> str:
    content_width = max(10, width - prefix_len)
    if len(text) > content_width:
        return text[: content_width - 3] + "..."
    return text.ljust(content_width)


def _render_response(
    console: Console, turn: int, parsed: ParsedResponse, verbosity: int
) -> None:
    if verbosity >= 3:
        reasoning = f"analysis:\n{parsed.analysis}\n\nplan:\n{parsed.plan}"
        console.print(
            Panel(reasoning, title=f"Turn {turn} Reasoning", border_style="magenta")
        )


def _render_command_output(
    console: Console,
    command: Command,
    output: str,
    verbosity: int,
) -> None:
    width = _display_width(console)

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
        preview = display_input
        response_preview = output_text.replace("\n", " ")
        console.print(Text("─" * width, style="dim"))
        in_line = Text(in_prefix, style="cyan")
        in_line.append(
            _fit_line(
                text=preview or "<wait>",
                width=width,
                prefix_len=len(in_prefix),
            ),
            style="white",
        )
        console.print(in_line)
        out_line = Text(out_prefix, style="green")
        out_line.append(
            _fit_line(
                text=response_preview,
                width=width,
                prefix_len=len(out_prefix),
            ),
            style="white",
        )
        console.print(out_line)
        return

    console.print(Text("─" * width, style="dim"))
    _render_labeled_fixed(
        console=console,
        width=width,
        label="cmd: ",
        label_style="cyan",
        content=display_input,
    )
    _render_labeled_fixed(
        console=console,
        width=width,
        label="out: ",
        label_style="green",
        content=output_text,
    )


def _render_issue_output(
    console: Console, kind: str, message: str, verbosity: int
) -> None:
    if verbosity == 0:
        return

    width = _display_width(console)
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
            f"Max Turns: {config.max_turns}\n"
            f"Max Wait: {config.max_wait_seconds}s",
            title="Terminus-2 Wrapper",
            border_style="cyan",
        )
    )

    callbacks = AgentCallbacks(
        on_reasoning=lambda turn, parsed: _render_response(
            console=console,
            turn=turn,
            parsed=parsed,
            verbosity=args.verbosity,
        ),
        on_command_output=lambda command, output: _render_command_output(
            console=console,
            command=command,
            output=output,
            verbosity=args.verbosity,
        ),
        on_issue=lambda kind, message: _render_issue_output(
            console=console,
            kind=kind,
            message=message,
            verbosity=args.verbosity,
        ),
        on_done=lambda done_text: console.print(
            Panel(done_text, title="Done", border_style="green")
        ),
        on_stopped=lambda max_turns: console.print(
            Panel(
                f"Reached max turns ({max_turns}) without completion.",
                title="Stopped",
                border_style="yellow",
            )
        ),
    )
    raise SystemExit(
        run_agent(
            instruction=instruction,
            cfg=config,
            api_key=api_key,
            callbacks=callbacks,
        )
    )


if __name__ == "__main__":
    main()
