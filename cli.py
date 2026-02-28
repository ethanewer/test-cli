from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from core_agent import (
    AgentCallbacks,
    Command,
    Config as AgentConfig,
    ModelConfig,
    ParsedResponse,
    run_agent,
)

CONFIG_PATH = Path(__file__).with_name("config.json")
ENV_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")


@dataclass
class ConfigModelEntry:
    model: str
    api_base: str
    api_key: str | None
    temperature: float | None


@dataclass
class LoadedConfig:
    default_model: str
    models: dict[str, ConfigModelEntry]
    verbosity: int
    max_turns: int
    max_wait_seconds: float


@dataclass
class InteractiveCommandResult:
    instruction: str
    selected_model: str | None = None
    updated_verbosity: int | None = None
    handled: bool = False


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
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model key from config.models to run with.",
    )

    return parser.parse_args(argv)


def _env_var_name(config_api_key: str | None) -> str | None:
    if not config_api_key:
        return None

    raw = config_api_key.strip()
    if not raw:
        return None

    if raw.startswith("$"):
        candidate = raw[1:]
        if ENV_NAME_PATTERN.fullmatch(candidate):
            return candidate
        return None

    if ENV_NAME_PATTERN.fullmatch(raw):
        return raw

    return None


def _shell_env_lookup(env_name: str) -> str | None:
    if not ENV_NAME_PATTERN.fullmatch(env_name):
        return None

    shell_command = f'source ~/.zshrc >/dev/null 2>&1; printf %s "${{{env_name}}}"'

    try:
        result = subprocess.run(
            ["zsh", "-ic", shell_command],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None

    candidate = result.stdout.strip()
    return candidate or None


def resolve_api_key(config_api_key: str | None) -> str | None:
    env_name = _env_var_name(config_api_key=config_api_key)
    if env_name:
        direct_value = os.getenv(env_name)
        if direct_value:
            return direct_value

        return _shell_env_lookup(env_name=env_name)

    if config_api_key and config_api_key.strip():
        return config_api_key.strip()

    return None


def load_config(path: Path) -> LoadedConfig:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    data = json.loads(path.read_text())
    raw_models = data.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("config.json must define a non-empty 'models' object.")

    models: dict[str, ConfigModelEntry] = {}
    for model_key, model_data in raw_models.items():
        if not isinstance(model_key, str) or not model_key.strip():
            raise ValueError("Each key in 'models' must be a non-empty string.")

        if not isinstance(model_data, dict):
            raise ValueError(f"Model '{model_key}' must be an object.")

        model_name = str(model_data.get("model", "")).strip()
        api_base = str(model_data.get("api_base", "")).strip()
        if not model_name or not api_base:
            raise ValueError(
                f"Model '{model_key}' must include non-empty 'model' and 'api_base'."
            )

        raw_temperature = model_data.get("temperature")
        temperature = float(raw_temperature) if raw_temperature is not None else None
        models[model_key] = ConfigModelEntry(
            model=model_name,
            api_base=api_base,
            api_key=model_data.get("api_key"),
            temperature=temperature,
        )

    default_model = str(data.get("default_model", "")).strip()
    if not default_model:
        raise ValueError("config.json must define 'default_model'.")
    if default_model not in models:
        raise ValueError("default_model must match a key in models.")

    return LoadedConfig(
        default_model=default_model,
        models=models,
        verbosity=int(data.get("verbosity", 1)),
        max_turns=int(data.get("max_turns", 50)),
        max_wait_seconds=float(data.get("max_wait_seconds", 60.0)),
    )


def select_model_dialog(console: Console, config: LoadedConfig) -> str:
    model_keys = list(config.models.keys())
    numbered_lines = []
    for index, model_key in enumerate(model_keys, start=1):
        numbered_lines.append(f"{index}. {model_key}")

    console.print(
        Panel("\n".join(numbered_lines), title="Available Models", border_style="cyan")
    )

    while True:
        raw_choice = Prompt.ask("[bold]Enter model number[/bold]").strip()
        if not raw_choice:
            continue
        if not raw_choice.isdigit():
            console.print(Panel("Please enter a valid number.", border_style="yellow"))
            continue

        selected_index = int(raw_choice)
        if selected_index < 1 or selected_index > len(model_keys):
            console.print(Panel("Model number out of range.", border_style="yellow"))
            continue

        return model_keys[selected_index - 1]


def select_verbosity_dialog(console: Console) -> int:
    console.print(
        Panel(
            "0. Minimal - shows one short line per tool call; does not show full I/O or reasoning\n"
            "1. Standard - shows full tool inputs and outputs; does not show reasoning\n"
            "3. Debug - shows full tool inputs/outputs and model reasoning (analysis/plan)",
            title="Verbosity Levels",
            border_style="cyan",
        )
    )

    while True:
        raw_choice = Prompt.ask("[bold]Enter verbosity (0, 1, or 3)[/bold]").strip()
        if not raw_choice:
            continue
        try:
            return _parse_verbosity(value=raw_choice)
        except ValueError:
            console.print(Panel("Please enter 0, 1, or 3.", border_style="yellow"))


def _interactive_help_panel() -> Panel:
    return Panel(
        "/model - choose active model from a numbered list\n"
        "/verbosity <0|1|3> - set output detail level\n"
        "  0: one line per tool call\n"
        "  1: full tool call inputs/responses\n"
        "  3: full inputs/responses + reasoning\n"
        "/max_turns <int>=1 - set max agent turns\n"
        "/max_wait_seconds <float>>0 - set per-command max wait",
        title="Interactive Commands",
        border_style="cyan",
    )


def parse_model_command(
    console: Console,
    instruction: str,
    config: LoadedConfig,
) -> tuple[str, str | None]:
    trimmed = instruction.strip()
    if trimmed != "/model" and not trimmed.startswith("/model "):
        return instruction, None

    remainder = trimmed.removeprefix("/model").strip()
    selected_model = select_model_dialog(console=console, config=config)
    return remainder, selected_model


def _parse_verbosity(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as err:
        raise ValueError("Verbosity must be one of: 0, 1, 3.") from err

    if parsed not in {0, 1, 3}:
        raise ValueError("Verbosity must be one of: 0, 1, 3.")

    return parsed


def _parse_max_turns(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as err:
        raise ValueError("max_turns must be an integer >= 1.") from err

    if parsed < 1:
        raise ValueError("max_turns must be an integer >= 1.")

    return parsed


def _parse_max_wait_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as err:
        raise ValueError("max_wait_seconds must be a number greater than 0.") from err

    if parsed <= 0:
        raise ValueError("max_wait_seconds must be a number greater than 0.")

    return parsed


def parse_interactive_command(
    console: Console,
    instruction: str,
    config: LoadedConfig,
) -> InteractiveCommandResult:
    trimmed = instruction.strip()
    if not trimmed:
        console.print(_interactive_help_panel())
        return InteractiveCommandResult(instruction="", handled=True)

    if not trimmed.startswith("/"):
        return InteractiveCommandResult(instruction=instruction, handled=False)

    if trimmed == "/model" or trimmed.startswith("/model "):
        remainder, selected_model = parse_model_command(
            console=console,
            instruction=instruction,
            config=config,
        )
        return InteractiveCommandResult(
            instruction=remainder,
            selected_model=selected_model,
            handled=True,
        )

    if trimmed == "/verbosity" or trimmed.startswith("/verbosity "):
        remainder = trimmed.removeprefix("/verbosity").strip()
        try:
            verbosity = (
                _parse_verbosity(value=remainder)
                if remainder
                else select_verbosity_dialog(console=console)
            )
        except ValueError as err:
            console.print(
                Panel(str(err), title="Invalid Command", border_style="yellow")
            )
            return InteractiveCommandResult(instruction="", handled=True)

        console.print(Panel(f"Verbosity set to {verbosity}.", border_style="cyan"))
        return InteractiveCommandResult(
            instruction="",
            updated_verbosity=verbosity,
            handled=True,
        )

    if trimmed == "/max_turns" or trimmed.startswith("/max_turns "):
        remainder = trimmed.removeprefix("/max_turns").strip()
        raw_value = (
            remainder or Prompt.ask("[bold]Enter max_turns (>= 1)[/bold]").strip()
        )
        try:
            config.max_turns = _parse_max_turns(value=raw_value)
        except ValueError as err:
            console.print(
                Panel(str(err), title="Invalid Command", border_style="yellow")
            )
            return InteractiveCommandResult(instruction="", handled=True)

        console.print(
            Panel(f"max_turns set to {config.max_turns}.", border_style="cyan")
        )
        return InteractiveCommandResult(instruction="", handled=True)

    if trimmed == "/max_wait_seconds" or trimmed.startswith("/max_wait_seconds "):
        remainder = trimmed.removeprefix("/max_wait_seconds").strip()
        raw_value = (
            remainder or Prompt.ask("[bold]Enter max_wait_seconds (> 0)[/bold]").strip()
        )
        try:
            config.max_wait_seconds = _parse_max_wait_seconds(value=raw_value)
        except ValueError as err:
            console.print(
                Panel(str(err), title="Invalid Command", border_style="yellow")
            )
            return InteractiveCommandResult(instruction="", handled=True)

        console.print(
            Panel(
                f"max_wait_seconds set to {config.max_wait_seconds}.",
                border_style="cyan",
            )
        )
        return InteractiveCommandResult(instruction="", handled=True)

    console.print(
        Panel(
            "Unknown command. Available commands: /model, /verbosity, /max_turns, /max_wait_seconds",
            title="Invalid Command",
            border_style="yellow",
        )
    )
    return InteractiveCommandResult(instruction="", handled=True)


def resolve_model_key(
    config: LoadedConfig,
    cli_model_key: str | None,
    selected_model_key: str | None,
) -> str:
    if cli_model_key:
        cleaned = cli_model_key.strip()
        if cleaned not in config.models:
            available = ", ".join(config.models.keys())
            raise ValueError(
                f"Unknown model key '{cleaned}'. Available model keys: {available}"
            )
        return cleaned

    if selected_model_key:
        return selected_model_key

    return config.default_model


def build_agent_config(config: LoadedConfig, model_key: str) -> AgentConfig:
    selected = config.models[model_key]
    return AgentConfig(
        active_model_key=model_key,
        active_model=ModelConfig(
            model=selected.model,
            api_base=selected.api_base,
            api_key=selected.api_key,
            temperature=selected.temperature,
        ),
        verbosity=config.verbosity,
        max_turns=config.max_turns,
        max_wait_seconds=config.max_wait_seconds,
    )


def _display_width(console: Console) -> int:
    fallback_width = max(20, console.width)
    detected_width = shutil.get_terminal_size(fallback=(fallback_width, 24)).columns
    return max(20, detected_width)


def _render_labeled_fixed(
    console: Console, width: int, label: str, label_style: str, content: str,
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
    console: Console, kind: str, message: str, verbosity: int,
) -> None:
    # Always surface fatal model-call failures, even in quiet mode.
    if verbosity == 0 and kind != "model":
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
        loaded_config = load_config(args.config)
    except Exception as err:
        console.print(Panel(str(err), title="Config Error", border_style="red"))
        raise SystemExit(1) from err

    if args.max_turns is not None:
        loaded_config.max_turns = max(1, args.max_turns)

    if args.verbosity is None:
        args.verbosity = loaded_config.verbosity

    instruction = " ".join(args.instruction).strip()
    selected_model_from_instruction: str | None = None
    if not instruction:
        while True:
            candidate_instruction = Prompt.ask("[bold]Enter instruction[/bold]").strip()
            command_result = parse_interactive_command(
                console=console,
                instruction=candidate_instruction,
                config=loaded_config,
            )
            if command_result.handled:
                if command_result.selected_model:
                    selected_model_from_instruction = command_result.selected_model
                if command_result.updated_verbosity is not None:
                    args.verbosity = command_result.updated_verbosity
                if command_result.instruction:
                    instruction = command_result.instruction
                    break
                continue

            if command_result.instruction:
                instruction = command_result.instruction
                break
    else:
        instruction, selected_model = parse_model_command(
            console=console,
            instruction=instruction,
            config=loaded_config,
        )
        if selected_model:
            selected_model_from_instruction = selected_model

    if not instruction:
        console.print(Panel("Instruction is required.", border_style="red"))
        raise SystemExit(1)

    try:
        active_model_key = resolve_model_key(
            config=loaded_config,
            cli_model_key=args.model,
            selected_model_key=selected_model_from_instruction,
        )
    except ValueError as err:
        console.print(Panel(str(err), title="Config Error", border_style="red"))
        raise SystemExit(1) from err

    agent_config = build_agent_config(config=loaded_config, model_key=active_model_key)
    agent_config.verbosity = args.verbosity
    api_key = resolve_api_key(config_api_key=agent_config.active_model.api_key)
    if not api_key:
        env_name = _env_var_name(config_api_key=agent_config.active_model.api_key)
        if env_name:
            message = (
                f"API key not found for model '{active_model_key}'. "
                f"Set env var {env_name} or provide a literal api_key."
            )
        else:
            message = (
                f"API key not found for model '{active_model_key}'. "
                "Set api_key to a literal value or env var name."
            )
        console.print(
            Panel(
                message,
                title="Missing API Key",
                border_style="red",
            )
        )
        raise SystemExit(1)

    if args.verbosity == 0:
        panel_lines = [f"Model: {active_model_key}"]
    else:
        panel_lines = [
            f"Model Key: {active_model_key}",
            f"Model: {agent_config.active_model.model}",
            f"API Base: {agent_config.active_model.api_base}",
            f"Verbosity: {args.verbosity}",
            f"Max Turns: {agent_config.max_turns}",
            f"Max Wait: {agent_config.max_wait_seconds}s",
        ]

    console.print(
        Panel(
            "\n".join(panel_lines),
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
            cfg=agent_config,
            api_key=api_key,
            callbacks=callbacks,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        Console().print(
            Panel("Cancelled by user.", title="Stopped", border_style="yellow")
        )
        raise SystemExit(130)
