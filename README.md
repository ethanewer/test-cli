# Terminus-2 Wrapper CLI

This project provides a local CLI wrapper around the Terminus-2 JSON interaction pattern.  
It runs commands in a persistent local shell and formats output with Rich.

## Requirements

- Python 3.13+
- API key(s) exported in your shell for any providers you configure:

```bash
export OPENROUTER_API_KEY="your_key_here"
export OPENAI_API_KEY="your_openai_key_here"
```

Each model can reference its own env var in config.

## Configuration

Model settings live under `models` in the repository `config.json`.
This file is tracked, so prefer env-var references for `api_key` values instead of literal secrets.

```json
{
  "default_model": "openrouter_qwen",
  "models": {
    "openrouter_qwen": {
      "model": "qwen/qwen3.5-35b-a3b",
      "api_base": "https://openrouter.ai/api/v1",
      "api_key": "OPENROUTER_API_KEY"
    },
    "openai_codex": {
      "model": "gpt-5.3-codex",
      "api_base": "https://api.openai.com/v1",
      "api_key": "OPENAI_API_KEY"
    }
  },
  "max_turns": 50,
  "max_wait_seconds": 60
}
```

- `default_model`: default model key to use from `models`
- `models`: dict of named model profiles
- model profile `model`: provider model ID (for example `gpt-5.3-codex`)
- model profile `api_base`: provider base URL
- model profile `api_key`: literal key, env var name (`OPENAI_API_KEY`), or `$ENV_VAR`
- model profile `temperature` (optional): sampling temperature for that model; if omitted, provider defaults are used
- `max_turns`: maximum model turns before stopping
- `max_wait_seconds`: max time to wait for each tool call/command completion

## Install

```bash
pip install -e .
```

This installs the `terminus2-cli` command from the project entrypoint.

## Usage

Run with a positional instruction:

```bash
terminus2-cli "List files in current directory, then explain what you see."
```

Or run without instruction and enter it interactively when prompted:

```bash
terminus2-cli
```

Optional flags:

```bash
terminus2-cli --verbosity 1 --max-turns 10 --model openai_codex --config ./config.json "Your instruction"
```

- `--model <key>` selects a model key from `config.models`.

### Interactive Commands

When entering instruction interactively, you can use slash commands before starting the run:

- `/model`: choose a model from a numbered list.
- `/verbosity [0|1|3]`: set runtime verbosity.
- `/max_turns [N]`: set runtime max turns (`N >= 1`).
- `/max_wait_seconds [S]`: set runtime max wait (`S > 0`).

If you omit a value (for example, `/verbosity`), the CLI prompts you for one.

Example:

```bash
terminus2-cli
# Enter instruction: /model
# Available Models:
# 1. openrouter_qwen (...)
# 2. openai_codex (...)
# Enter model number: 2
# Enter instruction: /max_turns 20
# Enter instruction: /verbosity
# Enter verbosity (0, 1, or 3): 1
# Enter instruction: Diagnose failing tests
```

## Verbosity Levels

- `0`: one line per tool call
- `1`: all tool call inputs and responses
- `3`: all tool call inputs and responses, plus reasoning (`analysis` and `plan`)

## Completion Message

After `task_complete` is confirmed twice in a row, the CLI sends one extra
post-run prompt to the model (using the same chat history) to generate a final
user-facing summary message.

If that post-run summary call fails or returns empty text, the CLI falls back
to the optional `final_message` from the agent JSON response, and then to the
default completion text.
