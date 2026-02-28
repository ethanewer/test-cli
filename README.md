# Terminus-2 Wrapper CLI

This project provides a local CLI wrapper around the Terminus-2 JSON interaction pattern.  
It runs commands in a persistent local shell and formats output with Rich.

## Requirements

- Python 3.13+
- OpenRouter API key exported in your shell:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

The CLI reads the key from `OPENROUTER_API_KEY` (for example from your `.zshrc`).

## Configuration

Copy the example config and fill in local secrets:

```bash
cp config.example.json config.json
```

Model and API settings live in `config.json` (which is gitignored):

```json
{
  "model": "qwen/qwen3.5-35b-a3b",
  "api_base": "https://openrouter.ai/api/v1",
  "api_key": "$OPENROUTER_API_KEY",
  "temperature": 0.7,
  "max_turns": 50
}
```

- `model`: OpenRouter model name
- `api_base`: OpenRouter API base URL
- `api_key`: API key string or env reference like `$OPENROUTER_API_KEY`
- `temperature`: sampling temperature
- `max_turns`: maximum model turns before stopping

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
terminus2-cli --verbosity 1 --max-turns 10 --config ./config.json "Your instruction"
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
