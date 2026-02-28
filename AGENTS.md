# AGENTS.md

## Tooling

- Use `uv` for Python commands and dependency management (for example: `uv run ...`, `uv add ...`).
- Format code with `ruff format` before finishing changes.
- Run lint checks with `ruff check` before finishing changes.

## Style Guidelines

### `if` / `else` spacing

- Do not add a blank line between an `if` block and its matching `else`.
- Add one blank line after an `else` block ends.
- For `if` blocks with no `else`, add one blank line after the `if` block ends.

#### Good

```python
if ready:
    start()
else:
    wait()

next_step()
```

```python
if ready:
    start()

next_step()
```

#### Bad

```python
if ready:
    start()

else:
    wait()
next_step()
```

```python
if ready:
    start()
next_step()
```

### Multi-argument function calls must use kwargs

- When a function call has multiple arguments, pass them as named keyword arguments.
- This applies to callbacks and helper calls like the `_render_response(...)` call near `cli.py`.

#### Good

```python
on_reasoning=lambda turn, parsed: _render_response(
    console=console,
    turn=turn,
    parsed=parsed,
    verbosity=args.verbosity,
)
```

#### Bad

```python
on_reasoning=lambda turn, parsed: _render_response(
    console,
    turn,
    parsed,
    args.verbosity,
)
```
