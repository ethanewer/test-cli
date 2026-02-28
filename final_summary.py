from __future__ import annotations

import json
import re
from typing import Any, Protocol


class CallModelFn(Protocol):
    def __call__(
        self,
        cfg: Any,
        prompt: str,
        history: list[dict[str, str]],
        api_key: str,
    ) -> str: ...


def post_run_summary_prompt() -> str:
    return (
        "The task is complete. Write the final user-facing summary as plain text or "
        "Markdown only.\n\n"
        "Requirements:\n"
        "- Do not use JSON\n"
        "- Do not wrap the response in code fences\n"
        "- Keep it concise and outcome-focused\n"
        "- Mention any caveats or follow-up steps if needed"
    )


def build_done_text(
    call_model_fn: CallModelFn,
    cfg: Any,
    history: list[dict[str, str]],
    api_key: str,
    pending_final_message: str | None,
) -> str:
    response: str | None = None
    try:
        response = call_model_fn(
            cfg=cfg,
            prompt=post_run_summary_prompt(),
            history=history,
            api_key=api_key,
        )
    except Exception:
        response = None

    normalized = normalize_summary_response(response or "")
    if normalized:
        return normalized
    if pending_final_message:
        return pending_final_message
    return "Task marked complete (double-confirmed)."


def normalize_summary_response(raw_response: str) -> str | None:
    text = _strip_code_fences(raw_response.strip())
    if not text:
        return None

    dict_payload = _extract_json_dict(text)
    if dict_payload is not None:
        for key in ("final_message", "message", "summary"):
            value = dict_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        analysis = dict_payload.get("analysis")
        if isinstance(analysis, str) and analysis.strip():
            return analysis.strip()

        return None

    return text


def _strip_code_fences(text: str) -> str:
    if not text.startswith("```"):
        return text

    fence_pattern = re.compile(r"^```[a-zA-Z0-9_-]*\n?(.*?)\n?```$", re.DOTALL)
    match = fence_pattern.match(text)
    if not match:
        return text
    return match.group(1).strip()


def _extract_json_dict(text: str) -> dict[str, Any] | None:
    parsed = _try_parse_json(text)
    if isinstance(parsed, dict):
        return parsed

    if text.startswith("{") and text.endswith("}"):
        return None

    candidate = _first_json_object(text)
    if not candidate:
        return None
    parsed = _try_parse_json(candidate)
    return parsed if isinstance(parsed, dict) else None


def _try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _first_json_object(text: str) -> str | None:
    start = -1
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            if depth == 0:
                start = index

            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : index + 1]

    return None
