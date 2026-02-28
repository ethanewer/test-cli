import subprocess
import unittest
from typing import Any
from unittest.mock import patch

import main


class FakeChild:
    def __init__(self) -> None:
        self.before = ""
        self.send_calls: list[str] = []
        self.sendline_calls: list[str] = []
        self.sendcontrol_calls: list[str] = []
        self.expect_calls: list[tuple[str, float]] = []

    def send(self, value: str) -> None:
        self.send_calls.append(value)

    def sendline(self, value: str) -> None:
        self.sendline_calls.append(value)

    def sendcontrol(self, value: str) -> None:
        self.sendcontrol_calls.append(value)

    def expect_exact(self, sentinel: str, timeout: float) -> None:
        self.expect_calls.append((sentinel, timeout))


def _as_any(value: object) -> Any:
    """Type-only helper so test doubles can satisfy strict annotations."""
    return value


class TestParsingAndExtraction(unittest.TestCase):
    def test_extract_json_content_handles_wrapped_text(self) -> None:
        response = 'prefix text {"a": 1, "nested": {"b": 2}} suffix text'
        self.assertEqual(
            main.extract_json_content(response), '{"a": 1, "nested": {"b": 2}}'
        )

    def test_extract_json_content_handles_escaped_quote_and_brace(self) -> None:
        response = 'noise {"text": "quote \\" and brace }", "ok": true} trailing'
        self.assertEqual(
            main.extract_json_content(response),
            '{"text": "quote \\" and brace }", "ok": true}',
        )

    def test_parse_response_coerces_task_complete_string(self) -> None:
        text = """
        {
          "analysis": "a",
          "plan": "p",
          "commands": [{"keystrokes": "ls\\n"}],
          "task_complete": "yes"
        }
        """
        parsed = main.parse_response(text)
        self.assertTrue(parsed.task_complete)
        self.assertEqual(parsed.commands[0].duration, 1.0)

    def test_parse_response_rejects_invalid_final_message_type(self) -> None:
        text = """
        {
          "analysis": "a",
          "plan": "p",
          "commands": [{"keystrokes": "ls\\n"}],
          "final_message": 42
        }
        """
        with self.assertRaises(ValueError):
            main.parse_response(text)


class TestOutputNormalization(unittest.TestCase):
    def test_normalize_command_output_filters_prompt_and_echo(self) -> None:
        cmd = main.Command(keystrokes="pwd\n", duration=0.1)
        raw = "\n".join(
            [
                main.PROMPT_SENTINEL.strip(),
                "pwd",
                "% zsh artifact",
                "'",
                '"',
                f"{main.PROMPT_SENTINEL} /tmp",
                "/Users/example",
                "",
            ]
        )
        normalized = main.normalize_command_output(raw, cmd)
        self.assertEqual(normalized, "/tmp\n/Users/example")

    def test_limit_output_length_short_output_unchanged(self) -> None:
        text = "hello"
        self.assertEqual(main.limit_output_length(text, max_bytes=20), text)

    def test_limit_output_length_long_output_contains_marker(self) -> None:
        text = "abcdef" * 200
        limited = main.limit_output_length(text, max_bytes=40)
        self.assertIn("output limited to 40 bytes", limited)
        self.assertTrue(limited.startswith("abcdef"))
        self.assertTrue(limited.endswith("abcdef"))


class TestResolveApiKey(unittest.TestCase):
    def test_resolve_api_key_prefers_literal_config_key(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-key"}, clear=True):
            self.assertEqual(main.resolve_api_key("literal-key"), "literal-key")

    def test_resolve_api_key_uses_expanded_variable(self) -> None:
        with patch.dict(
            "os.environ", {"OPENROUTER_API_KEY": "expanded-key"}, clear=True
        ):
            self.assertEqual(
                main.resolve_api_key("$OPENROUTER_API_KEY"), "expanded-key"
            )

    def test_resolve_api_key_falls_back_to_subprocess(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("subprocess.run") as run_mock:
                run_mock.return_value = subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="from-zsh",
                    stderr="",
                )
                self.assertEqual(main.resolve_api_key(None), "from-zsh")


class TestExecuteCommand(unittest.TestCase):
    def test_execute_command_wait_keystrokes_only_sleeps(self) -> None:
        child = FakeChild()
        cmd = main.Command(keystrokes="", duration=0.25)
        with patch("time.sleep") as sleep_mock:
            output = main.execute_command(_as_any(child), cmd)
        self.assertEqual(output, "")
        sleep_mock.assert_called_once_with(0.25)
        self.assertEqual(child.send_calls, [])
        self.assertEqual(child.sendline_calls, [])
        self.assertEqual(child.sendcontrol_calls, [])

    def test_execute_command_ctrl_c_path(self) -> None:
        child = FakeChild()
        child.before = "result"
        cmd = main.Command(keystrokes="C-c", duration=0.1)
        output = main.execute_command(_as_any(child), cmd)
        self.assertEqual(child.sendcontrol_calls, ["c"])
        self.assertEqual(output, "result")

    def test_execute_command_ctrl_d_path(self) -> None:
        child = FakeChild()
        child.before = "result"
        cmd = main.Command(keystrokes="C-d", duration=0.1)
        output = main.execute_command(_as_any(child), cmd)
        self.assertEqual(child.sendcontrol_calls, ["d"])
        self.assertEqual(output, "result")

    def test_execute_command_single_newline_uses_sendline(self) -> None:
        child = FakeChild()
        child.before = "ok"
        cmd = main.Command(keystrokes="echo hi\n", duration=0.1)
        output = main.execute_command(_as_any(child), cmd)
        self.assertEqual(child.sendline_calls, ["echo hi"])
        self.assertEqual(child.send_calls, [])
        self.assertEqual(output, "ok")

    def test_execute_command_multiline_uses_send(self) -> None:
        child = FakeChild()
        child.before = "done"
        cmd = main.Command(keystrokes="echo one\necho two\n", duration=0.1)
        output = main.execute_command(_as_any(child), cmd)
        self.assertEqual(child.send_calls, ["echo one\necho two\n"])
        self.assertEqual(child.sendline_calls, [])
        self.assertEqual(output, "done")


if __name__ == "__main__":
    unittest.main()
