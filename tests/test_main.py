import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = PROJECT_ROOT / "core_agent.py"
CLI_PATH = PROJECT_ROOT / "cli.py"
PROMPT_FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "terminus-json-plain.txt"

core_spec = importlib.util.spec_from_file_location("core_agent", CORE_PATH)
assert core_spec and core_spec.loader
core_agent = importlib.util.module_from_spec(core_spec)
sys.modules["core_agent"] = core_agent
core_spec.loader.exec_module(core_agent)

summary_spec = importlib.util.spec_from_file_location(
    "final_summary", PROJECT_ROOT / "final_summary.py"
)
assert summary_spec and summary_spec.loader
final_summary = importlib.util.module_from_spec(summary_spec)
sys.modules["final_summary"] = final_summary
summary_spec.loader.exec_module(final_summary)

cli_spec = importlib.util.spec_from_file_location("cli", CLI_PATH)
assert cli_spec and cli_spec.loader
cli = importlib.util.module_from_spec(cli_spec)
sys.modules["cli"] = cli
cli_spec.loader.exec_module(cli)


class FakeChild:
    def __init__(self) -> None:
        self.before = ""
        self.send_calls: list[str] = []
        self.sendline_calls: list[str] = []
        self.sendcontrol_calls: list[str] = []
        self.expect_calls: list[tuple[str, float]] = []
        self.closed = False

    def send(self, value: str) -> None:
        self.send_calls.append(value)

    def sendline(self, value: str) -> None:
        self.sendline_calls.append(value)

    def sendcontrol(self, value: str) -> None:
        self.sendcontrol_calls.append(value)

    def expect_exact(self, sentinel: str, timeout: float) -> None:
        self.expect_calls.append((sentinel, timeout))

    def close(self, force: bool = False) -> None:
        self.closed = force


def _as_any(value: object) -> Any:
    return value


class TestPromptParity(unittest.TestCase):
    def test_system_prompt_matches_terminus2_template(self) -> None:
        expected = PROMPT_FIXTURE_PATH.read_text()
        self.assertEqual(core_agent.SYSTEM_PROMPT, expected)

    def test_build_prompt_matches_rendered_reference_template(self) -> None:
        expected_template = PROMPT_FIXTURE_PATH.read_text()
        instruction = "List files"
        terminal_state = "Current Terminal Screen:\n(empty)"
        expected = expected_template.format(
            instruction=instruction, terminal_state=terminal_state
        )
        actual = core_agent.build_prompt(
            instruction=instruction,
            terminal_state=terminal_state,
            max_wait_seconds=60.0,
        )
        self.assertEqual(actual, expected)

    def test_completion_confirmation_message_matches_terminus2(self) -> None:
        output = "line one\nline two"
        expected = (
            f"Current terminal state:\n{output}\n\n"
            "Are you sure you want to mark the task as complete? "
            "This will trigger your solution to be graded and you won't be able to "
            'make any further corrections. If so, include "task_complete": true '
            "in your JSON response again."
        )
        self.assertEqual(core_agent.completion_confirmation_message(output), expected)


class TestParserParity(unittest.TestCase):
    def test_extract_json_content_handles_wrapped_text(self) -> None:
        response = 'prefix text {"a": 1, "nested": {"b": 2}} suffix text'
        self.assertEqual(
            core_agent.extract_json_content(response), '{"a": 1, "nested": {"b": 2}}'
        )

    def test_extract_json_content_handles_escaped_quote_and_brace(self) -> None:
        response = 'noise {"text": "quote \\" and brace }", "ok": true} trailing'
        self.assertEqual(
            core_agent.extract_json_content(response),
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
        parsed = core_agent.parse_response(text)
        self.assertTrue(parsed.task_complete)
        self.assertEqual(parsed.commands[0].duration, 1.0)

    def test_parse_response_reports_missing_fields_in_single_error(self) -> None:
        text = """
        {
          "analysis": "a",
          "commands": []
        }
        """
        with self.assertRaisesRegex(ValueError, "Missing required fields: plan"):
            core_agent.parse_response(text)

    def test_parse_response_invalid_duration_falls_back_to_default(self) -> None:
        text = """
        {
          "analysis": "a",
          "plan": "p",
          "commands": [{"keystrokes": "ls\\n", "duration": "slow"}]
        }
        """
        parsed = core_agent.parse_response(text)
        self.assertEqual(parsed.commands[0].duration, 1.0)

    def test_parse_response_tolerates_bad_commands_when_task_complete(self) -> None:
        text = """
        {
          "analysis": "a",
          "plan": "p",
          "commands": [{"duration": 1.0}],
          "task_complete": true
        }
        """
        parsed = core_agent.parse_response(text)
        self.assertTrue(parsed.task_complete)
        self.assertEqual(parsed.commands, [])

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
            core_agent.parse_response(text)


class TestExecutionAndLoop(unittest.TestCase):
    def test_limit_output_length_short_output_unchanged(self) -> None:
        text = "hello"
        self.assertEqual(core_agent.limit_output_length(text, max_bytes=20), text)

    def test_limit_output_length_long_output_contains_marker(self) -> None:
        text = "abcdef" * 200
        limited = core_agent.limit_output_length(text, max_bytes=40)
        self.assertIn("output limited to 40 bytes", limited)
        self.assertTrue(limited.startswith("abcdef"))
        self.assertTrue(limited.endswith("abcdef"))

    def test_execute_command_wait_keystrokes_only_sleeps(self) -> None:
        child = FakeChild()
        cmd = core_agent.Command(keystrokes="", duration=0.25)
        with patch("time.sleep") as sleep_mock:
            output = core_agent.execute_command(
                child=_as_any(child),
                cmd=cmd,
                max_wait_seconds=60.0,
            )
        self.assertEqual(output, "")
        sleep_mock.assert_called_once_with(0.25)
        self.assertEqual(child.send_calls, [])
        self.assertEqual(child.sendline_calls, [])
        self.assertEqual(child.sendcontrol_calls, [])

    def test_execute_command_single_newline_uses_sendline(self) -> None:
        child = FakeChild()
        child.before = "ok"
        cmd = core_agent.Command(keystrokes="echo hi\n", duration=0.1)
        output = core_agent.execute_command(
            child=_as_any(child),
            cmd=cmd,
            max_wait_seconds=60.0,
        )
        self.assertEqual(child.sendline_calls, ["echo hi"])
        self.assertEqual(child.send_calls, [])
        self.assertEqual(output, "ok")

    def test_run_agent_uses_terminus2_parse_error_feedback_and_confirmation(
        self,
    ) -> None:
        cfg = core_agent.Config(
            active_model_key="test",
            active_model=core_agent.ModelConfig(model="x", api_base="y"),
            max_turns=6,
        )
        child = FakeChild()
        prompts: list[str] = []
        responses = iter(
            [
                "not json",
                '{"analysis":"a","plan":"p","commands":[],"task_complete":true}',
                '{"analysis":"a2","plan":"p2","commands":[],"task_complete":true}',
                "post-run summary",
            ]
        )

        def fake_call_model(
            cfg: Any,
            prompt: str,
            history: list[dict[str, str]],
            api_key: str,
        ) -> str:
            del cfg, history, api_key
            prompts.append(prompt)
            return next(responses)

        with (
            patch.object(core_agent, "start_shell", return_value=child),
            patch.object(core_agent, "call_model", side_effect=fake_call_model),
        ):
            exit_code = core_agent.run_agent(
                instruction="do thing",
                cfg=cfg,
                api_key="k",
            )

        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(len(prompts), 4)
        self.assertIn(
            "Please fix these issues and provide a proper JSON response.", prompts[1]
        )
        self.assertEqual(
            prompts[2], core_agent.completion_confirmation_message("[no new output]")
        )
        self.assertEqual(prompts[3], final_summary.post_run_summary_prompt())
        self.assertTrue(child.closed)


class TestFinalSummaryNormalization(unittest.TestCase):
    def test_normalize_summary_response_prefers_final_message_from_json(self) -> None:
        normalized = final_summary.normalize_summary_response(
            '{"analysis":"a","plan":"p","commands":[],"final_message":"Done cleanly."}'
        )
        self.assertEqual(normalized, "Done cleanly.")

    def test_normalize_summary_response_strips_code_fences(self) -> None:
        normalized = final_summary.normalize_summary_response(
            "```markdown\nCompleted successfully.\n```"
        )
        self.assertEqual(normalized, "Completed successfully.")


class TestResolveApiKey(unittest.TestCase):
    def test_resolve_api_key_prefers_literal_config_key(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}, clear=True):
            self.assertEqual(cli.resolve_api_key("literal-key"), "literal-key")

    def test_resolve_api_key_uses_dollar_variable(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "expanded-key"}, clear=True):
            self.assertEqual(cli.resolve_api_key("$OPENAI_API_KEY"), "expanded-key")

    def test_resolve_api_key_uses_variable_name_without_dollar(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "named-key"}, clear=True):
            self.assertEqual(cli.resolve_api_key("OPENAI_API_KEY"), "named-key")

    def test_resolve_api_key_falls_back_to_subprocess_for_env_var(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("subprocess.run") as run_mock:
                run_mock.return_value = subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="from-zsh",
                    stderr="",
                )
                self.assertEqual(cli.resolve_api_key("OPENAI_API_KEY"), "from-zsh")


class TestModelSelection(unittest.TestCase):
    def test_resolve_model_key_uses_cli_model_when_provided(self) -> None:
        loaded = cli.LoadedConfig(
            default_model="a",
            models={
                "a": cli.ConfigModelEntry(
                    model="model-a",
                    api_base="https://example.com/v1",
                    api_key="KEY_A",
                    temperature=0.7,
                ),
                "b": cli.ConfigModelEntry(
                    model="model-b",
                    api_base="https://example.com/v1",
                    api_key="KEY_B",
                    temperature=0.7,
                ),
            },
            verbosity=1,
            max_turns=5,
            max_wait_seconds=10.0,
        )
        self.assertEqual(
            cli.resolve_model_key(
                config=loaded, cli_model_key="b", selected_model_key=None
            ),
            "b",
        )

    def test_resolve_model_key_raises_for_unknown_cli_model(self) -> None:
        loaded = cli.LoadedConfig(
            default_model="a",
            models={
                "a": cli.ConfigModelEntry(
                    model="model-a",
                    api_base="https://example.com/v1",
                    api_key="KEY_A",
                    temperature=0.7,
                )
            },
            verbosity=1,
            max_turns=5,
            max_wait_seconds=10.0,
        )
        with self.assertRaisesRegex(ValueError, "Unknown model key"):
            cli.resolve_model_key(
                config=loaded, cli_model_key="missing", selected_model_key=None
            )


class TestInteractiveCommands(unittest.TestCase):
    def _loaded_config(self) -> Any:
        return cli.LoadedConfig(
            default_model="a",
            models={
                "a": cli.ConfigModelEntry(
                    model="model-a",
                    api_base="https://example.com/v1",
                    api_key="KEY_A",
                    temperature=0.7,
                )
            },
            verbosity=1,
            max_turns=5,
            max_wait_seconds=10.0,
        )

    def test_parse_interactive_command_sets_verbosity_inline(self) -> None:
        loaded = self._loaded_config()
        result = cli.parse_interactive_command(
            console=cli.Console(record=True),
            instruction="/verbosity 3",
            config=loaded,
        )
        self.assertTrue(result.handled)
        self.assertEqual(result.updated_verbosity, 3)
        self.assertEqual(result.instruction, "")

    def test_parse_interactive_command_prompts_for_missing_verbosity(self) -> None:
        loaded = self._loaded_config()
        with patch.object(cli.Prompt, "ask", return_value="1"):
            result = cli.parse_interactive_command(
                console=cli.Console(record=True),
                instruction="/verbosity",
                config=loaded,
            )
        self.assertTrue(result.handled)
        self.assertEqual(result.updated_verbosity, 1)

    def test_parse_interactive_command_sets_max_turns(self) -> None:
        loaded = self._loaded_config()
        result = cli.parse_interactive_command(
            console=cli.Console(record=True),
            instruction="/max_turns 12",
            config=loaded,
        )
        self.assertTrue(result.handled)
        self.assertEqual(loaded.max_turns, 12)
        self.assertIsNone(result.updated_verbosity)

    def test_parse_interactive_command_sets_max_wait_seconds(self) -> None:
        loaded = self._loaded_config()
        with patch.object(cli.Prompt, "ask", return_value="2.5"):
            result = cli.parse_interactive_command(
                console=cli.Console(record=True),
                instruction="/max_wait_seconds",
                config=loaded,
            )
        self.assertTrue(result.handled)
        self.assertAlmostEqual(loaded.max_wait_seconds, 2.5)

    def test_parse_interactive_command_unknown_slash_command_is_handled(self) -> None:
        loaded = self._loaded_config()
        result = cli.parse_interactive_command(
            console=cli.Console(record=True),
            instruction="/unknown",
            config=loaded,
        )
        self.assertTrue(result.handled)
        self.assertEqual(result.instruction, "")

    def test_parse_interactive_command_empty_input_shows_help_and_is_handled(
        self,
    ) -> None:
        loaded = self._loaded_config()
        result = cli.parse_interactive_command(
            console=cli.Console(record=True),
            instruction="",
            config=loaded,
        )
        self.assertTrue(result.handled)
        self.assertEqual(result.instruction, "")


if __name__ == "__main__":
    unittest.main()
