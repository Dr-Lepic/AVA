"""CLI streaming and the REPL loop.

Two things pinned here that only a real streaming run reveals, both found by
probing the SDK before writing the CLI:

1. `Runner.run_streamed(...)` is NOT awaitable. Awaiting it raises
   "object RunResultStreaming can't be used in 'await' expression".

2. `raw_response_event.data.delta` carries tool-call argument JSON as well as
   assistant text. A naive implementation that prints every delta shows the
   user: {"input":"milk"} mid-sentence. Deltas must only be printed once the
   run is producing message output, or the tool JSON must be suppressed.
"""

import asyncio

import pytest
from agents import Agent
from agents.testing import ScriptedModel, assistant_message, function_call

from ava import cli


class FakeEvent:
    def __init__(self, type_, **kw):
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


class FakeDelta:
    def __init__(self, delta, chunk_type="response.output_text.delta"):
        self.delta = delta
        self.type = chunk_type


def _raw(text, chunk_type=cli.OUTPUT_TEXT_DELTA):
    return FakeEvent("raw_response_event", data=FakeDelta(text, chunk_type))


def _item(item):
    return FakeEvent("run_item_stream_event", item=item, name=getattr(item, "name", ""))


def _install_stream(monkeypatch, events, final="done."):
    """Patch Runner.run_streamed to yield `events`."""

    class FakeResult:
        final_output = final
        last_agent = type("A", (), {"name": "AVA"})()
        raw_responses = [1]

        async def stream_events(self):
            for e in events:
                yield e

    def fake_run_streamed(*args, **kwargs):
        return FakeResult()

    monkeypatch.setattr(cli.Runner, "run_streamed", fake_run_streamed)


# --- tool-call formatting ------------------------------------------------


def test_format_tool_call_summarises_args():
    item = FakeEvent("x", name="save_note")
    item.raw_item = type("R", (), {"name": "save_note", "arguments": '{"title": "x"}'})()
    line = cli._format_tool_call(item)
    assert "save_note" in line
    assert "title" in line


def test_format_tool_call_handles_bad_json():
    item = FakeEvent("x")
    item.raw_item = type("R", (), {"name": "t", "arguments": "not json"})()
    assert "t(" in cli._format_tool_call(item)


def test_format_tool_call_ignores_non_tool_items():
    assert cli._format_tool_call(FakeEvent("x")) is None


def test_format_tool_call_truncates_many_args():
    item = FakeEvent("x")
    item.raw_item = type(
        "R", (), {"name": "t", "arguments": '{"a1":1,"a2":2,"a3":3,"a4":4,"a5":5}'}
    )()
    assert cli._format_tool_call(item).count("=") == 3


def test_conversation_goes_to_stdout_and_tools_to_stderr(capsys):
    """Diagnostics must not pollute piped stdout."""
    cli.write_text("hello")
    assert "hello" in capsys.readouterr().out

    cli.write_diagnostic("  · save_note()")
    captured = capsys.readouterr()
    assert "save_note" in captured.err
    assert "save_note" not in captured.out


# --- reasoning-leak suppression (the second real bug) --------------------


def test_reasoning_deltas_are_not_output_text():
    """A reasoning model streams its scratchpad through the same event type."""
    event = FakeEvent(
        "raw_response_event",
        data=FakeDelta(" — the user wants a note", "response.reasoning_text.delta"),
    )
    assert cli._is_output_text(event) is False


def test_output_text_deltas_are_shown():
    event = FakeEvent(
        "raw_response_event", data=FakeDelta("Here you go", "response.output_text.delta")
    )
    assert cli._is_output_text(event) is True


def test_unlabelled_chunk_is_shown():
    """A provider that does not label chunks must not lose its output."""
    event = FakeEvent("raw_response_event", data=FakeDelta("text", None))
    assert cli._is_output_text(event) is True


async def test_reasoning_text_never_reaches_the_user(monkeypatch, capsys):
    _install_stream(
        monkeypatch,
        [
            _raw("I should hand off to Notes.", "response.reasoning_text.delta"),
            _raw("Saved your wifi password."),
        ],
        final="Saved your wifi password.",
    )
    await cli._respond(_agent(), None, "remember it", None)
    out = capsys.readouterr().out
    assert "Saved your wifi password." in out
    assert "I should hand off" not in out


# --- tool-json detection (the real fix) ---------------------------------


def test_tool_json_delta_is_detected():
    assert cli._looks_like_tool_json('{"input":"milk"}') is True


def test_assistant_text_is_not_tool_json():
    assert cli._looks_like_tool_json("Noted.") is False
    assert cli._looks_like_tool_json("Here you go: ") is False
    assert cli._looks_like_tool_json('{"title": "x"}') is True


def test_empty_delta_is_not_tool_json():
    assert cli._looks_like_tool_json("") is False
    assert cli._looks_like_tool_json("   ") is False


# --- streaming -----------------------------------------------------------


async def test_streams_assistant_text(monkeypatch, capsys):
    _install_stream(monkeypatch, [_raw("Hello "), _raw("there.")], final="Hello there.")
    await cli._respond(_agent(), None, "hi", None)
    assert "Hello there." in capsys.readouterr().out


async def test_does_not_print_tool_json(monkeypatch, capsys):
    """The bug this file exists for: delta carries tool-call JSON too."""
    _install_stream(
        monkeypatch,
        [_raw('{"input":"milk"}'), _raw("Noted.")],
        final="Noted.",
    )
    await cli._respond(_agent(), None, "remember milk", None)
    printed = capsys.readouterr().out
    assert "Noted." in printed
    assert '"input"' not in printed


async def test_reports_when_a_specialist_handled_it(monkeypatch, capsys):
    class SpecialistResult:
        final_output = "ok"
        last_agent = type("A", (), {"name": "Notes"})()
        raw_responses = [1, 2]

        async def stream_events(self):
            yield _raw("ok")

    monkeypatch.setattr(
        cli.Runner, "run_streamed", staticmethod(lambda *a, **k: SpecialistResult())
    )
    await cli._respond(_agent(), None, "note this", None)
    # The specialist line is a diagnostic, so it goes to stderr by design.
    assert "Notes" in capsys.readouterr().err


async def test_returns_call_count_and_summary(monkeypatch):
    _install_stream(monkeypatch, [_raw("ok")], final="ok")
    calls, summary = await cli._respond(_agent(), None, "hi", None)
    assert isinstance(calls, int) and calls >= 1
    assert isinstance(summary, str)


def _agent():
    return Agent(name="AVA", model=ScriptedModel([]), instructions="x")


# --- loop exit paths -----------------------------------------------------


async def test_quit_exits_cleanly(monkeypatch):
    monkeypatch.setattr(cli, "build_agent", lambda *a, **k: None)
    monkeypatch.setattr(cli, "configure_llm", lambda *a, **k: None)
    monkeypatch.setattr(cli, "build_model", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_report_overdue", lambda *a, **k: None)

    async def fake_input(*a, **k):
        return "/quit"

    monkeypatch.setattr(asyncio, "to_thread", fake_input)
    assert await cli._run(_fake_config()) == 0


async def test_eof_exits_cleanly(monkeypatch):
    """Piped input runs out; the loop must exit, not hang."""
    monkeypatch.setattr(cli, "build_agent", lambda *a, **k: None)
    monkeypatch.setattr(cli, "configure_llm", lambda *a, **k: None)
    monkeypatch.setattr(cli, "build_model", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_report_overdue", lambda *a, **k: None)

    async def fake_input(*a, **k):
        raise EOFError

    monkeypatch.setattr(asyncio, "to_thread", fake_input)
    assert await cli._run(_fake_config()) == 0


async def test_ctrl_c_at_prompt_exits(monkeypatch):
    monkeypatch.setattr(cli, "build_agent", lambda *a, **k: None)
    monkeypatch.setattr(cli, "configure_llm", lambda *a, **k: None)
    monkeypatch.setattr(cli, "build_model", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_report_overdue", lambda *a, **k: None)

    async def fake_input(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(asyncio, "to_thread", fake_input)
    assert await cli._run(_fake_config()) == 0


async def test_blank_line_is_ignored(monkeypatch):
    monkeypatch.setattr(cli, "build_agent", lambda *a, **k: None)
    monkeypatch.setattr(cli, "configure_llm", lambda *a, **k: None)
    monkeypatch.setattr(cli, "build_model", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_report_overdue", lambda *a, **k: None)

    replies = iter(["", "   ", "/quit"])

    async def fake_input(*a, **k):
        return next(replies)

    monkeypatch.setattr(asyncio, "to_thread", fake_input)
    assert await cli._run(_fake_config()) == 0


def _fake_config():
    """Mirrors the real Config: db_path, vault_path, and state_path are Paths."""
    from pathlib import Path

    class FakeConfig:
        db_path = Path("/tmp/ava-cli-fake.db")
        vault_path = Path("/tmp/ava-cli-fake-vault")
        state_path = Path("/tmp/ava-cli-fake.json")
        model = "openrouter/free"
        api_key = "sk-or-v1-fake"
        tz = "Asia/Dhaka"

    return FakeConfig()


# --- config errors -------------------------------------------------------


def test_main_reports_missing_key(monkeypatch, capsys):
    from ava.config import ConfigError

    def boom():
        raise ConfigError("OPENROUTER_API_KEY is not set.")

    monkeypatch.setattr(cli, "load_config", boom)
    assert cli.main() == 1
    # Diagnostics deliberately go to stderr so stdout stays pipeable.
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_banner_mentions_quit():
    assert "/quit" in cli.BANNER
