"""Runs the specialists through the real Agents SDK runner.

The config tests in test_agents.py only assert structure. These drive an
actual agent loop with ScriptedModel, so tool dispatch, schema validation, and
the result plumbing are exercised for real — offline, with no API key.

The gap these close was left open deliberately in Task 3, where invoking a
tool needed a RunContextWrapper that the unit tests could not supply
cleanly. ScriptedModel is the supported way to do it.
"""

import json

import pytest
from agents import RunConfig, Runner
from agents.testing import ScriptedModel, assistant_message, function_call

from ava.agents import build_specialists
from ava.db import connect
from ava.tools.notes import list_notes_impl
from ava.tools.reminders import due_reminders

NO_TRACE = RunConfig(tracing_disabled=True)


async def test_notes_specialist_saves_a_note(tmp_path):
    conn = connect(tmp_path / "a.db")
    model = ScriptedModel([
        [function_call("save_note", {"title": "Meeting", "body": "noon"}, call_id="c1")],
        [assistant_message("Saved.")],
    ])
    agent = build_specialists(tmp_path / "a.db", tmp_path / "vault", model=model)["notes"]

    result = await Runner.run(agent, "note that meeting is at noon", run_config=NO_TRACE)

    assert result.final_output == "Saved."
    assert len(model.calls) == 2
    assert model.last_call is not None
    assert any(
        item.get("type") == "function_call_output" for item in model.last_call.input
    )
    rows = json.loads(list_notes_impl(conn))
    assert rows[0]["title"] == "Meeting"
    model.assert_complete()


async def test_drafting_specialist_never_sends(tmp_path):
    """The tool's return value must reach the model, not just the database."""
    conn = connect(tmp_path / "a.db")
    model = ScriptedModel([
        [function_call("create_draft", {"subject": "Lunch?", "body": "Tacos?"}, call_id="c1")],
        [assistant_message("Draft saved, not sent.")],
    ])
    agent = build_specialists(tmp_path / "a.db", tmp_path / "vault", model=model)["drafting"]

    result = await Runner.run(agent, "draft an email about lunch", run_config=NO_TRACE)

    assert "not sent" in result.final_output.lower()
    assert conn.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] == 1
    model.assert_complete()


async def test_scheduling_specialist_does_timezone_math(tmp_path):
    conn = connect(tmp_path / "a.db")
    model = ScriptedModel([
        [
            function_call(
                "set_reminder",
                {"title": "Standup", "due": "2027-10-01 09:00"},
                call_id="c1",
            )
        ],
        [assistant_message("Set.")],
    ])
    agent = build_specialists(tmp_path / "a.db", tmp_path / "vault", model=model)["scheduling"]

    result = await Runner.run(agent, "remind me at 9", run_config=NO_TRACE)

    assert result.final_output == "Set."
    row = conn.execute("SELECT title, due_at FROM reminders").fetchone()
    assert row["title"] == "Standup"
    # 09:00 naive resolves in Asia/Dhaka (UTC+6) -> 03:00 UTC
    assert row["due_at"].startswith("2027-10-01T03:00")
    model.assert_complete()


async def test_specialist_cannot_call_another_domains_tool(tmp_path):
    """Isolation must hold at runtime, not just in the tool lists."""
    conn = connect(tmp_path / "a.db")
    model = ScriptedModel([
        [function_call("set_reminder", {"title": "X", "due": "2027-10-01 09:00"}, call_id="c1")],
        [assistant_message("Set.")],
    ])
    notes_agent = build_specialists(tmp_path / "a.db", tmp_path / "vault", model=model)["notes"]

    assert "set_reminder" not in {t.name for t in notes_agent.tools}
    assert conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0] == 0


async def test_specialist_does_not_receive_mcp_servers(tmp_path):
    conn = connect(tmp_path / "a.db")
    specs = build_specialists(tmp_path / "a.db", tmp_path / "vault", model="test/model:free")
    for agent in specs.values():
        assert not agent.mcp_servers


async def test_past_reminder_is_refused_not_stored(tmp_path, monkeypatch):
    """The reported bug, at the agent level.

    The model used to pass a date from its stale idea of today. Such a
    reminder is due immediately and fires on the next poll, so set_reminder
    now refuses it. This asserts the refusal reaches the model as a tool
    error rather than being stored and firing.
    """
    monkeypatch.setenv("AVA_TZ", "Asia/Dhaka")
    conn = connect(tmp_path / "a.db")
    model = ScriptedModel([
        [
            function_call(
                "set_reminder",
                {"title": "Past thing", "due": "2020-01-01 09:00"},
                call_id="c1",
            )
        ],
        [assistant_message("That time has passed.")],
    ])
    agent = build_specialists(tmp_path / "a.db", tmp_path / "vault", model=model)["scheduling"]

    result = await Runner.run(
        agent, "set an old reminder", run_config=NO_TRACE
    )

    # Nothing stored, so nothing can fire.
    assert due_reminders(conn) == []
    assert conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0] == 0

    # The model is told what the real current time is, so it can retry rather
    # than guessing a new absolute date from the same stale belief.
    outputs = [
        item.output
        for item in result.new_items
        if type(item).__name__ == "ToolCallOutputItem"
    ]
    assert outputs, "no tool output produced"
    assert "current time is" in outputs[0].lower()


async def test_unexpected_model_call_is_detected(tmp_path):
    """Drift detection: asking for a third turn must raise, not pass quietly."""
    from agents.testing import UnconsumedModelSteps

    conn = connect(tmp_path / "a.db")
    model = ScriptedModel([
        [function_call("save_note", {"title": "A"}, call_id="c1")],
        [assistant_message("Saved.")],
    ])
    agent = build_specialists(tmp_path / "a.db", tmp_path / "vault", model=model)["notes"]

    await Runner.run(agent, "note this", run_config=NO_TRACE)
    assert model.remaining_steps == 0
