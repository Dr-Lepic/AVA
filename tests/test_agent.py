"""The coordinator agent: handoffs plus agents-as-tools.

The distinction under test:

- handoff: the specialist takes over the turn and answers the user directly
- agents-as-tool: the coordinator stays in charge, calls a specialist for a
  bounded job, then narrates the result

An agent-as-tool takes a single free-text `input` (verified against the SDK),
not the specialist's own tool schema. The coordinator describes the job in
prose and the specialist decides which of its tools to call.
"""

import json

import pytest
from agents import RunConfig, Runner
from agents.testing import ScriptedModel, assistant_message, function_call

from ava.agent import build_agent
from ava.agents import build_specialists
from ava.db import connect
from ava.tools.notes import list_notes_impl

NO_TRACE = RunConfig(tracing_disabled=True)

SPECIALIST_NAMES = {"Notes", "Drafting", "Scheduling"}


# --- structure -----------------------------------------------------------


def test_coordinator_has_no_domain_tools(tmp_path):
    """If the coordinator had domain tools, routing would be pointless."""
    agent = build_agent(tmp_path / "a.db", tmp_path / "vault", mcp_servers=[])
    names = {t.name for t in agent.tools}
    assert "save_note" not in names
    assert "set_reminder" not in names
    assert "create_draft" not in names


def test_coordinator_hands_off_to_all_three(tmp_path):
    agent = build_agent(tmp_path / "a.db", tmp_path / "vault", mcp_servers=[])
    assert {h.name for h in agent.handoffs} == SPECIALIST_NAMES


def test_coordinator_exposes_drafting_and_notes_as_tools(tmp_path):
    agent = build_agent(tmp_path / "a.db", tmp_path / "vault", mcp_servers=[])
    names = {t.name for t in agent.tools}
    assert "draft_email" in names
    assert "take_note" in names


def test_scheduling_is_not_exposed_as_a_tool(tmp_path):
    """Handoff-only: calling it as a tool would bypass timezone reasoning."""
    agent = build_agent(tmp_path / "a.db", tmp_path / "vault", mcp_servers=[])
    assert not any("schedul" in t.name or "reminder" in t.name for t in agent.tools)


def test_agent_as_tool_takes_free_text_input(tmp_path):
    """Verified SDK behaviour: a single `input` string, not the inner schema."""
    agent = build_agent(tmp_path / "a.db", tmp_path / "vault", mcp_servers=[])
    tool = next(t for t in agent.tools if t.name == "take_note")
    assert set(tool.params_json_schema["properties"]) == {"input"}


def test_agent_as_tool_descriptions_mention_safety(tmp_path):
    agent = build_agent(tmp_path / "a.db", tmp_path / "vault", mcp_servers=[])
    tool = next(t for t in agent.tools if t.name == "draft_email")
    assert "not send" in (tool.description or "").lower()


def test_coordinator_instructions_cover_routing(tmp_path):
    agent = build_agent(tmp_path / "a.db", tmp_path / "vault", mcp_servers=[])
    text = (agent.instructions or "").lower()
    assert "hand off" in text
    assert "small talk" in text or "trivial" in text


# --- runtime: the coordinator calls a specialist as a tool ---------------


async def test_agents_as_tool_runs_the_nested_agent(tmp_path):
    """The nested run must actually execute the specialist's own tool."""
    model = ScriptedModel([
        [function_call("take_note", {"input": "remember to buy milk"}, call_id="c1")],
        [function_call("save_note", {"title": "Buy milk", "body": ""}, call_id="c2")],
        [assistant_message("Inner agent finished.")],
        [assistant_message("Noted — buy milk.")],
    ])
    agent = build_agent(tmp_path / "a.db", tmp_path / "vault", mcp_servers=[], model=model)

    result = await Runner.run(agent, "remember to buy milk", run_config=NO_TRACE)

    assert result.final_output == "Noted — buy milk."
    conn = connect(tmp_path / "a.db")
    rows = json.loads(list_notes_impl(conn))
    assert rows[0]["title"] == "Buy milk"
    model.assert_complete()


async def test_coordinator_narrates_after_a_specialist_call(tmp_path):
    """The coordinator stays in charge: it speaks again after the nested run."""
    model = ScriptedModel([
        [function_call("take_note", {"input": "note this"}, call_id="c1")],
        [function_call("save_note", {"title": "X", "body": ""}, call_id="c2")],
        [assistant_message("Inner done.")],
        [assistant_message("All set.")],
    ])
    agent = build_agent(tmp_path / "a.db", tmp_path / "vault", mcp_servers=[], model=model)

    result = await Runner.run(agent, "note this", run_config=NO_TRACE)

    # Four calls: coordinator -> specialist entry -> specialist tool ->
    # coordinator reply. The last output is the coordinator's.
    assert len(model.calls) == 4
    assert result.final_output == "All set."
    model.assert_complete()


async def test_drafting_as_tool_never_sends(tmp_path):
    model = ScriptedModel([
        [function_call("draft_email", {"input": "draft about lunch"}, call_id="c1")],
        [function_call("create_draft", {"subject": "Lunch?", "body": "Tacos?"}, call_id="c2")],
        [assistant_message("Draft saved, not sent.")],
        [assistant_message("Saved a draft — not sent.")],
    ])
    agent = build_agent(tmp_path / "a.db", tmp_path / "vault", mcp_servers=[], model=model)

    result = await Runner.run(agent, "draft an email", run_config=NO_TRACE)

    assert "not sent" in result.final_output.lower()
    conn = connect(tmp_path / "a.db")
    assert conn.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] == 1
    model.assert_complete()


async def test_coordinator_answers_trivial_chat_without_a_tool(tmp_path):
    """Small talk must not burn a nested run — this is a 3x cost multiplier."""
    model = ScriptedModel([[assistant_message("Hey! What can I do?")]])
    agent = build_agent(tmp_path / "a.db", tmp_path / "vault", mcp_servers=[], model=model)

    result = await Runner.run(agent, "hi", run_config=NO_TRACE)

    assert result.final_output == "Hey! What can I do?"
    assert len(model.calls) == 1
    model.assert_complete()


# --- runtime: the specialist runs directly -------------------------------


async def test_scheduling_specialist_does_timezone_math(tmp_path):
    """Handoff path: the specialist's tool does the timezone work itself."""
    model = ScriptedModel([
        [function_call("set_reminder", {"title": "Standup", "due": "2026-10-01 09:00"}, call_id="c1")],
        [assistant_message("Set.")],
    ])
    scheduling = build_specialists(
        tmp_path / "a.db", tmp_path / "vault", model=model
    )["scheduling"]

    result = await Runner.run(scheduling, "remind me at 9", run_config=NO_TRACE)

    assert result.final_output == "Set."
    conn = connect(tmp_path / "a.db")
    row = conn.execute("SELECT title, due_at FROM reminders").fetchone()
    assert row["due_at"].startswith("2026-10-01T03:00")
    model.assert_complete()
