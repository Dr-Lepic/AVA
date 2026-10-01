"""Slash commands.

These bypass the model entirely: direct SQLite reads and in-memory mutations,
so they are instant, free, and deterministic. That is the point — `/reminders`
costs nothing, asking the agent the same question costs a model call.
"""

import json

import pytest

from ava.commands import HELP, Context, handle
from ava.db import connect
from ava.tools.notes import save_note_impl
from ava.tools.reminders import set_reminder_impl


class FakeTool:
    def __init__(self, name):
        self.name = name


class FakeAgent:
    def __init__(self):
        self.tools = [FakeTool(n) for n in ("save_note", "set_reminder")]
        self.model = "old-model"


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("AVA_TZ", "Asia/Dhaka")
    agent = FakeAgent()
    return Context(
        agent=agent,
        conn=connect(tmp_path / "c.db"),
        model="old-model",
    )


# --- help and discovery --------------------------------------------------


def test_help_lists_the_commands(ctx):
    out = handle("/help", ctx)
    for command in ("/notes", "/reminders", "/tools", "/model", "/quit"):
        assert command in out


def test_help_mentions_how_to_reach_the_agent(ctx):
    assert "anything else" in HELP.lower()


def test_unknown_command_suggests_help(ctx):
    assert "/help" in handle("/nope", ctx)


def test_quit_raises_system_exit(ctx):
    with pytest.raises(SystemExit):
        handle("/quit", ctx)


def test_exit_alias_also_quits(ctx):
    with pytest.raises(SystemExit):
        handle("/exit", ctx)


# --- state inspection ----------------------------------------------------


def test_notes_lists_saved_notes(ctx):
    save_note_impl(ctx.conn, "Groceries", "milk")
    assert "Groceries" in handle("/notes", ctx)


def test_notes_when_empty(ctx):
    assert "No notes" in handle("/notes", ctx)


def test_drafts_when_empty(ctx):
    assert "No drafts" in handle("/drafts", ctx)


def test_reminders_when_empty(ctx):
    assert "No reminders" in handle("/reminders", ctx)


def test_reminders_lists_set_ones(ctx):
    set_reminder_impl(ctx.conn, "Standup", "2027-10-01 09:00")
    assert "Standup" in handle("/reminders", ctx)


def test_tools_lists_tool_names(ctx):
    out = handle("/tools", ctx)
    assert "save_note" in out
    assert "set_reminder" in out


# --- model ----------------------------------------------------------------


def test_model_shows_current(ctx):
    assert "old-model" in handle("/model", ctx)


def test_model_switches(ctx):
    out = handle("/model qwen/qwen3.8-27b:free", ctx)
    assert "qwen/qwen3.8-27b:free" in out
    assert ctx.model == "qwen/qwen3.8-27b:free"
    assert ctx.agent.model == "qwen/qwen3.8-27b:free"


def test_model_refuses_a_paid_slug(ctx):
    """The free-only constraint must hold here too, not just at startup."""
    out = handle("/model openai/gpt-4o-mini", ctx)
    assert "Refused" in out
    assert "not a free model" in out
    # State must be unchanged.
    assert ctx.model == "old-model"
    assert ctx.agent.model == "old-model"


def test_model_accepts_the_free_router(ctx):
    out = handle("/model openrouter/free", ctx)
    assert "openrouter/free" in out


# --- conversation --------------------------------------------------------


def test_new_increments_the_session_id(ctx):
    first = handle("/new", ctx)
    second = handle("/new", ctx)
    assert "ava-cli-1" in first
    assert "ava-cli-2" in second


def test_usage_before_any_turn(ctx):
    assert "no turn" in handle("/usage", ctx).lower()


def test_usage_reports_the_last_turn(ctx):
    ctx.last_usage = "3 call(s), 120 in / 45 out"
    out = handle("/usage", ctx)
    assert "3 call(s)" in out
    assert "120" in out


# --- dispatch ------------------------------------------------------------


@pytest.mark.parametrize("alias", ["/quit", "/exit", "/QUIT"])
def test_commands_are_case_insensitive(ctx, alias):
    if alias == "/QUIT":
        with pytest.raises(SystemExit):
            handle(alias, ctx)
    else:
        with pytest.raises(SystemExit):
            handle(alias, ctx)


def test_extra_whitespace_is_tolerated(ctx):
    assert "old-model" in handle("  /model  ", ctx)


def test_unknown_command_does_not_mutate_state(ctx):
    handle("/bogus", ctx)
    assert ctx.model == "old-model"
