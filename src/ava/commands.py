"""Slash commands.

These bypass the model entirely: direct SQLite reads and in-memory mutations.
That is the point — `/reminders` is instant and costs no requests, where
asking the agent the same question spends part of a 50/day free-tier budget.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from agents import Agent

from ava.config import ConfigError, assert_free_model
from ava.tools.drafts import list_drafts_impl
from ava.tools.notes import list_notes_impl
from ava.tools.reminders import list_reminders_impl

# /new hands the CLI a new session id through its return value rather than
# the CLI guessing at it, so the two cannot drift apart.
NEW_SESSION_PREFIX = "__new_session__:"

HELP = """\
Commands:
  /help          show this list
  /notes         list recent notes
  /drafts        list recent drafts
  /reminders     list pending reminders
  /tools         list tool names the agent can call
  /model         show the current model
  /model <slug>  switch model (free models only)
  /new           start a fresh conversation
  /usage         model calls and tokens used by the last turn
  /quit          exit

Anything else is sent to the agent."""


@dataclass
class Context:
    """State the handlers read and mutate. /model and /new change it in place."""

    # The coordinator Agent. /tools reads its tool list and /model rewrites its
    # model, so both attributes are genuinely needed.
    agent: Agent
    conn: sqlite3.Connection
    model: str
    last_usage: str = "no turn yet"
    session_index: int = 0

    def new_session_id(self) -> str:
        self.session_index += 1
        return f"ava-cli-{self.session_index}"


def handle(command: str, ctx: Context) -> str:
    """Dispatch one slash command and return the text to print.

    Raises SystemExit for /quit so the CLI can break its loop, keeping control
    flow out of the handlers.
    """
    parts = command.strip().split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if name in {"/quit", "/exit"}:
        raise SystemExit(0)

    if name in {"/help", "/?"}:
        return HELP

    if name == "/notes":
        return list_notes_impl(ctx.conn, limit=10)

    if name == "/drafts":
        return list_drafts_impl(ctx.conn, limit=10)

    if name == "/reminders":
        return list_reminders_impl(ctx.conn)

    if name == "/tools":
        return "\n".join(sorted(t.name for t in ctx.agent.tools))

    if name == "/model":
        if not arg:
            return f"Current model: {ctx.model}"
        # Same guard the config uses, so /model cannot switch to a paid slug
        # and quietly break the free-only constraint.
        try:
            chosen = assert_free_model(arg)
        except ConfigError as exc:
            return f"Refused: {exc}"
        ctx.model = chosen
        ctx.agent.model = chosen
        return f"Model set to {chosen}. Takes effect next message."

    if name == "/new":
        # A new session id rather than truncating: a fresh id cannot corrupt an
        # existing transcript. The CLI rebuilds the session from this id.
        return f"{NEW_SESSION_PREFIX}{ctx.new_session_id()}"

    if name == "/usage":
        return ctx.last_usage

    return f"Unknown command {name}. Try /help."
