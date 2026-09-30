"""Builds the AVA coordinator agent.

The coordinator owns no domain tools. It routes to three specialists, exposed
both as handoffs (the specialist takes the turn) and as agents-as-tools (the
coordinator stays in charge and narrates).

An agent-as-tool takes a single free-text `input`, not the inner agent's tool
schema: the coordinator describes the job in prose and the specialist chooses
which of its own tools to call.

Scheduling is deliberately handoff-only. Exposing it as a tool would let the
coordinator set a reminder without routing through the specialist that owns
the timezone reasoning, and a reminder set in the wrong timezone is worse than
no reminder.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from agents import Agent
from agents.mcp import MCPServer

from ava.agents import ModelLike, build_specialists
from ava.mcp_servers import build_servers

INSTRUCTIONS = """\
You are AVA, a personal assistant for one person, running in a terminal.

You coordinate three specialists:
- Notes: saves and searches notes
- Drafting: writes email drafts (never sends)
- Scheduling: sets, snoozes, and cancels reminders

How to route:
- If the request is purely one domain, HAND OFF to that specialist and let
  it answer the user directly.
- If the request spans domains, call the specialists as tools (draft_email,
  take_note) and combine their results yourself.
- For trivial conversation ("hi", "thanks", "what can you do"), just answer
  directly. Do not delegate small talk — each delegation costs extra model
  requests.

Rules:
- Never claim you did something without calling a specialist or tool.
- Drafts are never sent. Say so whenever a draft is created.
- Keep replies short. This is a terminal.
"""


def build_agent(
    db_path: Path | str,
    vault_path: Path,
    mcp_servers: list[MCPServer] | None = None,
    model: ModelLike = "openrouter/free",
    instructions: str = INSTRUCTIONS,
) -> Agent:
    """Assemble the coordinator with handoffs and agents-as-tools."""
    specs = build_specialists(db_path, vault_path, model=model)

    # Sequence, not list: list is invariant, so list[MCPServerStdio] is not
    # assignable to list[MCPServer]. Sequence is covariant, so it is.
    servers: Sequence[MCPServer] = (
        mcp_servers if mcp_servers is not None else build_servers(vault_path)
    )

    return Agent(
        name="AVA",
        instructions=instructions,
        model=model,
        # Handoffs: the specialist takes over the turn and answers directly.
        handoffs=[specs["notes"], specs["drafting"], specs["scheduling"]],
        # Agents-as-tools: coordinator keeps control and narrates the result.
        # Scheduling is absent on purpose — see the module docstring.
        tools=[
            specs["drafting"].as_tool(
                tool_name="draft_email",
                tool_description=(
                    "Ask the drafting specialist to write an email draft. "
                    "Pass what the draft should say. This does NOT send mail."
                ),
            ),
            specs["notes"].as_tool(
                tool_name="take_note",
                tool_description=(
                    "Ask the notes specialist to save or search the user's "
                    "notes. Pass what to remember or what to look for."
                ),
            ),
        ],
        mcp_servers=(
            mcp_servers if mcp_servers is not None else build_servers(vault_path)
        ),
    )
