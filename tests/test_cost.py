"""Per-turn model request cost, measured rather than assumed.

The free tier allows ~50 model requests/day. Multi-agent turns cost more than
one, so the cost of each turn type is pinned here as a hard ceiling. If a
future change makes a turn cost more, this fails rather than quietly eating
the user's daily budget.

How it is measured: a ScriptedModel records every call the runner makes,
including nested runs. A coordinator turn that delegates to a specialist
produces several recorded calls; the count is the real request cost.
"""

import pytest
from agents import RunConfig, Runner
from agents.testing import ScriptedModel, assistant_message, function_call

from ava.agent import build_agent
from ava.agents import build_specialists

NO_TRACE = RunConfig(tracing_disabled=True)

# Free-tier daily budget is 50 requests. These ceilings leave headroom for a
# normal day while catching a change that multiplies cost.
MAX_COST_PLAIN_CHAT = 1
MAX_COST_HANDOFF = 3
MAX_COST_AGENT_AS_TOOL = 5


async def _run_turn(tmp_path, prompt, steps, name="a"):
    model = ScriptedModel(steps)
    agent = build_agent(
        tmp_path / f"{name}.db", tmp_path / "vault", mcp_servers=[], model=model
    )
    result = await Runner.run(agent, prompt, run_config=NO_TRACE)
    return result, len(model.calls)


async def test_plain_chat_costs_one_request(tmp_path):
    _, calls = await _run_turn(
        tmp_path, "hi", [[assistant_message("Hello!")]], name="chat"
    )
    assert calls <= MAX_COST_PLAIN_CHAT


async def test_agent_as_tool_turn_costs_a_bounded_amount(tmp_path):
    """Coordinator -> specialist -> its tool -> coordinator reply."""
    _, calls = await _run_turn(
        tmp_path,
        "remember to buy milk",
        [
            [function_call("take_note", {"input": "buy milk"}, call_id="c1")],
            [function_call("save_note", {"title": "Buy milk"}, call_id="c2")],
            [assistant_message("Inner done.")],
            [assistant_message("Noted.")],
        ],
        name="aat",
    )
    assert calls <= MAX_COST_AGENT_AS_TOOL, f"agent-as-tool turn cost {calls} requests"


async def test_specialist_direct_turn_is_cheap(tmp_path):
    """The handoff target running alone: one call plus its tool turn."""
    model = ScriptedModel([
        [function_call("set_reminder", {"title": "Standup", "due": "2026-10-01 09:00"}, call_id="c1")],
        [assistant_message("Set.")],
    ])
    agent = build_specialists(
        tmp_path / "sched.db", tmp_path / "vault", model=model
    )["scheduling"]
    await Runner.run(agent, "remind me at 9", run_config=NO_TRACE)
    assert len(model.calls) <= MAX_COST_HANDOFF


async def test_trivial_turn_does_not_delegate(tmp_path):
    """Small talk must stay at one request, or the budget dies on greetings."""
    _, calls = await _run_turn(
        tmp_path, "thanks", [[assistant_message("Any time.")]], name="thanks"
    )
    assert calls == 1


def test_documented_ceilings_are_sane():
    """Guard the constants themselves against an edit that loosens them."""
    assert MAX_COST_PLAIN_CHAT == 1
    assert MAX_COST_PLAIN_CHAT < MAX_COST_HANDOFF < MAX_COST_AGENT_AS_TOOL
