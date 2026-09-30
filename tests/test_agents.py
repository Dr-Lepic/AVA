"""Specialist agent definitions.

Two properties matter beyond "does it construct":

- tool isolation. A specialist that can reach another domain's tools makes the
  split pointless, and the coordinator would have no reason to route.
- handoff descriptions. The coordinator chooses a specialist by reading these.
  An empty or vague description means routing silently fails against a real
  model, which costs a request every time.
"""

from ava.agents import build_specialists
from ava.db import connect

EXPECTED_KEYS = {"notes", "drafting", "scheduling"}

NOTES_TOOLS = {"save_note", "list_notes", "search_notes"}
DRAFT_TOOLS = {"create_draft", "list_drafts"}
SCHEDULING_TOOLS = {
    "set_reminder",
    "list_reminders",
    "cancel_reminder",
    "snooze_reminder",
}


def _specs(tmp_path, model="test/model:free"):
    return build_specialists(tmp_path / "a.db", tmp_path / "vault", model=model)


def _tools(agent):
    return {t.name for t in agent.tools}


# --- shape ---------------------------------------------------------------


def test_builds_exactly_three_specialists(tmp_path):
    assert set(_specs(tmp_path)) == EXPECTED_KEYS


def test_specialists_have_the_expected_names(tmp_path):
    specs = _specs(tmp_path)
    assert {s.name for s in specs.values()} == {"Notes", "Drafting", "Scheduling"}


def test_all_use_the_given_model(tmp_path):
    specs = _specs(tmp_path, model="some/model:free")
    assert all(s.model == "some/model:free" for s in specs.values())


# --- tool isolation ------------------------------------------------------


def test_notes_specialist_has_only_note_tools(tmp_path):
    assert _tools(_specs(tmp_path)["notes"]) == NOTES_TOOLS


def test_drafting_specialist_has_only_draft_tools(tmp_path):
    assert _tools(_specs(tmp_path)["drafting"]) == DRAFT_TOOLS


def test_scheduling_specialist_has_only_reminder_tools(tmp_path):
    assert _tools(_specs(tmp_path)["scheduling"]) == SCHEDULING_TOOLS


def test_no_tool_leaks_across_domains(tmp_path):
    """The core invariant. A cross-domain tool makes routing meaningless."""
    specs = _specs(tmp_path)
    seen: set[str] = set()
    for agent in specs.values():
        names = _tools(agent)
        assert not (names & seen), f"tool leak into {agent.name}"
        seen |= names


def test_specialists_do_not_own_mcp_servers(tmp_path):
    """MCP file access belongs to the coordinator, not every specialist."""
    for agent in _specs(tmp_path).values():
        assert not agent.mcp_servers


def test_specialists_have_no_nested_handoffs(tmp_path):
    """A specialist that could hand off again would blur the two levels."""
    for agent in _specs(tmp_path).values():
        assert not agent.handoffs


# --- routing affordances -------------------------------------------------


def test_every_specialist_has_a_handoff_description(tmp_path):
    for key, agent in _specs(tmp_path).items():
        assert agent.handoff_description, f"{key} missing handoff_description"


def test_handoff_descriptions_are_distinct(tmp_path):
    """Identical descriptions give the model nothing to choose between."""
    specs = _specs(tmp_path)
    descriptions = [a.handoff_description for a in specs.values()]
    assert len(set(descriptions)) == len(descriptions)


def test_handoff_descriptions_mention_their_domain(tmp_path):
    """The coordinator routes on this text alone."""
    specs = _specs(tmp_path)
    assert "note" in specs["notes"].handoff_description.lower()
    assert "draft" in specs["drafting"].handoff_description.lower()
    assert "reminder" in specs["scheduling"].handoff_description.lower()


# --- instructions carry the domain rules ---------------------------------


def test_drafting_promises_never_sends(tmp_path):
    instructions = _specs(tmp_path)["drafting"].instructions.lower()
    assert "never send" in instructions


def test_scheduling_mentions_timezone(tmp_path):
    """The specialist owns timezone reasoning, so it must say so."""
    instructions = _specs(tmp_path)["scheduling"].instructions.lower()
    assert "timezone" in instructions


def test_notes_instructions_mention_saving(tmp_path):
    assert "save" in _specs(tmp_path)["notes"].instructions.lower()


def test_all_specialists_have_instructions(tmp_path):
    for key, agent in _specs(tmp_path).items():
        assert agent.instructions, f"{key} has no instructions"


# --- construction hygiene ------------------------------------------------


def test_build_specialists_creates_the_vault(tmp_path):
    vault = tmp_path / "nested" / "vault"
    build_specialists(tmp_path / "a.db", vault, model="m")
    assert vault.is_dir()


def test_specs_are_independent_objects(tmp_path):
    """Two builds must not share mutable tool state."""
    first = _specs(tmp_path)["notes"]
    second = build_specialists(
        tmp_path / "b.db", tmp_path / "vault2", model="m"
    )["notes"]
    assert first is not second
    assert first.tools[0] is not second.tools[0]
