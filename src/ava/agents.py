"""Specialist agent definitions.

Each specialist owns exactly one domain's tools. The coordinator (see agent.py)
holds none of these tools itself — it routes to them.

Tool isolation is the point: a reminder request should never compete for the
model's attention with email drafts. The tests in test_agents.py pin that no
tool appears on more than one specialist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from agents import Agent, Model

from ava.tools import drafts, notes, reminders

# A specialist takes whatever the SDK's `model` accepts: a slug string in
# production, or a Model instance (e.g. ScriptedModel) in tests.
ModelLike = Union[str, Model]

NOTES_INSTRUCTIONS = """\
You handle notes for AVA, a personal assistant.

- Save a note whenever the user states something worth remembering.
- Search before creating, in case it already exists.
- Use tags (comma separated) when the user implies a category.
- Keep titles short. Confirm what you saved in one line.
"""

DRAFTING_INSTRUCTIONS = """\
You write email drafts for AVA, a personal assistant.

- Draft only. You NEVER send mail. There is no send capability.
- If the user gives a rough idea, write the actual email body yourself.
- Always state clearly that the draft was saved and not sent.
"""

SCHEDULING_INSTRUCTIONS = """\
You manage reminders for AVA, a personal assistant.

- Convert relative times ("tomorrow 9am", "in 20 minutes") to an explicit
  local ISO-8601 time before calling set_reminder. The naive time is read in
  the user's configured timezone, so "09:00" means 9am their time.
- Always echo the resolved local time back so the user can catch a mistake.
- To snooze or cancel you need the numeric id — call list_reminders first.
"""


def build_specialists(
    db_path: Path | str,
    vault_path: Path,
    model: ModelLike,
) -> dict[str, Agent]:
    """Build the three domain specialists, keyed by short name.

    Takes the database *path*, not a connection: the SDK runs tool functions
    on worker threads and a sqlite3 connection is bound to its creating
    thread, so each call resolves its own via get_connection.

    handoff_description is not decoration: the coordinator chooses a
    specialist by reading it, so it must state the domain plainly.
    """
    vault_path.mkdir(parents=True, exist_ok=True)

    return {
        "notes": Agent(
            name="Notes",
            handoff_description="Saves, lists, and searches the user's notes.",
            instructions=NOTES_INSTRUCTIONS,
            model=model,
            tools=notes.as_tools(db_path),
        ),
        "drafting": Agent(
            name="Drafting",
            handoff_description="Writes email drafts for the user. Never sends mail.",
            instructions=DRAFTING_INSTRUCTIONS,
            model=model,
            tools=drafts.as_tools(db_path, vault_path),
        ),
        "scheduling": Agent(
            name="Scheduling",
            handoff_description="Sets, lists, snoozes, and cancels the user's reminders.",
            instructions=SCHEDULING_INSTRUCTIONS,
            model=model,
            tools=reminders.as_tools(db_path),
        ),
    }
