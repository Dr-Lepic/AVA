"""Current time for the agent's instructions.

The model has no clock. It answers from training data, where the date was
2025, and produces confident absolute timestamps a year or more stale.

Two defences, because either alone is insufficient:

1. The tools resolve relative phrases themselves, so the model can pass
   "in 20 minutes" straight through and never do date arithmetic.
2. The current time is injected into the instructions, so that if the model
   does compute an absolute date, it is anchored to reality.

And a past time is refused outright. Defence 2 relies on a small free model
using a date string correctly; defence 3 does not.
"""

from __future__ import annotations

from datetime import datetime

from ava.config import DEFAULT_TZ
from ava.tools.reminders import _local_tz

TIME_GUIDANCE = """\
- You have no clock. Never compute today's date yourself.
- For times, pass the user's words through as-is: "in 20 minutes",
  "tomorrow 9am", "today 15:00". The tools resolve them against the real
  clock.
- If a time is rejected as being in the past, use the current time given
  above to pick a correct one, or pass a relative phrase.
"""


def current_time_block(tz_name: str = DEFAULT_TZ) -> str:
    """A current-time block for injection into agent instructions."""
    now = datetime.now(_local_tz(tz_name))
    return (
        f"Current date and time: {now:%Y-%m-%d %H:%M} ({now:%Z}, {tz_name})\n"
        f"{TIME_GUIDANCE}"
    )
