"""Daily model-request counter for the free tier.

A free OpenRouter account allows roughly 50 model requests per day, and a
delegated turn costs about four. The only other signal is a 429 from
OpenRouter, by which point the day is spent — and failed requests count too,
so a bug that burns requests is worth seeing locally.

State is a small JSON file rather than a table: it is not relational data, and
deleting it resets the counter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Free-tier daily budget, with a little headroom: the exact published figure
# has moved over time, and overshooting costs a confusing failure.
DAILY_LIMIT = 50

# Start warning here, leaving enough requests for one more delegated turn.
WARN_AT = 40

FALLBACK_HINT = "openrouter/free"


def today() -> str:
    return date.today().isoformat()


@dataclass
class BudgetState:
    day: str
    used: int


class Budget:
    """Counts model requests per calendar day, persisted as JSON."""

    def __init__(self, path: Path | str, limit: int = DAILY_LIMIT):
        self.path = Path(path)
        self.limit = limit

    def _load(self) -> BudgetState:
        day = today()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if raw.get("day") == day:
                return BudgetState(day=day, used=int(raw.get("used", 0)))
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        # Missing, corrupt, or from a previous day: start fresh. A counter that
        # kept yesterday's total would read full at breakfast and spent at
        # dinner, which is worse than no counter.
        return BudgetState(day=day, used=0)

    def _save(self, state: BudgetState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"day": state.day, "used": state.used}), encoding="utf-8"
        )

    def used(self) -> int:
        return self._load().used

    def remaining(self) -> int:
        return max(0, self.limit - self.used())

    def record(self, n: int = 1) -> int:
        """Record n model requests. Returns the new total."""
        state = self._load()
        state.used += n
        self._save(state)
        return state.used

    def should_warn(self) -> bool:
        return self.used() >= WARN_AT

    def is_exhausted(self) -> bool:
        return self.used() >= self.limit

    def summary(self) -> str:
        used = self.used()
        left = max(0, self.limit - used)
        if self.is_exhausted():
            return (
                f"Daily free limit reached ({used}/{self.limit}). "
                f"Try /model {FALLBACK_HINT}, or come back after midnight."
            )
        warn = " (approaching limit)" if self.should_warn() else ""
        return f"Requests today: {used}/{self.limit} — {left} left{warn}."
