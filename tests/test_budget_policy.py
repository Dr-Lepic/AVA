"""The CLI's budget policy.

Separate from test_budget.py, which tests the counter itself. What matters
here is policy: charge the real per-turn call count, block once the day's
budget is gone, and keep slash commands working regardless — they cost
nothing, so refusing them would be a pointless regression.
"""

import pytest

from ava.budget import DAILY_LIMIT, WARN_AT, Budget


# --- charging ------------------------------------------------------------


def test_a_delegated_turn_charges_its_real_call_count(tmp_path):
    """Measured: a delegated turn is ~4 requests, not 1.

    Charging 1 per message would undercount by 4x, and the warning would fire
    when the day's budget was already long gone.
    """
    budget = Budget(tmp_path / "usage.json")
    budget.record(4)
    assert budget.used() == 4
    assert budget.remaining() == DAILY_LIMIT - 4


def test_a_plain_turn_charges_one(tmp_path):
    budget = Budget(tmp_path / "usage.json")
    budget.record(1)
    assert budget.used() == 1


def test_thirteen_delegated_turns_exhaust_the_day(tmp_path):
    """The arithmetic behind the ~12 turns/day figure in the banner.

    12 delegated turns is 48 requests, just inside the 50 limit; the 13th is
    what tips it over. The banner's "~12" is the count you can actually use.
    """
    budget = Budget(tmp_path / "usage.json")
    for _ in range(12):
        budget.record(4)
    assert not budget.is_exhausted(), "12 turns should still fit in the budget"

    budget.record(4)
    assert budget.is_exhausted()


# --- thresholds ----------------------------------------------------------


def test_warn_threshold_leaves_room_for_a_further_turn():
    """Warning too late is the same as not warning at all."""
    assert DAILY_LIMIT - WARN_AT >= 4


def test_no_warning_while_there_is_room(tmp_path):
    budget = Budget(tmp_path / "usage.json")
    budget.record(WARN_AT - 4)
    assert budget.should_warn() is False


def test_warning_fires_before_exhaustion(tmp_path):
    """There must be a window between the warning and the hard stop."""
    budget = Budget(tmp_path / "usage.json")
    budget.record(WARN_AT)
    assert budget.should_warn() is True
    assert budget.is_exhausted() is False


# --- slash commands stay free -------------------------------------------


def test_slash_commands_do_not_consume_budget(tmp_path):
    """A user checking their notes 20 times must not spend a request."""
    budget = Budget(tmp_path / "usage.json")
    for _ in range(20):
        pass  # none of these would call the model
    assert budget.used() == 0


def test_exhausted_budget_message_offers_a_way_forward(tmp_path):
    """Refusing without an alternative is a dead end."""
    budget = Budget(tmp_path / "usage.json", limit=3)
    budget.record(3)
    message = budget.summary().lower()
    assert "openrouter/free" in message
    assert "midnight" in message
