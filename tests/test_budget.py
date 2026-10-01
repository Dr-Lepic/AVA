"""Daily request budget.

The free tier allows ~50 model requests/day, and a delegated turn costs about
4. Without local tracking the only signal is an opaque 429 from OpenRouter, by
which point the day's budget is gone.

The counter exists to warn *before* that happens, and because failed requests
still count toward the quota — so a bug that burns requests is worth seeing.
"""

import json

import pytest

from ava.budget import (
    DAILY_LIMIT,
    WARN_AT,
    Budget,
)


# --- counting ------------------------------------------------------------


def test_starts_at_zero(tmp_path):
    budget = Budget(tmp_path / "usage.json")
    assert budget.used() == 0
    assert budget.remaining() == DAILY_LIMIT


def test_record_increments(tmp_path):
    budget = Budget(tmp_path / "usage.json")
    assert budget.record() == 1
    assert budget.record(3) == 4
    assert budget.used() == 4


def test_persists_across_instances(tmp_path):
    path = tmp_path / "usage.json"
    Budget(path).record(5)
    assert Budget(path).used() == 5


def test_remaining_never_negative(tmp_path):
    budget = Budget(tmp_path / "usage.json", limit=3)
    budget.record(10)
    assert budget.remaining() == 0


def test_custom_limit_is_respected(tmp_path):
    budget = Budget(tmp_path / "usage.json", limit=10)
    assert budget.limit == 10


# --- reset behaviour -----------------------------------------------------


def test_corrupt_file_resets_to_zero(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("{not json", encoding="utf-8")
    assert Budget(path).used() == 0


def test_file_from_a_previous_day_resets(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text(json.dumps({"day": "1999-01-01", "used": 50}), encoding="utf-8")
    assert Budget(path).used() == 0


def test_reset_preserves_todays_count(tmp_path):
    """Only a stale day resets; today's count must survive reopening."""
    from ava.budget import today

    path = tmp_path / "usage.json"
    path.write_text(json.dumps({"day": today(), "used": 7}), encoding="utf-8")
    assert Budget(path).used() == 7


def test_missing_file_is_not_an_error(tmp_path):
    assert Budget(tmp_path / "nope" / "usage.json").used() == 0


def test_writes_into_a_missing_directory(tmp_path):
    budget = Budget(tmp_path / "a" / "b" / "usage.json")
    budget.record()
    assert (tmp_path / "a" / "b" / "usage.json").exists()


# --- thresholds ----------------------------------------------------------


def test_does_not_warn_at_the_start(tmp_path):
    assert Budget(tmp_path / "usage.json").should_warn() is False


def test_warns_at_the_threshold(tmp_path):
    budget = Budget(tmp_path / "usage.json")
    budget.record(WARN_AT)
    assert budget.should_warn() is True


def test_exhausted_at_the_limit(tmp_path):
    budget = Budget(tmp_path / "usage.json", limit=2)
    budget.record(2)
    assert budget.is_exhausted() is True


def test_not_exhausted_below_the_limit(tmp_path):
    budget = Budget(tmp_path / "usage.json", limit=5)
    budget.record(4)
    assert budget.is_exhausted() is False


# --- messages ------------------------------------------------------------


def test_summary_shows_the_numbers(tmp_path):
    budget = Budget(tmp_path / "usage.json")
    budget.record(10)
    out = budget.summary()
    assert "10" in out
    assert str(DAILY_LIMIT) in out


def test_summary_mentions_approaching_the_limit(tmp_path):
    budget = Budget(tmp_path / "usage.json")
    budget.record(WARN_AT)
    assert "approaching" in budget.summary().lower()


def test_summary_when_exhausted_suggests_a_fallback(tmp_path):
    budget = Budget(tmp_path / "usage.json", limit=2)
    budget.record(2)
    out = budget.summary()
    assert "limit" in out.lower()
    assert "openrouter/free" in out


def test_summary_before_any_use(tmp_path):
    assert "0" in Budget(tmp_path / "usage.json").summary()
