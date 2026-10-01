"""Resolving reminder times against the real clock.

The bug this exists for: the model has no clock, so it computes "tomorrow"
from its training data, where the date was 2025. It passed 2025-09-24 to
set_reminder, which stored it happily, and the watcher fired it immediately
because a past reminder is due on the next tick.

So the model must not do date arithmetic. It passes through what the user
actually said — "in 20 minutes", "tomorrow 9am" — and this module resolves it
against datetime.now(). Anything resolving to the past is refused.
"""

from datetime import datetime, timedelta, timezone

import pytest

from ava.db import connect
from ava.tools.reminders import (
    PastDueError,
    ReminderError,
    parse_relative,
    parse_today_word,
    set_reminder_impl,
)

TZ = "Asia/Dhaka"


@pytest.fixture(autouse=True)
def dhaka(monkeypatch):
    monkeypatch.setenv("AVA_TZ", TZ)


def _local_now():
    from ava.tools.reminders import _local_tz

    return datetime.now(_local_tz())


# --- relative phrases ----------------------------------------------------


def test_in_20_minutes_resolves_forward():
    now = _local_now()
    resolved = parse_relative("in 20 minutes", TZ)
    delta = resolved - now
    assert timedelta(minutes=19) < delta <= timedelta(minutes=20)


def test_in_hours_and_days():
    now = _local_now()
    assert timedelta(hours=1, minutes=59) < (parse_relative("in 2 hours", TZ) - now) <= timedelta(hours=2)
    assert timedelta(days=6, hours=23) < (parse_relative("in 1 week", TZ) - now) <= timedelta(days=7)


def test_in_30_seconds():
    now = _local_now()
    assert timedelta(seconds=29) < (parse_relative("in 30 seconds", TZ) - now) <= timedelta(seconds=30)


def test_in_0_minutes_is_already_due():
    """Zero is a real input; it should resolve to now, not fail."""
    assert parse_relative("in 0 minutes", TZ) is not None


def test_relative_is_case_insensitive():
    assert parse_relative("IN 5 MINUTES", TZ) is not None


def test_non_relative_returns_none():
    """Falls through to absolute parsing rather than guessing."""
    assert parse_relative("2026-10-01 09:00", TZ) is None
    assert parse_relative("tomorrow 9am", TZ) is None
    assert parse_relative("sometime", TZ) is None


def test_unknown_unit_returns_none():
    assert parse_relative("in 5 fortnights", TZ) is None


# --- day words -----------------------------------------------------------


def test_tomorrow_9am_is_tomorrow():
    now = _local_now()
    resolved = parse_today_word("tomorrow 9am", TZ)
    assert resolved.date() == (now + timedelta(days=1)).date()
    assert resolved.hour == 9
    assert resolved.minute == 0


def test_today_9am_is_today():
    now = _local_now()
    resolved = parse_today_word("today 9am", TZ)
    assert resolved.date() == now.date()
    assert resolved.hour == 9


def test_day_after_tomorrow():
    now = _local_now()
    resolved = parse_today_word("day after tomorrow 9am", TZ)
    assert resolved.date() == (now + timedelta(days=2)).date()


def test_pm_is_converted():
    assert parse_today_word("today 3pm", TZ).hour == 15
    assert parse_today_word("today 3:30pm", TZ).hour == 15
    assert parse_today_word("today 3:30pm", TZ).minute == 30


def test_midnight_and_noon_edges():
    assert parse_today_word("today 12am", TZ).hour == 0
    assert parse_today_word("today 12pm", TZ).hour == 12


def test_colon_form_without_suffix():
    assert parse_today_word("today 21:30", TZ).hour == 21


def test_day_word_without_a_clock_returns_none():
    assert parse_today_word("tomorrow", TZ) is None


def test_unrecognised_day_word_returns_none():
    assert parse_today_word("someday 9am", TZ) is None


# --- the actual bug ------------------------------------------------------


def test_stale_absolute_date_is_refused(tmp_path):
    """The reported failure: a 2025 date must not be stored as due-now."""
    conn = connect(tmp_path / "r.db")
    with pytest.raises(PastDueError):
        set_reminder_impl(conn, "Standup", "2025-09-24 09:00")
    assert conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0] == 0


def test_past_due_error_names_the_real_current_time(tmp_path):
    """The error has to tell the model what now actually is, so it can retry."""
    conn = connect(tmp_path / "r.db")
    with pytest.raises(PastDueError) as exc:
        set_reminder_impl(conn, "X", "2025-09-24 09:00")
    message = str(exc.value)
    assert str(_local_now().year) in message


def test_past_due_is_a_reminder_error(tmp_path):
    """Callers catching ReminderError keep working."""
    conn = connect(tmp_path / "r.db")
    with pytest.raises(ReminderError):
        set_reminder_impl(conn, "X", "2025-09-24 09:00")


def test_future_date_is_accepted(tmp_path):
    conn = connect(tmp_path / "r.db")
    future = (_local_now() + timedelta(days=1)).replace(microsecond=0)
    out = set_reminder_impl(conn, "Tomorrow", future.isoformat())
    assert "Reminder #1 set" in out


def test_relative_phrase_is_accepted_end_to_end(tmp_path):
    """The model can now pass the user's words straight through."""
    conn = connect(tmp_path / "r.db")
    out = set_reminder_impl(conn, "Stretch", "in 20 minutes")
    assert "Reminder #1 set: Stretch" in out


def test_day_word_is_accepted_end_to_end(tmp_path):
    conn = connect(tmp_path / "r.db")
    out = set_reminder_impl(conn, "Standup", "tomorrow 9am")
    assert "Reminder #1 set: Standup" in out


def test_relative_phrase_is_not_due_immediately(tmp_path):
    """The symptom of the bug: stored-then-instantly-due."""
    from ava.tools.reminders import due_reminders

    conn = connect(tmp_path / "r.db")
    set_reminder_impl(conn, "Stretch", "in 20 minutes")
    assert due_reminders(conn) == []
