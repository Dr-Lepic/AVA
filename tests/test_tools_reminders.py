"""Reminder storage and due-time parsing.

Timezone handling is the load-bearing part of this module and the easiest thing
to get silently wrong, so it gets the most tests: a reminder set for 9am that
fires at 3pm is worse than no reminder, because the user stops trusting it.
"""

from datetime import datetime, timedelta, timezone

import pytest

from ava.db import connect
from ava.tools import reminders as rem

# 2027-10-01T09:00 in Asia/Dhaka (UTC+6, no DST) is 03:00 UTC.
DHAKA_9AM_UTC = "2027-10-01T03:00:00+00:00"


def _tz(monkeypatch, name):
    monkeypatch.setenv("AVA_TZ", name)


# --- parse_due -----------------------------------------------------------


def test_parse_due_interprets_naive_as_local(monkeypatch):
    _tz(monkeypatch, "Asia/Dhaka")
    utc = rem.parse_due("2027-10-01 09:00")
    assert utc.tzinfo == timezone.utc
    assert utc.hour == 3


def test_parse_due_respects_explicit_offset():
    utc = rem.parse_due("2027-10-01T09:00:00+00:00")
    assert utc.hour == 9
    assert utc.tzinfo == timezone.utc


def test_parse_due_converts_other_offset_to_utc():
    utc = rem.parse_due("2027-10-01T09:00:00+06:00")
    assert utc.hour == 3
    assert utc.tzinfo == timezone.utc


def test_parse_due_t_iso_separator(monkeypatch):
    _tz(monkeypatch, "Asia/Dhaka")
    assert rem.parse_due("2027-10-01T09:00:00").hour == 3


def test_parse_due_rejects_date_only(monkeypatch):
    """A bare date is ambiguous: 2027-10-01 could mean midnight or all day.

    Rejecting it forces the model to state a time, which is safer than
    silently guessing midnight and firing a day early.
    """
    _tz(monkeypatch, "Asia/Dhaka")
    with pytest.raises(rem.ReminderError, match="no time of day"):
        rem.parse_due("2027-10-01")


def test_parse_due_accepts_explicit_midnight(monkeypatch):
    """The date-only guard must not reject a time the caller stated."""
    _tz(monkeypatch, "Asia/Dhaka")
    utc = rem.parse_due("2027-10-01T00:00:00")
    assert utc.hour == 18  # previous day, 18:00 UTC


def test_parse_due_rejects_garbage():
    with pytest.raises(rem.ReminderError):
        rem.parse_due("next tuesday-ish")


def test_parse_due_rejects_empty():
    with pytest.raises(rem.ReminderError):
        rem.parse_due("   ")


def test_parse_due_error_names_the_offending_value():
    with pytest.raises(rem.ReminderError, match="nonsense"):
        rem.parse_due("nonsense")


def test_parse_due_uses_configured_timezone(monkeypatch):
    """A UTC-configured user reading 09:00 must get 09:00, not 03:00."""
    _tz(monkeypatch, "UTC")
    assert rem.parse_due("2027-10-01 09:00").hour == 9


def test_parse_due_honours_western_timezone(monkeypatch):
    _tz(monkeypatch, "America/New_York")
    # 09:00 EDT (UTC-4) is 13:00 UTC
    assert rem.parse_due("2027-10-01 09:00").hour == 13


def test_parse_due_southern_hemisphere_dst(monkeypatch):
    """Sydney is UTC+11 in January and UTC+10 in July — inverted vs the north.

    Verified against zoneinfo: 09:00 local is 22:00 UTC the previous day in
    January, and 23:00 UTC in July. Getting this backwards would be a
    one-hour reminder error nobody would notice until it fired.
    """
    _tz(monkeypatch, "Australia/Sydney")
    january = rem.parse_due("2026-01-15 09:00")
    july = rem.parse_due("2026-07-15 09:00")
    assert january.hour == 22  # previous day in UTC
    assert july.hour == 23


def test_bad_timezone_raises_friendly_error(monkeypatch):
    _tz(monkeypatch, "Mars/Olympus_Mons")
    with pytest.raises(rem.ReminderError, match="timezone"):
        rem.parse_due("2027-10-01 09:00")


# --- set_reminder --------------------------------------------------------


def test_set_reminder_confirms_with_local_time(tmp_path, monkeypatch):
    _tz(monkeypatch, "Asia/Dhaka")
    conn = connect(tmp_path / "rem.db")
    out = rem.set_reminder_impl(conn, "Standup", "2027-10-01 09:00")
    assert "Reminder #1 set: Standup" in out
    assert "2027-10-01 09:00" in out


def test_set_reminder_stores_utc(tmp_path, monkeypatch):
    _tz(monkeypatch, "Asia/Dhaka")
    conn = connect(tmp_path / "r.db")
    rem.set_reminder_impl(conn, "Standup", "2027-10-01 09:00")
    stored = conn.execute("SELECT due_at FROM reminders").fetchone()["due_at"]
    assert datetime.fromisoformat(stored) == datetime.fromisoformat(DHAKA_9AM_UTC)


def test_set_reminder_propagates_parse_error(tmp_path, monkeypatch):
    _tz(monkeypatch, "Asia/Dhaka")
    conn = connect(tmp_path / "r.db")
    with pytest.raises(rem.ReminderError):
        rem.set_reminder_impl(conn, "X", "whenever")
    assert conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0] == 0


def test_set_reminder_confirms_across_day_boundary(tmp_path, monkeypatch):
    """Dhaka 00:30 is the previous day in UTC — the echo must not mislead."""
    _tz(monkeypatch, "Asia/Dhaka")
    conn = connect(tmp_path / "rem.db")
    out = rem.set_reminder_impl(conn, "Late", "2027-10-01 00:30")
    assert "2027-10-01 00:30" in out


# --- list / cancel -------------------------------------------------------


def test_list_reminders_empty(tmp_path):
    conn = connect(tmp_path / "r.db")
    assert "No reminders" in rem.list_reminders_impl(conn)


def test_list_reminders_earliest_first(tmp_path, monkeypatch):
    _tz(monkeypatch, "Asia/Dhaka")
    conn = connect(tmp_path / "r.db")
    rem.set_reminder_impl(conn, "later", "2027-10-02 09:00")
    rem.set_reminder_impl(conn, "sooner", "2027-10-01 09:00")
    assert rem.list_reminders_impl(conn).index("sooner") < rem.list_reminders_impl(conn).index("later")


def test_list_reminders_hides_done_by_default(tmp_path, monkeypatch):
    _tz(monkeypatch, "Asia/Dhaka")
    conn = connect(tmp_path / "r.db")
    rem.set_reminder_impl(conn, "pending", "2027-10-02 09:00")
    rem.set_reminder_impl(conn, "finished", "2027-10-03 09:00")
    conn.execute("UPDATE reminders SET done = 1 WHERE title = 'finished'")
    conn.commit()
    assert "finished" not in rem.list_reminders_impl(conn)


def test_list_reminders_includes_done_on_request(tmp_path, monkeypatch):
    _tz(monkeypatch, "Asia/Dhaka")
    conn = connect(tmp_path / "r.db")
    rem.set_reminder_impl(conn, "finished", "2027-10-03 09:00")
    conn.execute("UPDATE reminders SET done = 1")
    conn.commit()
    assert "finished" in rem.list_reminders_impl(conn, include_done=True)


def test_cancel_reminder_missing_id(tmp_path):
    conn = connect(tmp_path / "r.db")
    assert "No reminder with id 999" in rem.cancel_reminder_impl(conn, 999)


def test_cancel_reminder_deletes(tmp_path, monkeypatch):
    _tz(monkeypatch, "Asia/Dhaka")
    conn = connect(tmp_path / "r.db")
    rem.set_reminder_impl(conn, "X", "2027-10-01 09:00")
    assert "cancelled" in rem.cancel_reminder_impl(conn, 1)
    assert "No reminders" in rem.list_reminders_impl(conn)


# --- due / mark_done (the watcher's interface) --------------------------


def test_due_reminders_filters_future(tmp_path):
    conn = connect(tmp_path / "r.db")
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
    conn.execute("INSERT INTO reminders (title,due_at,created_at) VALUES ('P',?,'x')", (past,))
    conn.execute("INSERT INTO reminders (title,due_at,created_at) VALUES ('F',?,'x')", (future,))
    conn.commit()
    assert [r["title"] for r in rem.due_reminders(conn)] == ["P"]


def test_due_reminders_excludes_done(tmp_path):
    conn = connect(tmp_path / "r.db")
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    conn.execute("INSERT INTO reminders (title,due_at,done,created_at) VALUES ('P',?,1,'x')", (past,))
    conn.commit()
    assert rem.due_reminders(conn) == []


def test_due_reminders_empty(tmp_path):
    conn = connect(tmp_path / "r.db")
    assert rem.due_reminders(conn) == []


def test_due_reminders_boundary_is_inclusive(tmp_path):
    """A reminder due exactly now must fire, not wait for the next tick."""
    conn = connect(tmp_path / "r.db")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("INSERT INTO reminders (title,due_at,created_at) VALUES ('now',?,'x')", (now,))
    conn.commit()
    assert len(rem.due_reminders(conn)) == 1


def test_mark_done_is_idempotent(tmp_path):
    conn = connect(tmp_path / "r.db")
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    conn.execute("INSERT INTO reminders (title,due_at,created_at) VALUES ('P',?,'x')", (past,))
    conn.commit()
    rem.mark_done(conn, 1)
    rem.mark_done(conn, 1)
    assert rem.due_reminders(conn) == []


def test_due_reminders_oldest_first(tmp_path):
    conn = connect(tmp_path / "r.db")
    now = datetime.now(timezone.utc)
    older = (now - timedelta(hours=3)).isoformat()
    newer = (now - timedelta(hours=1)).isoformat()
    conn.execute("INSERT INTO reminders (title,due_at,created_at) VALUES ('newer',?,'x')", (newer,))
    conn.execute("INSERT INTO reminders (title,due_at,created_at) VALUES ('older',?,'x')", (older,))
    conn.commit()
    assert [r["title"] for r in rem.due_reminders(conn)] == ["older", "newer"]


# --- snooze --------------------------------------------------------------


def test_snooze_pushes_due_time_forward(tmp_path, monkeypatch):
    _tz(monkeypatch, "Asia/Dhaka")
    conn = connect(tmp_path / "s.db")
    rem.set_reminder_impl(conn, "Water plants", "2027-10-01 09:00")
    before = datetime.fromisoformat(
        conn.execute("SELECT due_at FROM reminders WHERE id=1").fetchone()["due_at"]
    )

    out = rem.snooze_reminder_impl(conn, 1, 45)
    after = datetime.fromisoformat(
        conn.execute("SELECT due_at FROM reminders WHERE id=1").fetchone()["due_at"]
    )

    assert (after - before).total_seconds() == 45 * 60
    assert "snoozed by 45 min" in out


def test_snooze_missing_id(tmp_path):
    conn = connect(tmp_path / "s.db")
    assert "No reminder with id 7" in rem.snooze_reminder_impl(conn, 7, 10)


def test_snooze_does_not_create_new_rows(tmp_path, monkeypatch):
    _tz(monkeypatch, "Asia/Dhaka")
    conn = connect(tmp_path / "s.db")
    rem.set_reminder_impl(conn, "X", "2027-10-01 09:00")
    rem.snooze_reminder_impl(conn, 1, 10)
    assert conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0] == 1


def test_snooze_across_day_boundary(tmp_path, monkeypatch):
    _tz(monkeypatch, "Asia/Dhaka")
    conn = connect(tmp_path / "s.db")
    rem.set_reminder_impl(conn, "Late", "2027-10-01 23:30")
    out = rem.snooze_reminder_impl(conn, 1, 60)
    assert "2027-10-02 00:30" in out
