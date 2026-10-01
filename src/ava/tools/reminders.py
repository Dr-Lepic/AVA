"""Reminder storage.

Naive datetimes are read in the user's configured timezone and stored as UTC.
Storage is *_impl; as_tools() wraps them under the names the agent uses.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from agents.decorators import tool

from ava.config import DEFAULT_TZ
from ava.db import delete, get_connection, utc_now, write


class ReminderError(ValueError):
    """Raised when a due date cannot be parsed or a timezone is invalid."""


class PastDueError(ReminderError):
    """Raised when a reminder resolves to a time that has already passed.

    Separate from ReminderError because the message should tell the model what
    the real current time is, so it can retry instead of guessing again.
    """


def _local_tz(tz_name: str | None = None) -> ZoneInfo:
    """Resolve the timezone used for reminder math.

    Prefers an explicitly passed name, then AVA_TZ, then the default.
    """
    name = tz_name or os.environ.get("AVA_TZ") or DEFAULT_TZ
    try:
        return ZoneInfo(name)
    except Exception as exc:
        raise ReminderError(
            f"Unknown timezone {name!r}. Use an IANA name like 'Asia/Dhaka'."
        ) from exc


# Relative phrases are resolved here rather than by the model. The model has
# no clock: asked for "tomorrow" it computes from its training data, which may
# be years stale, and the reminder lands in the past.
_UNITS = {
    "second": "seconds", "seconds": "seconds", "sec": "seconds", "s": "seconds",
    "minute": "minutes", "minutes": "minutes", "min": "minutes", "m": "minutes",
    "hour": "hours", "hours": "hours", "hr": "hours", "h": "hours",
    "day": "days", "days": "days", "d": "days",
    "week": "weeks", "weeks": "weeks", "w": "weeks",
}

_IN_RE = re.compile(r"^in\s+(\d+)\s+([a-z]+)$")


def _parse_clock(remainder: str) -> tuple[int, int] | None:
    """Pull an hour and minute out of '9am', '9:30pm', '21:00', or '9'.

    Handles a space before the meridiem ("3:30 pm") by removing spaces first.
    """
    text = remainder.strip().rstrip(".").replace(" ", "").lower()
    if not text:
        return None

    meridiem = None
    for suffix in ("am", "pm"):
        if text.endswith(suffix):
            meridiem = suffix
            text = text[: -len(suffix)]
            break

    if ":" in text:
        head, _, tail = text.partition(":")
        if not head.isdigit() or not tail.isdigit():
            return None
        hour, minute = int(head), int(tail)
    elif text.isdigit():
        hour, minute = int(text), 0
    else:
        return None

    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    return hour % 24, minute % 60


def parse_relative(text: str, tz_name: str | None = None) -> datetime | None:
    """Resolve 'in 20 minutes' against the real clock.

    Returns None when the text is not a relative phrase, so the caller can
    fall through to absolute parsing rather than guessing.
    """
    match = _IN_RE.match(text.strip().lower())
    if not match:
        return None
    key = _UNITS.get(match.group(2))
    if key is None:
        return None
    return datetime.now(_local_tz(tz_name)) + timedelta(**{key: int(match.group(1))})


def parse_today_word(text: str, tz_name: str | None = None) -> datetime | None:
    """Resolve 'today 9am' / 'tomorrow 9am' / 'day after tomorrow 9am'."""
    lowered = text.strip().lower()
    for word, offset in (("day after tomorrow", 2), ("tomorrow", 1), ("today", 0)):
        if lowered.startswith(word):
            clock = _parse_clock(lowered[len(word) :])
            if clock is None:
                return None
            base = datetime.now(_local_tz(tz_name)) + timedelta(days=offset)
            return base.replace(
                hour=clock[0], minute=clock[1], second=0, microsecond=0
            )
    return None


def _format_local(moment_utc: datetime, tz: ZoneInfo) -> str:
    """Render a UTC instant in the user's zone, for echoing back to them."""
    return moment_utc.astimezone(tz).strftime("%Y-%m-%d %H:%M")


def parse_due(due: str, tz_name: str | None = None) -> datetime:
    """Parse a due time into an absolute UTC datetime.

    Three forms are accepted, tried in order:

    1. a relative phrase — "in 20 minutes"
    2. a day word — "tomorrow 9am", "today 15:30"
    3. absolute ISO-8601 — "2026-10-01T09:00:00"

    The first two are resolved against the real clock. This is the point:
    the model has no clock, so a "tomorrow" it computes itself lands in the
    past. It should pass the user's words through untouched instead.

    A naive absolute value is read in the user's configured timezone. A value
    that has already passed is refused — see set_reminder_impl.
    """
    text = due.strip()

    resolved = parse_relative(text, tz_name)
    if resolved is None:
        resolved = parse_today_word(text, tz_name)
    if resolved is not None:
        return resolved.astimezone(timezone.utc)

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ReminderError(
            f"Could not parse due time {due!r}. Use a relative phrase like "
            "'in 20 minutes' or 'tomorrow 9am', or ISO-8601 like "
            "'2026-10-01 09:00'."
        ) from exc

    # datetime.fromisoformat accepts a bare date, yielding midnight. Reject it
    # so the caller must state a time rather than inherit a silent assumption.
    if parsed.hour == 0 and parsed.minute == 0 and len(text) <= 10:
        raise ReminderError(
            f"Due time {due!r} has no time of day. Include one, e.g. "
            f"'{text}T09:00:00'."
        )

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_local_tz(tz_name))

    return parsed.astimezone(timezone.utc)


def set_reminder_impl(conn: sqlite3.Connection, title: str, due: str) -> str:
    tz = _local_tz()
    now = datetime.now(tz)
    due_utc = parse_due(due)

    # Safety net. The relative and day-word forms above handle the common
    # cases, but the model can still pass an absolute date computed from a
    # stale idea of what today is. Such a reminder is due immediately and
    # fires on the next poll, so refuse it and tell the model the real time.
    if due_utc <= now.astimezone(timezone.utc):
        raise PastDueError(
            f"{due_utc.astimezone(tz):%Y-%m-%d %H:%M} is in the past. "
            f"The current time is {now:%Y-%m-%d %H:%M}. "
            f"Pass a relative phrase like 'in 20 minutes' or 'tomorrow 9am', "
            f"or an ISO-8601 date in the future."
        )

    write(
        conn,
        "INSERT INTO reminders (title, due_at, created_at) VALUES (?,?,?)",
        (title, due_utc.isoformat(), utc_now()),
    )
    rem_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    return (
        f"Reminder #{rem_id} set: {title} at {_format_local(due_utc, tz)} "
        f"local (stored {due_utc.isoformat()})."
    )


def list_reminders_impl(conn: sqlite3.Connection, include_done: bool = False) -> str:
    sql = "SELECT id, title, due_at, done FROM reminders"
    if not include_done:
        sql += " WHERE done = 0"
    sql += " ORDER BY due_at ASC"
    rows = conn.execute(sql).fetchall()
    if not rows:
        return "No reminders set."
    return json.dumps([dict(r) for r in rows], indent=2)


def cancel_reminder_impl(conn: sqlite3.Connection, reminder_id: int) -> str:
    cursor = delete(conn, "DELETE FROM reminders WHERE id = ?", (reminder_id,))
    if cursor.rowcount == 0:
        return f"No reminder with id {reminder_id}."
    return f"Reminder #{reminder_id} cancelled."


def snooze_reminder_impl(
    conn: sqlite3.Connection, reminder_id: int, minutes: int
) -> str:
    """Push a reminder's due time later, for 'remind me in 10 minutes'."""
    row = conn.execute(
        "SELECT title, due_at FROM reminders WHERE id = ?", (reminder_id,)
    ).fetchone()
    if row is None:
        return f"No reminder with id {reminder_id}."

    new_due = datetime.fromisoformat(row["due_at"]) + timedelta(minutes=minutes)
    write(
        conn,
        "UPDATE reminders SET due_at = ? WHERE id = ?",
        (new_due.isoformat(), reminder_id),
    )

    return (
        f"Reminder #{reminder_id} ({row['title']}) snoozed by {minutes} min, "
        f"now due {_format_local(new_due, _local_tz())} local."
    )


def due_reminders(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Rows due at or before now and not yet done. Used by the watcher.

    The bound is inclusive so a reminder due exactly now fires this tick
    rather than waiting for the next one.
    """
    now = datetime.now(timezone.utc).isoformat()
    return conn.execute(
        "SELECT id, title, due_at FROM reminders "
        "WHERE done = 0 AND due_at <= ? ORDER BY due_at ASC",
        (now,),
    ).fetchall()


def mark_done(conn: sqlite3.Connection, reminder_id: int) -> None:
    write(conn, "UPDATE reminders SET done = 1 WHERE id = ?", (reminder_id,))


def as_tools(db_path: Path | str) -> list:
    """Bind the database path and return the four reminder tools.

    The connection is resolved per call via get_connection, because the SDK
    invokes tool functions on worker threads and a sqlite3 connection is
    bound to its creating thread. It is never a model-supplied argument.
    """

    def set_reminder(title: str, due: str) -> str:
        """Set a reminder. `due` is local time as ISO-8601, e.g. 2026-10-01T09:00:00."""
        return set_reminder_impl(get_connection(db_path), title, due)

    def list_reminders(include_done: bool = False) -> str:
        """List reminders the user has set, earliest first."""
        return list_reminders_impl(get_connection(db_path), include_done)

    def cancel_reminder(reminder_id: int) -> str:
        """Delete a reminder by its numeric id. Use list_reminders to find ids."""
        return cancel_reminder_impl(get_connection(db_path), reminder_id)

    def snooze_reminder(reminder_id: int, minutes: int) -> str:
        """Delay an existing reminder by a number of minutes, by numeric id."""
        return snooze_reminder_impl(get_connection(db_path), reminder_id, minutes)

    return [
        tool(set_reminder, strict_mode=False),
        tool(list_reminders, strict_mode=False),
        tool(cancel_reminder, strict_mode=False),
        tool(snooze_reminder, strict_mode=False),
    ]
