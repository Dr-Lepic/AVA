"""Reminder storage.

Naive datetimes are read in the user's configured timezone and stored as UTC.
Storage is *_impl; as_tools() wraps them under the names the agent uses.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from agents.decorators import tool

from ava.config import DEFAULT_TZ
from ava.db import utc_now


class ReminderError(ValueError):
    """Raised when a due date cannot be parsed or a timezone is invalid."""


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


def _format_local(moment_utc: datetime, tz: ZoneInfo) -> str:
    """Render a UTC instant in the user's zone, for echoing back to them."""
    return moment_utc.astimezone(tz).strftime("%Y-%m-%d %H:%M")


def parse_due(due: str, tz_name: str | None = None) -> datetime:
    """Parse a due time into an absolute UTC datetime.

    Accepts ISO-8601 with a time component. A naive value
    ('2026-10-01 09:00') is interpreted in the user's configured timezone, so
    "9am" means 9am where they are.

    A date with no time ('2026-10-01') is rejected: it is ambiguous between
    midnight and all-day, and guessing midnight would fire a day early.
    """
    text = due.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ReminderError(
            f"Could not parse due time {due!r}. Use ISO-8601 with a time, e.g. "
            "'2026-10-01T09:00:00' or '2026-10-01 09:00' (local time)."
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
    due_utc = parse_due(due)

    conn.execute(
        "INSERT INTO reminders (title, due_at, created_at) VALUES (?,?,?)",
        (title, due_utc.isoformat(), utc_now()),
    )
    conn.commit()
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
    cursor = conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
    conn.commit()
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
    conn.execute(
        "UPDATE reminders SET due_at = ? WHERE id = ?",
        (new_due.isoformat(), reminder_id),
    )
    conn.commit()

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
    conn.execute("UPDATE reminders SET done = 1 WHERE id = ?", (reminder_id,))
    conn.commit()


def as_tools(conn: sqlite3.Connection) -> list:
    """Bind the connection and return the four reminder tools.

    The connection is closed over rather than bound: function_tool rejects
    functools.partial, and a `conn` argument must never reach the model.
    """

    def set_reminder(title: str, due: str) -> str:
        """Set a reminder. `due` is local time as ISO-8601, e.g. 2026-10-01T09:00:00."""
        return set_reminder_impl(conn, title, due)

    def list_reminders(include_done: bool = False) -> str:
        """List reminders the user has set, earliest first."""
        return list_reminders_impl(conn, include_done)

    def cancel_reminder(reminder_id: int) -> str:
        """Delete a reminder by its numeric id. Use list_reminders to find ids."""
        return cancel_reminder_impl(conn, reminder_id)

    def snooze_reminder(reminder_id: int, minutes: int) -> str:
        """Delay an existing reminder by a number of minutes, by numeric id."""
        return snooze_reminder_impl(conn, reminder_id, minutes)

    return [
        tool(set_reminder, strict_mode=False),
        tool(list_reminders, strict_mode=False),
        tool(cancel_reminder, strict_mode=False),
        tool(snooze_reminder, strict_mode=False),
    ]
