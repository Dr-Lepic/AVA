"""SQLite storage for AVA. The only module that knows SQL."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT    NOT NULL,
    body       TEXT    NOT NULL DEFAULT '',
    tags       TEXT    NOT NULL DEFAULT '',
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS drafts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    to_addr    TEXT NOT NULL DEFAULT '',
    subject    TEXT NOT NULL,
    body       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reminders (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT    NOT NULL,
    due_at     TEXT    NOT NULL,
    done       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders (done, due_at);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open a connection with row access by name and the schema applied.

    Safe to call on an existing database: the schema uses IF NOT EXISTS, so
    reopening preserves existing rows. Accepts a str for convenience and
    creates the parent directory if missing, so a first run against a fresh
    AVA_HOME cannot fail here.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string.

    All timestamps are stored in UTC; conversion to local time happens at the
    edge, in the reminder tools.
    """
    return datetime.now(timezone.utc).isoformat()
