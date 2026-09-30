"""SQLite storage for AVA. The only module that knows SQL."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

# The Agents SDK runs synchronous tool functions via asyncio.to_thread, i.e. on a
# worker thread. A sqlite3 connection is bound to the thread that created it, so
# a single shared connection would raise "SQLite objects created in a thread can
# only be used in that same thread" on every tool call — and the SDK converts that
# into a model-visible "please try again", so the run looks successful while
# nothing was written.
#
# Connections are therefore pooled per thread. SQLite handles concurrent readers
# fine, and writes are serialised behind a lock.
_local = threading.local()
_write_lock = threading.Lock()

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

    check_same_thread=False because the SDK invokes tool functions on worker
    threads. Writes still need `write()` below for safety.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def write(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> None:
    """Run a write and commit, serialised across threads.

    Use this instead of conn.execute + conn.commit for anything that mutates.
    """
    with _write_lock:
        conn.execute(sql, params)
        conn.commit()


def delete(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> sqlite3.Cursor:
    """Run a delete and commit, serialised across threads. Returns the cursor."""
    with _write_lock:
        cursor = conn.execute(sql, params)
        conn.commit()
    return cursor


def get_connection(db_path: Path | str) -> sqlite3.Connection:
    """Return this thread's connection to db_path, opening it on first use.

    Tools should call this per invocation rather than closing over a single
    connection created at startup.
    """
    path = Path(db_path)
    key = str(path)
    cache = getattr(_local, "conns", None)
    if not isinstance(cache, dict):
        cache = {}
        _local.conns = cache
    if key not in cache:
        cache[key] = connect(path)
    return cache[key]


def close_thread_connections() -> None:
    """Close and forget this thread's connections. For tests and shutdown."""
    cache: dict[str, sqlite3.Connection] = getattr(_local, "conns", {})
    for conn in cache.values():
        try:
            conn.close()
        except sqlite3.Error:
            pass
    _local.conns = {}


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string.

    All timestamps are stored in UTC; conversion to local time happens at the
    edge, in the reminder tools.
    """
    return datetime.now(timezone.utc).isoformat()
