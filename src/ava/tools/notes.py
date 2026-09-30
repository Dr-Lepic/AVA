"""Note storage.

These are plain functions over a sqlite3 connection. `as_tools` wraps them for
the agent; the tests call them directly, so no SDK involvement is needed to
verify the storage behaviour.
"""

from __future__ import annotations

import json
import sqlite3

from agents.decorators import tool

from ava.db import utc_now


def _like_pattern(query: str) -> str:
    """Escape LIKE wildcards so a user's '%' or '_' is matched literally.

    Without this, searching "100%" turns into a match-everything pattern and
    dumps the whole table into the model's context.
    """
    escaped = (
        query.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )
    return f"%{escaped}%"


def save_note_impl(
    conn: sqlite3.Connection, title: str, body: str = "", tags: str = ""
) -> str:
    conn.execute(
        "INSERT INTO notes (title, body, tags, created_at) VALUES (?, ?, ?, ?)",
        (title, body, tags, utc_now()),
    )
    conn.commit()
    note_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return f"Saved note #{note_id}: {title}"


def list_notes_impl(conn: sqlite3.Connection, limit: int = 20) -> str:
    """Recent notes as a summary list. Body is omitted to keep listings cheap."""
    rows = conn.execute(
        "SELECT id, title, tags, created_at FROM notes "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    if not rows:
        return "No notes saved yet."
    return json.dumps([dict(r) for r in rows], indent=2)


def search_notes_impl(conn: sqlite3.Connection, query: str) -> str:
    """Notes whose title or body matches, including the body text."""
    cleaned = query.strip()
    if not cleaned:
        return f"No notes matching {query!r}."

    pattern = _like_pattern(cleaned)
    rows = conn.execute(
        "SELECT id, title, body FROM notes "
        "WHERE title LIKE ? ESCAPE '\\' OR body LIKE ? ESCAPE '\\' "
        "ORDER BY id DESC",
        (pattern, pattern),
    ).fetchall()
    if not rows:
        return f"No notes matching {cleaned!r}."
    return json.dumps([dict(r) for r in rows], indent=2)


def as_tools(conn: sqlite3.Connection) -> list:
    """Bind the connection and return the three note tools.

    Two things the SDK forces here, both found by tests/test_tools_notes_schema.py:

    1. function_tool refuses functools.partial ("does not infer
       functools.partial contracts"), so the connection is closed over in thin
       wrapper functions instead of bound.
    2. strict_mode turns *every* parameter into a required one, including those
       with defaults, so an optional `body` would be demanded on every call.
       strict_mode=False is what makes `body`/`tags`/`limit` genuinely optional.
       The downside is that the JSON schema stops saying "no additional
       properties", which is a reasonable trade for these three read/write
       helpers.
    """

    def save_note(title: str, body: str = "", tags: str = "") -> str:
        """Save a note for the user. Use when they state something worth keeping."""
        return save_note_impl(conn, title, body, tags)

    def list_notes(limit: int = 20) -> str:
        """List the user's most recent notes, newest first."""
        return list_notes_impl(conn, limit)

    def search_notes(query: str) -> str:
        """Search notes by keyword across titles and bodies."""
        return search_notes_impl(conn, query)

    return [
        tool(save_note, strict_mode=False),
        tool(list_notes, strict_mode=False),
        tool(search_notes, strict_mode=False),
    ]
