"""Email drafts.

This module NEVER sends mail. It writes an `.eml` file to the vault and a row
in SQLite, and every return value says so explicitly so the model cannot imply
otherwise.

Storage functions are suffixed `_impl`; `as_tools` wraps them under the public
names the agent instructions use.
"""

from __future__ import annotations

import json
import sqlite3
from email.message import EmailMessage
from pathlib import Path

from agents.decorators import tool

from ava.db import get_connection, utc_now, write

# Characters that are illegal in filenames on at least one supported platform,
# plus path separators, so a subject can never steer the write outside the vault.
_UNSAFE_FILENAME_CHARS = '/\\:*?"<>|'
_MAX_SUBJECT_IN_FILENAME = 40


def _safe_filename_part(subject: str) -> str:
    """Reduce a subject to a filename fragment that cannot escape the vault."""
    cleaned = "".join(
        c if (c.isalnum() or c in "._-") else "_" for c in subject
    ).strip("._")
    return (cleaned or "draft")[:_MAX_SUBJECT_IN_FILENAME]


def create_draft_impl(
    conn: sqlite3.Connection,
    subject: str,
    body: str,
    to_addr: str = "",
    vault_path: Path | None = None,
) -> str:
    """Store an email draft. Does not send anything."""
    write(
        conn,
        "INSERT INTO drafts (to_addr, subject, body, created_at) VALUES (?,?,?,?)",
        (to_addr, subject, body, utc_now()),
    )
    draft_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    if vault_path is not None:
        vault_path.mkdir(parents=True, exist_ok=True)
        msg = EmailMessage()
        msg["Subject"] = subject
        if to_addr:
            msg["To"] = to_addr
        msg.set_content(body)

        filename = f"draft-{draft_id:03d}-{_safe_filename_part(subject)}.eml"
        (vault_path / filename).write_bytes(msg.as_bytes())

    return (
        f"Draft #{draft_id} saved"
        + (f" for {to_addr}" if to_addr else "")
        + ". NOT SENT — review and send it yourself."
    )


def list_drafts_impl(conn: sqlite3.Connection, limit: int = 20) -> str:
    """Recent drafts as a summary list. Body is omitted to keep listings cheap."""
    rows = conn.execute(
        "SELECT id, to_addr, subject, created_at FROM drafts "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    if not rows:
        return "No drafts yet."
    return json.dumps([dict(r) for r in rows], indent=2)


def as_tools(db_path: Path | str, vault_path: Path) -> list:
    """Bind the database path and vault, and return the two draft tools.

    The vault path is closed over: it must never be a model-supplied
    argument, since a model that chose where drafts are written could write
    anywhere on disk. The database path is likewise closed over, and the
    connection is resolved per call, because the SDK runs tool functions on
    worker threads and a sqlite3 connection is bound to its creating thread.

    strict_mode=False keeps `body` and `to_addr` genuinely optional.
    """

    def create_draft(subject: str, body: str = "", to_addr: str = "") -> str:
        """Write an email draft to the user's vault. This does NOT send mail."""
        return create_draft_impl(get_connection(db_path), subject, body, to_addr, vault_path)

    def list_drafts(limit: int = 20) -> str:
        """List the user's recent email drafts, newest first. Never sends."""
        return list_drafts_impl(get_connection(db_path), limit)

    return [
        tool(create_draft, strict_mode=False),
        tool(list_drafts, strict_mode=False),
    ]
