"""In-process reminder watcher.

Polls the database and fires a macOS notification for anything due. It lives
inside the CLI process: there is no daemon, so reminders only fire while AVA
is running. Anything missed is reported on the next start by the CLI.

This costs nothing against the free-tier request budget — no model calls.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ava import notifier
from ava.db import get_connection
from ava.tools.reminders import due_reminders, mark_done

POLL_SECONDS = 30

TITLE = "AVA — reminder"


async def check_once(db_path: Path | str) -> int:
    """Notify for every due reminder and mark each done. Returns how many fired.

    Takes a database path and resolves a connection per call: tools run on
    worker threads and a sqlite3 connection is bound to its creating thread.

    A reminder is marked done whether or not the notification was delivered.
    Leaving it pending on a failed notification would make it fire again on
    every tick, which trains the user to ignore reminders.
    """
    conn = get_connection(db_path)
    fired = 0
    for row in due_reminders(conn):
        delivered = notifier.notify(TITLE, row["title"])
        mark_done(conn, row["id"])
        fired += 1
        if not delivered:
            # No notification available (non-macOS, or osascript failed).
            # Print it so the reminder is not lost silently.
            print(f"[reminder] {row['title']}")
    return fired


async def run_forever(db_path: Path | str, stop: asyncio.Event) -> None:
    """Poll for due reminders until `stop` is set.

    A failing tick is logged and retried rather than propagated: this task
    runs for the life of the CLI, and one transient error should not silently
    disable reminders for the rest of the session.
    """
    while not stop.is_set():
        try:
            await check_once(db_path)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the loop must survive
            print(f"[reminder] watcher error: {type(exc).__name__}: {exc}")

        try:
            await asyncio.wait_for(stop.wait(), timeout=POLL_SECONDS)
        except asyncio.TimeoutError:
            continue
