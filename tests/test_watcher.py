"""The reminder watcher.

Pure local logic: polls the database, fires a macOS notification, marks the
reminder done. No model calls, so this costs nothing against the free-tier
request budget.

Two properties matter beyond "does it fire":

- it must not re-fire. A reminder that pops up every 30 seconds is worse than
  one that never fires, because the user learns to ignore notifications.
- it must not wedge. The poll loop runs for the life of the CLI, so a hung
  notification or an unexpected error has to be survivable.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from ava import watcher
from ava.db import connect


def _insert_due(conn, title="Standup", minutes_ago=1):
    past = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    write = conn.execute
    write(
        "INSERT INTO reminders (title,due_at,created_at) VALUES (?,?,?)",
        (title, past, "x"),
    )
    conn.commit()


def _insert_future(conn, title="Later", hours=5):
    future = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()
    conn.execute(
        "INSERT INTO reminders (title,due_at,created_at) VALUES (?,?,?)",
        (title, future, "x"),
    )
    conn.commit()


@pytest.fixture
def notified(monkeypatch):
    """Capture notifications instead of showing them."""
    seen = []
    monkeypatch.setattr(
        watcher.notifier, "notify", lambda title, message: seen.append(message) or True
    )
    return seen


# --- check_once ----------------------------------------------------------


async def test_check_once_fires_and_marks_done(tmp_path, notified):
    conn = connect(tmp_path / "w.db")
    _insert_due(conn)

    assert await watcher.check_once(conn) == 1
    assert notified == ["Standup"]
    assert watcher.due_reminders(conn) == []


async def test_check_once_does_not_refire(tmp_path, notified):
    """The important one: a reminder must not repeat every tick."""
    conn = connect(tmp_path / "w.db")
    _insert_due(conn)

    assert await watcher.check_once(conn) == 1
    assert await watcher.check_once(conn) == 0
    assert notified == ["Standup"]


async def test_check_once_ignores_future_reminders(tmp_path, notified):
    conn = connect(tmp_path / "w.db")
    _insert_future(conn)
    assert await watcher.check_once(conn) == 0
    assert notified == []


async def test_check_once_with_nothing_due(tmp_path, notified):
    conn = connect(tmp_path / "w.db")
    assert await watcher.check_once(conn) == 0


async def test_check_once_handles_several_due_reminders(tmp_path, notified):
    conn = connect(tmp_path / "w.db")
    _insert_due(conn, "First", minutes_ago=30)
    _insert_due(conn, "Second", minutes_ago=20)
    _insert_due(conn, "Third", minutes_ago=10)

    assert await watcher.check_once(conn) == 3
    # Oldest first, so a backlog notifies in the order it built up.
    assert notified == ["First", "Second", "Third"]


async def test_check_once_still_marks_done_when_notify_fails(tmp_path, monkeypatch):
    """A failed notification must not leave the reminder firing forever."""
    monkeypatch.setattr(watcher.notifier, "notify", lambda t, m: False)
    conn = connect(tmp_path / "w.db")
    _insert_due(conn)

    assert await watcher.check_once(conn) == 1
    assert watcher.due_reminders(conn) == []


async def test_check_once_falls_back_to_printing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(watcher.notifier, "notify", lambda t, m: False)
    conn = connect(tmp_path / "w.db")
    _insert_due(conn)

    await watcher.check_once(conn)
    assert "Standup" in capsys.readouterr().out


# --- run_forever ---------------------------------------------------------


async def test_run_forever_stops_on_event(tmp_path, monkeypatch, notified):
    monkeypatch.setattr(watcher, "POLL_SECONDS", 0.01)
    conn = connect(tmp_path / "w.db")

    stop = asyncio.Event()
    task = asyncio.create_task(watcher.run_forever(conn, stop))
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, timeout=2)


async def test_run_forever_polls_repeatedly(tmp_path, monkeypatch, notified):
    """A reminder added while running must still be caught."""
    monkeypatch.setattr(watcher, "POLL_SECONDS", 0.01)
    conn = connect(tmp_path / "w.db")

    stop = asyncio.Event()
    task = asyncio.create_task(watcher.run_forever(conn, stop))
    await asyncio.sleep(0.03)
    _insert_due(conn)
    await asyncio.sleep(0.06)
    stop.set()
    await asyncio.wait_for(task, timeout=2)

    assert "Standup" in notified


async def test_run_forever_survives_a_check_error(tmp_path, monkeypatch):
    """One bad tick must not kill the watcher for the rest of the session."""
    monkeypatch.setattr(watcher, "POLL_SECONDS", 0.01)
    conn = connect(tmp_path / "w.db")

    calls = {"n": 0}
    real_check = watcher.check_once

    async def flaky(conn):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return await real_check(conn)

    monkeypatch.setattr(watcher, "check_once", flaky)

    stop = asyncio.Event()
    task = asyncio.create_task(watcher.run_forever(conn, stop))
    await asyncio.sleep(0.08)
    stop.set()
    await asyncio.wait_for(task, timeout=2)

    assert calls["n"] > 1, "watcher stopped after one error"
