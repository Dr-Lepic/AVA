"""Thread-safety regression tests for the storage layer.

Background: the Agents SDK runs synchronous tool functions via
asyncio.to_thread, i.e. on worker threads. A sqlite3 connection is bound to
its creating thread, so a single shared connection made *every* tool call
fail with "SQLite objects created in a thread can only be used in that same
thread".

The SDK converts a tool exception into model-visible text ("please try
again"), so the agent loop still completes and final_output looks fine — the
failure is invisible unless you check the tool output or the database. The
tests in test_agents_runs.py are what caught it; these pin the underlying
property so it cannot regress.
"""

import asyncio
import threading

from ava.db import close_thread_connections, connect, get_connection, write


def test_get_connection_is_stable_within_a_thread(tmp_path):
    a = get_connection(tmp_path / "a.db")
    b = get_connection(tmp_path / "a.db")
    assert a is b


def test_get_connection_differs_per_thread(tmp_path):
    path = tmp_path / "a.db"
    main_conn = get_connection(path)
    other = {}

    def grab():
        other["conn"] = get_connection(path)

    t = threading.Thread(target=grab)
    t.start()
    t.join()

    assert other["conn"] is not main_conn


def test_connection_is_usable_from_another_thread(tmp_path):
    """The exact failure this file exists to prevent."""
    path = tmp_path / "a.db"
    conn = get_connection(path)
    errors = []

    def use_it():
        try:
            conn.execute("SELECT 1").fetchone()
            write(conn, "INSERT INTO notes (title, created_at) VALUES (?, 'x')", ("from thread",))
        except Exception as exc:  # noqa: BLE001 - the point is to record it
            errors.append(exc)

    t = threading.Thread(target=use_it)
    t.start()
    t.join()

    assert errors == []
    assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1


def test_concurrent_writes_all_land(tmp_path):
    """Writes are serialised behind a lock; none may be lost."""
    conn = connect(tmp_path / "a.db")
    results = []

    def writer(n):
        for i in range(5):
            write(
                conn,
                "INSERT INTO notes (title, created_at) VALUES (?, 'x')",
                (f"t{n}-{i}",),
            )
        results.append(n)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 4
    assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 20


async def test_tool_runs_on_a_worker_thread_and_succeeds(tmp_path):
    """End-to-end: the SDK's to_thread path must produce a real write."""
    from agents import RunConfig, Runner
    from agents.testing import ScriptedModel, assistant_message, function_call

    from ava.agents import build_specialists

    model = ScriptedModel([
        [function_call("save_note", {"title": "Threaded", "body": ""}, call_id="c1")],
        [assistant_message("Saved.")],
    ])
    agent = build_specialists(
        tmp_path / "a.db", tmp_path / "vault", model=model
    )["notes"]

    result = await Runner.run(
        agent, "note this", run_config=RunConfig(tracing_disabled=True)
    )

    outputs = [
        item.output
        for item in result.new_items
        if type(item).__name__ == "ToolCallOutputItem"
    ]
    assert outputs, "no tool output item produced"
    assert "An error occurred" not in outputs[0], outputs[0]
    assert "Saved note" in outputs[0]
    model.assert_complete()


def test_close_thread_connections_is_safe_to_call(tmp_path):
    get_connection(tmp_path / "a.db")
    close_thread_connections()
    # After closing, a new call must transparently reopen.
    assert get_connection(tmp_path / "a.db") is not None
