from datetime import datetime, timezone

from ava.db import connect, utc_now

EXPECTED_TABLES = {"notes", "drafts", "reminders"}


def _table_names(conn):
    return {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def test_connect_creates_tables(tmp_path):
    conn = connect(tmp_path / "test.db")
    assert EXPECTED_TABLES <= _table_names(conn)


def test_connect_is_idempotent(tmp_path):
    """Opening twice must not fail or wipe existing rows."""
    path = tmp_path / "twice.db"
    first = connect(path)
    first.execute("INSERT INTO notes (title, body, created_at) VALUES ('keep', '', 'x')")
    first.commit()
    first.close()

    second = connect(path)
    assert second.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1


def test_rows_are_accessible_by_name(tmp_path):
    """row_factory must be on, or every consumer would use index access."""
    conn = connect(tmp_path / "rows.db")
    conn.execute("INSERT INTO notes (title, created_at) VALUES ('hello', 'x')")
    conn.commit()
    row = conn.execute("SELECT title FROM notes").fetchone()
    assert row["title"] == "hello"


def test_utc_now_is_iso_8601():
    value = utc_now()
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None


def test_utc_now_is_utc():
    parsed = datetime.fromisoformat(utc_now())
    assert abs((parsed - datetime.now(timezone.utc)).total_seconds()) < 5


def test_defaults_applied_on_insert(tmp_path):
    """body, tags, done must default so callers can omit them."""
    conn = connect(tmp_path / "defaults.db")
    conn.execute("INSERT INTO notes (title, created_at) VALUES ('t', 'x')")
    conn.execute("INSERT INTO drafts (subject, created_at) VALUES ('s', 'x')")
    conn.execute("INSERT INTO reminders (title, due_at, created_at) VALUES ('r', 'x', 'x')")
    conn.commit()

    note = conn.execute("SELECT body, tags FROM notes").fetchone()
    assert (note["body"], note["tags"]) == ("", "")

    draft = conn.execute("SELECT to_addr, body FROM drafts").fetchone()
    assert (draft["to_addr"], draft["body"]) == ("", "")

    reminder = conn.execute("SELECT done FROM reminders").fetchone()
    assert reminder["done"] == 0


def test_ids_autoincrement(tmp_path):
    conn = connect(tmp_path / "ids.db")
    for i in range(3):
        conn.execute("INSERT INTO notes (title, created_at) VALUES (?, 'x')", (f"n{i}",))
    conn.commit()
    ids = [r["id"] for r in conn.execute("SELECT id FROM notes ORDER BY id")]
    assert ids == [1, 2, 3]
