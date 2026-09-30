import json

from ava.db import connect
from ava.tools import notes


def test_save_then_list_notes(tmp_path):
    conn = connect(tmp_path / "n.db")
    result = notes.save_note_impl(conn, "Groceries", "milk, eggs", "home")
    assert "Saved note #1" in result

    listed = json.loads(notes.list_notes_impl(conn))
    assert len(listed) == 1
    assert listed[0]["title"] == "Groceries"


def test_save_note_optional_args(tmp_path):
    """body and tags must be optional — a title-only note is the common case."""
    conn = connect(tmp_path / "n.db")
    notes.save_note_impl(conn, "Just a title")
    listed = json.loads(notes.list_notes_impl(conn))
    assert listed[0]["title"] == "Just a title"


def test_list_notes_excludes_body(tmp_path):
    """list is a summary view; body belongs in search or a single read."""
    conn = connect(tmp_path / "n.db")
    notes.save_note_impl(conn, "T", "secret body text")
    listed = json.loads(notes.list_notes_impl(conn))
    assert "body" not in listed[0]


def test_list_notes_newest_first(tmp_path):
    conn = connect(tmp_path / "n.db")
    for t in ("first", "second", "third"):
        notes.save_note_impl(conn, t)
    listed = json.loads(notes.list_notes_impl(conn))
    assert [n["title"] for n in listed] == ["third", "second", "first"]


def test_list_notes_respects_limit(tmp_path):
    conn = connect(tmp_path / "n.db")
    for i in range(5):
        notes.save_note_impl(conn, f"note{i}")
    assert len(json.loads(notes.list_notes_impl(conn, limit=2))) == 2


def test_list_notes_empty_returns_message(tmp_path):
    conn = connect(tmp_path / "n.db")
    assert "No notes" in notes.list_notes_impl(conn)


def test_search_matches_title(tmp_path):
    conn = connect(tmp_path / "n.db")
    notes.save_note_impl(conn, "Idea", "unrelated body")
    notes.save_note_impl(conn, "Shopping", "buy milk")
    found = json.loads(notes.search_notes_impl(conn, "Shop"))
    assert len(found) == 1
    assert found[0]["title"] == "Shopping"


def test_search_matches_body(tmp_path):
    conn = connect(tmp_path / "n.db")
    notes.save_note_impl(conn, "Idea", "build a weather bot")
    notes.save_note_impl(conn, "Other", "unrelated")
    found = json.loads(notes.search_notes_impl(conn, "weather"))
    assert len(found) == 1
    assert found[0]["title"] == "Idea"


def test_search_is_case_insensitive(tmp_path):
    """SQLite LIKE is case-insensitive for ASCII by default; pin that."""
    conn = connect(tmp_path / "n.db")
    notes.save_note_impl(conn, "Weather", "x")
    assert len(json.loads(notes.search_notes_impl(conn, "weather"))) == 1


def test_search_includes_body_in_results(tmp_path):
    conn = connect(tmp_path / "n.db")
    notes.save_note_impl(conn, "T", "findable text")
    found = json.loads(notes.search_notes_impl(conn, "findable"))
    assert found[0]["body"] == "findable text"


def test_search_no_match_returns_message(tmp_path):
    conn = connect(tmp_path / "n.db")
    assert "No notes matching" in notes.search_notes_impl(conn, "zzz")


def test_search_empty_query_returns_message(tmp_path):
    """A bare LIKE '%%' would otherwise dump the whole table to the model."""
    conn = connect(tmp_path / "n.db")
    notes.save_note_impl(conn, "T", "x")
    assert "No notes matching" in notes.search_notes_impl(conn, "")


def test_wildcards_in_query_are_escaped(tmp_path):
    """A '%' from the user must not turn into a match-everything pattern."""
    conn = connect(tmp_path / "n.db")
    notes.save_note_impl(conn, "100% done", "x")
    notes.save_note_impl(conn, "unrelated", "y")
    found = json.loads(notes.search_notes_impl(conn, "100%"))
    assert len(found) == 1
    assert found[0]["title"] == "100% done"
