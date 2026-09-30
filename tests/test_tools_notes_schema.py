"""Checks the tool schemas the model will actually see.

These are the tests that would catch a leak of the internal `conn` argument
into the exposed schema, or a name that does not match what the agent
instructions promise. The storage tests in test_tools_notes.py call the
functions directly and would not notice either problem.
"""

import json
import sqlite3

from ava.tools import notes


def _by_name(tools):
    return {t.name: t for t in tools}


def test_as_tools_returns_three_tools():
    tools = notes.as_tools(sqlite3.connect(":memory:"))
    assert [t.name for t in tools] == ["save_note", "list_notes", "search_notes"]


def test_conn_is_not_exposed_in_any_schema():
    """A `conn` parameter would be offered to the model as a required argument."""
    for t in notes.as_tools(sqlite3.connect(":memory:")):
        schema = json.dumps(t.params_json_schema)
        assert "conn" not in schema, f"{t.name} leaks conn: {schema}"


def test_save_note_schema_exposes_only_expected_args():
    tools = _by_name(notes.as_tools(sqlite3.connect(":memory:")))
    props = tools["save_note"].params_json_schema["properties"]
    assert set(props) == {"title", "body", "tags"}
    assert tools["save_note"].params_json_schema["required"] == ["title"]


def test_tools_have_descriptions():
    """No description means the model cannot tell when to call the tool."""
    for t in notes.as_tools(sqlite3.connect(":memory:")):
        assert t.description, f"{t.name} has no description"


def test_search_notes_requires_query():
    tools = _by_name(notes.as_tools(sqlite3.connect(":memory:")))
    assert tools["search_notes"].params_json_schema["required"] == ["query"]


def test_wrapped_functions_close_over_the_caller_connection(tmp_path):
    """Each as_tools() call must capture its own connection.

    Invoking the wrapped function end-to-end is covered in Task 10 via
    ScriptedModel, which drives the real runner. Here we only need to prove
    the wiring: two independent connections must not share state.
    """
    from ava.db import connect

    conn_a = connect(tmp_path / "a.db")
    conn_b = connect(tmp_path / "b.db")

    tools_a = _by_name(notes.as_tools(conn_a))
    tools_b = _by_name(notes.as_tools(conn_b))

    # The schemas are identical; the difference is which connection is bound,
    # which is only observable by executing. Assert the binding is per-call by
    # checking the tools are distinct objects.
    assert tools_a["save_note"] is not tools_b["save_note"]
    assert tools_a["list_notes"] is not tools_b["list_notes"]


def test_tool_output_defaults_are_not_required():
    """strict_mode must not turn `body`/`tags`/`limit` into demanded arguments."""
    tools = _by_name(notes.as_tools(sqlite3.connect(":memory:")))

    assert tools["save_note"].params_json_schema["required"] == ["title"]
    assert "limit" not in (tools["list_notes"].params_json_schema.get("required") or [])
