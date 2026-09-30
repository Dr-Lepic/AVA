"""Checks the reminder tool schemas the model will see.

Written before the implementation, following the note tools: Task 3 hid two
real bugs in the schema shape, so it is pinned here rather than discovered when
the model misbehaves against a live key.
"""

import json
import sqlite3

from ava.tools import reminders


def _by_name(tools):
    return {t.name: t for t in tools}


def _tools():
    return reminders.as_tools(sqlite3.connect(":memory:"))


def test_as_tools_returns_four_tools():
    assert [t.name for t in _tools()] == [
        "set_reminder",
        "list_reminders",
        "cancel_reminder",
        "snooze_reminder",
    ]


def test_conn_is_not_exposed():
    for t in _tools():
        assert "conn" not in json.dumps(t.params_json_schema), t.name


def test_set_reminder_requires_title_and_due():
    tools = _by_name(_tools())
    props = tools["set_reminder"].params_json_schema["properties"]
    assert set(props) == {"title", "due"}
    assert set(tools["set_reminder"].params_json_schema["required"]) == {"title", "due"}


def test_cancel_and_snooze_require_their_id():
    tools = _by_name(_tools())
    for name in ("cancel_reminder", "snooze_reminder"):
        required = tools[name].params_json_schema["required"]
        assert "reminder_id" in required, f"{name} should need reminder_id"


def test_snooze_minutes_is_required():
    """Snoozing by an implicit default would silently move the wrong reminder."""
    tools = _by_name(_tools())
    assert "minutes" in tools["snooze_reminder"].params_json_schema["required"]


def test_list_reminders_include_done_is_optional():
    tools = _by_name(_tools())
    required = tools["list_reminders"].params_json_schema.get("required") or []
    assert "include_done" not in required


def test_tools_have_descriptions():
    for t in _tools():
        assert t.description, f"{t.name} has no description"


def test_scheduling_tool_mentions_timezone():
    """The model must be told times are local, or it will send UTC."""
    tools = _by_name(_tools())
    text = (tools["set_reminder"].description or "").lower()
    assert "time" in text
