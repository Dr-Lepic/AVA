"""Checks the draft tool schemas the model will see.

Two bugs lived in exactly this area during Task 3, so the shape is pinned here
rather than discovered at runtime:

- `conn` and `vault_path` must not appear as model-supplied arguments. The
  vault in particular is a filesystem path; a model that could choose it could
  aim a write anywhere on the machine.
- optional arguments must stay optional, which needs strict_mode=False.
"""

import json
import sqlite3
from pathlib import Path

from ava.tools import drafts


def _by_name(tools):
    return {t.name: t for t in tools}


def _tools(tmp_path):
    return drafts.as_tools(sqlite3.connect(":memory:"), tmp_path / "vault")


def test_as_tools_returns_two_tools(tmp_path):
    assert [t.name for t in _tools(tmp_path)] == ["create_draft", "list_drafts"]


def test_no_internal_argument_is_exposed(tmp_path):
    for t in _tools(tmp_path):
        schema = json.dumps(t.params_json_schema)
        assert "conn" not in schema, f"{t.name} leaks conn"
        assert "vault_path" not in schema, f"{t.name} leaks vault_path"


def test_create_draft_args_are_optional_where_expected(tmp_path):
    tools = _by_name(_tools(tmp_path))
    props = tools["create_draft"].params_json_schema["properties"]
    assert set(props) == {"subject", "body", "to_addr"}
    assert tools["create_draft"].params_json_schema["required"] == ["subject"]


def test_list_drafts_limit_is_optional(tmp_path):
    tools = _by_name(_tools(tmp_path))
    assert "limit" not in (tools["list_drafts"].params_json_schema.get("required") or [])


def test_tools_have_descriptions(tmp_path):
    for t in _tools(tmp_path):
        assert t.description, f"{t.name} has no description"


def test_drafting_tool_description_says_never_sends(tmp_path):
    """The model must know it cannot send, or it will claim it did."""
    tools = _by_name(_tools(tmp_path))
    text = (tools["create_draft"].description or "").lower()
    assert "draft" in text
    assert "not send" in text or "never send" in text or "does not send" in text


def test_each_call_binds_its_own_vault(tmp_path):
    a = _by_name(drafts.as_tools(sqlite3.connect(":memory:"), tmp_path / "a"))
    b = _by_name(drafts.as_tools(sqlite3.connect(":memory:"), tmp_path / "b"))
    assert a["create_draft"] is not b["create_draft"]
