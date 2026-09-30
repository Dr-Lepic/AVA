import email
import json
from email import policy

from ava.db import connect
from ava.tools import drafts


def test_create_draft_returns_not_sent(tmp_path):
    """The 'NOT SENT' string is what stops the model implying it sent mail."""
    conn = connect(tmp_path / "d.db")
    result = drafts.create_draft_impl(conn, "Lunch?", "Tacos at noon?", vault_path=tmp_path / "vault")
    assert "NOT SENT" in result


def test_create_draft_saves_row(tmp_path):
    conn = connect(tmp_path / "d.db")
    drafts.create_draft_impl(conn, "Lunch?", "Tacos?", to_addr="sam@example.com")
    rows = json.loads(drafts.list_drafts_impl(conn))
    assert len(rows) == 1
    assert rows[0]["to_addr"] == "sam@example.com"
    assert rows[0]["subject"] == "Lunch?"


def test_create_draft_writes_eml_file(tmp_path):
    vault = tmp_path / "vault"
    conn = connect(tmp_path / "d.db")
    drafts.create_draft_impl(conn, "Lunch?", "Tacos at noon?", vault_path=vault)

    files = list(vault.glob("*.eml"))
    assert len(files) == 1
    parsed = email.message_from_bytes(files[0].read_bytes(), policy=policy.default)
    assert parsed["Subject"] == "Lunch?"
    assert "Tacos at noon?" in parsed.get_content()


def test_eml_carries_to_header(tmp_path):
    vault = tmp_path / "vault"
    conn = connect(tmp_path / "d.db")
    drafts.create_draft_impl(conn, "Hi", "there", to_addr="sam@example.com", vault_path=vault)

    parsed = email.message_from_bytes(
        next(vault.glob("*.eml")).read_bytes(), policy=policy.default
    )
    assert parsed["To"] == "sam@example.com"


def test_eml_without_recipient_has_no_to_header(tmp_path):
    vault = tmp_path / "vault"
    conn = connect(tmp_path / "d.db")
    drafts.create_draft_impl(conn, "Note to self", "body", vault_path=vault)

    parsed = email.message_from_bytes(
        next(vault.glob("*.eml")).read_bytes(), policy=policy.default
    )
    assert parsed["To"] is None


def test_no_vault_path_still_saves_row(tmp_path):
    """Draft storage must not depend on the filesystem being available."""
    conn = connect(tmp_path / "d.db")
    result = drafts.create_draft_impl(conn, "S", "b")
    assert "NOT SENT" in result
    assert len(json.loads(drafts.list_drafts_impl(conn))) == 1


def test_vault_dir_is_created(tmp_path):
    vault = tmp_path / "nested" / "vault"
    conn = connect(tmp_path / "d.db")
    drafts.create_draft_impl(conn, "S", "b", vault_path=vault)
    assert vault.is_dir()


def test_list_drafts_empty_returns_message(tmp_path):
    conn = connect(tmp_path / "d.db")
    assert "No drafts" in drafts.list_drafts_impl(conn)


def test_list_drafts_newest_first(tmp_path):
    conn = connect(tmp_path / "d.db")
    for s in ("first", "second", "third"):
        drafts.create_draft_impl(conn, s, "b")
    rows = json.loads(drafts.list_drafts_impl(conn))
    assert [r["subject"] for r in rows] == ["third", "second", "first"]


def test_list_drafts_respects_limit(tmp_path):
    conn = connect(tmp_path / "d.db")
    for i in range(5):
        drafts.create_draft_impl(conn, f"s{i}", "b")
    assert len(json.loads(drafts.list_drafts_impl(conn, limit=2))) == 2


def test_filename_sanitizes_path_separators(tmp_path):
    vault = tmp_path / "vault"
    conn = connect(tmp_path / "d.db")
    drafts.create_draft_impl(conn, "../../etc/passwd", "b", vault_path=vault)

    files = list(vault.glob("*.eml"))
    assert len(files) == 1
    assert files[0].parent == vault


def test_filename_sanitizes_windows_illegal_chars(tmp_path):
    vault = tmp_path / "vault"
    conn = connect(tmp_path / "d.db")
    drafts.create_draft_impl(conn, 'a/b:c*d?e"f<g>h|i', "b", vault_path=vault)
    assert len(list(vault.glob("*.eml"))) == 1


def test_long_subject_is_truncated(tmp_path):
    vault = tmp_path / "vault"
    conn = connect(tmp_path / "d.db")
    drafts.create_draft_impl(conn, "x" * 300, "b", vault_path=vault)
    name = next(vault.glob("*.eml")).name
    # draft-001- + truncated subject + .eml
    assert len(name) < 80


def test_unicode_subject_does_not_crash(tmp_path):
    vault = tmp_path / "vault"
    conn = connect(tmp_path / "d.db")
    drafts.create_draft_impl(conn, "Café ☕ notes", "body", vault_path=vault)
    assert len(list(vault.glob("*.eml"))) == 1


def test_eml_body_roundtrips_multiline(tmp_path):
    vault = tmp_path / "vault"
    conn = connect(tmp_path / "d.db")
    body = "line one\n\nline two\n- bullet"
    drafts.create_draft_impl(conn, "S", body, vault_path=vault)

    parsed = email.message_from_bytes(
        next(vault.glob("*.eml")).read_bytes(), policy=policy.default
    )
    assert "line one" in parsed.get_content()
    assert "bullet" in parsed.get_content()
