"""Dotenv loading.

This was planned in Task 1 but not implemented, because no test required it —
the config tests pass an explicit dict. The CLI would then have found no key
at all, so it is tested here directly.
"""

import os
from pathlib import Path

from ava import find_project_root, load_dotenv


def test_loads_key_value_pairs(tmp_path, monkeypatch):
    monkeypatch.delenv("AVA_TEST_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("AVA_TEST_KEY=abc123\n", encoding="utf-8")

    load_dotenv(env)

    assert os.environ["AVA_TEST_KEY"] == "abc123"


def test_ignores_comments_and_blanks(tmp_path, monkeypatch):
    monkeypatch.delenv("AVA_TEST_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("# a comment\n\nAVA_TEST_KEY=v\n", encoding="utf-8")

    load_dotenv(env)

    assert os.environ["AVA_TEST_KEY"] == "v"


def test_strips_surrounding_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("AVA_TEST_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("AVA_TEST_KEY='quoted value'\n", encoding="utf-8")

    load_dotenv(env)

    assert os.environ["AVA_TEST_KEY"] == "quoted value"


def test_real_env_wins_over_file(tmp_path, monkeypatch):
    """An exported variable must not be clobbered by the file."""
    monkeypatch.setenv("AVA_TEST_KEY", "from-shell")
    env = tmp_path / ".env"
    env.write_text("AVA_TEST_KEY=from-file\n", encoding="utf-8")

    load_dotenv(env)

    assert os.environ["AVA_TEST_KEY"] == "from-shell"


def test_missing_file_returns_none(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") is None


def test_value_may_contain_equals(tmp_path, monkeypatch):
    monkeypatch.delenv("AVA_TEST_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("AVA_TEST_KEY=a=b=c\n", encoding="utf-8")

    load_dotenv(env)

    assert os.environ["AVA_TEST_KEY"] == "a=b=c"


def test_find_project_root_locates_pyproject():
    root = find_project_root()
    assert (root / "pyproject.toml").exists()


def test_package_import_loads_the_project_env():
    """The whole point: importing ava must make the key visible."""
    import ava  # noqa: F401

    assert os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-")
