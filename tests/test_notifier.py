"""macOS notification delivery.

The security-relevant test is test_quote_cannot_break_out_of_the_literal:
model-produced text (a reminder title) is interpolated into an AppleScript
string, so an unescaped double quote would let a crafted title run arbitrary
commands. The escaping tests pin that closed.
"""

import platform
import subprocess

import pytest

from ava import notifier


class FakeRun:
    """Records the argv instead of executing it."""

    calls = []

    def __init__(self, args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        FakeRun.calls.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_run(monkeypatch):
    FakeRun.calls = []
    monkeypatch.setattr(notifier.subprocess, "run", FakeRun)
    return FakeRun


@pytest.fixture
def on_macos(monkeypatch):
    monkeypatch.setattr(notifier.platform, "system", lambda: "Darwin")


def _script(fake_run):
    return fake_run.calls[-1].args[-1]


# --- escaping ------------------------------------------------------------


def test_escape_leaves_plain_text_alone():
    assert notifier._escape("standup") == "standup"


def test_escape_quotes():
    assert notifier._escape('say "hi"') == 'say \\"hi\\"'


def test_escape_backslash():
    assert notifier._escape("a\\b") == "a\\\\b"


def test_escape_backslash_before_quote():
    """Backslash must be escaped first, or it double-escapes the quote."""
    assert notifier._escape('\\"') == '\\\\\\"'


def test_escape_neutralises_script_injection():
    """The payload a title like this could carry must not survive as code."""
    payload = '"; do shell script "rm -rf ~"; --'
    escaped = notifier._escape(payload)
    # Every quote is escaped, so the literal cannot be terminated early.
    assert '"' not in escaped.replace('\\"', "")
    assert escaped.count('"') == escaped.count('\\"')


def test_escape_backtick_and_newline_are_harmless():
    """Backticks and newlines are not special inside an AppleScript literal."""
    assert notifier._escape("a`b") == "a`b"
    assert notifier._escape("a\nb") == "a\nb"


# --- platform gate -------------------------------------------------------


def test_notify_returns_false_on_linux(monkeypatch, fake_run):
    monkeypatch.setattr(notifier.platform, "system", lambda: "Linux")
    assert notifier.notify("t", "m") is False
    assert fake_run.calls == []


def test_notify_returns_false_on_windows(monkeypatch, fake_run):
    monkeypatch.setattr(notifier.platform, "system", lambda: "Windows")
    assert notifier.notify("t", "m") is False
    assert fake_run.calls == []


# --- happy path ----------------------------------------------------------


def test_notify_invokes_osascript(on_macos, fake_run):
    assert notifier.notify("AVA", "Standup in 5 min") is True
    assert fake_run.calls[0].args[0] == "osascript"
    assert "display notification" in _script(fake_run)


def test_notify_includes_title_and_message(on_macos, fake_run):
    notifier.notify("AVA", "Standup in 5 min")
    script = _script(fake_run)
    assert "AVA" in script
    assert "Standup in 5 min" in script


def test_notify_passes_script_via_dash_e(on_macos, fake_run):
    """The script must be an argv element, never interpolated into a shell string."""
    notifier.notify("AVA", "body")
    assert fake_run.calls[0].args[1] == "-e"


def test_notify_uses_check_true(on_macos, fake_run):
    notifier.notify("AVA", "body")
    assert fake_run.calls[0].kwargs["check"] is True


def test_notify_has_a_timeout(on_macos, fake_run):
    """A hung osascript must not wedge the reminder watcher forever."""
    notifier.notify("AVA", "body")
    assert fake_run.calls[0].kwargs["timeout"] == 5


def test_notify_escapes_the_message_on_the_way_through(on_macos, fake_run):
    notifier.notify("AVA", 'he said "hi"')
    assert 'he said \\"hi\\"' in _script(fake_run)


def test_notify_escapes_the_title_on_the_way_through(on_macos, fake_run):
    notifier.notify('Bad "Title"', "body")
    assert 'Bad \\"Title\\"' in _script(fake_run)


# --- failure handling ----------------------------------------------------


def test_notify_returns_false_when_osascript_missing(on_macos, monkeypatch):
    def raise_missing(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(notifier.subprocess, "run", raise_missing)
    assert notifier.notify("t", "m") is False


def test_notify_returns_false_on_timeout(on_macos, monkeypatch):
    def raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="osascript", timeout=5)

    monkeypatch.setattr(notifier.subprocess, "run", raise_timeout)
    assert notifier.notify("t", "m") is False


def test_notify_returns_false_on_nonzero_exit(on_macos, monkeypatch):
    def raise_called(*a, **k):
        raise subprocess.CalledProcessError(returncode=1, cmd="osascript")

    monkeypatch.setattr(notifier.subprocess, "run", raise_called)
    assert notifier.notify("t", "m") is False


def test_notify_never_raises(on_macos, monkeypatch):
    """A notification failure must not kill the watcher loop."""
    def explode(*a, **k):
        raise OSError("something unexpected")

    monkeypatch.setattr(notifier.subprocess, "run", explode)
    assert notifier.notify("t", "m") is False
