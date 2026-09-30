"""macOS desktop notifications via osascript.

The only platform-bound module in AVA. Everything else stays portable, so a
non-macOS run degrades to a printed message rather than failing.

Security note: title and message are model-produced text — a reminder title
the user typed, relayed through a free model — and are interpolated into an
AppleScript string literal. They are escaped, and the script is passed as an
argv element rather than interpolated into a shell string, so a crafted title
cannot run commands.
"""

from __future__ import annotations

import platform
import subprocess

TIMEOUT_SECONDS = 5


def _escape(text: str) -> str:
    """Escape backslashes and double quotes for an AppleScript string literal.

    Backslashes must be escaped before quotes, or an escaped quote would be
    escaped a second time and the literal could be broken out of.
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')


def notify(title: str, message: str) -> bool:
    """Show a macOS notification. Returns True if it was delivered.

    Fails soft by design: returns False on non-macOS, a missing osascript, a
    timeout, or any other error. The reminder watcher calls this in a poll
    loop, so a failure here must never propagate.
    """
    if platform.system() != "Darwin":
        return False

    script = (
        f'display notification "{_escape(message)}" '
        f'with title "{_escape(title)}"'
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return True
