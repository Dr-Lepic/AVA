"""Guards the packaging itself, not application behaviour.

These exist because `pythonpath = ["src"]` in pyproject.toml only applies to
pytest. Without the editable install configured via [build-system], the package
imports fine under `uv run pytest` and fails in every other script — a failure
mode the unit tests cannot see.
"""

import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )


def test_ava_importable_outside_pytest():
    """`import ava` must work from a plain interpreter, not only under pytest."""
    result = _run("import ava.config; print(ava.config.DEFAULT_MODEL)")
    assert result.returncode == 0, result.stderr
    assert ":free" in result.stdout


def test_free_model_guard_works_outside_pytest():
    """The billing guard must hold in the real CLI, not just in unit tests."""
    code = (
        "from ava.config import load_config, ConfigError\n"
        "try:\n"
        "    load_config({'OPENROUTER_API_KEY':'k','AVA_HOME':'/tmp/ava-pkg-1',"
        "'AVA_MODEL':'openai/gpt-4o-mini'})\n"
        "    print('ACCEPTED')\n"
        "except ConfigError:\n"
        "    print('REFUSED')\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "REFUSED"
