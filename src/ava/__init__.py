"""AVA — a terminal personal assistant on the OpenAI Agents SDK.

Importing this package loads a project-local `.env` into os.environ, so the
key and model settings work without exporting them in every shell.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from this file to the directory holding pyproject.toml."""
    current = (start or Path(__file__).resolve().parent).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    # Installed non-editable: fall back to the package's parent's parent.
    return Path(__file__).resolve().parent.parent.parent


def load_dotenv(env_path: Path | None = None) -> Path | None:
    """Load KEY=VALUE pairs from .env into os.environ without overriding.

    Returns the path loaded, or None if there was no .env. Real environment
    variables win, so an exported key is never clobbered by the file.
    """
    path = env_path or (find_project_root() / ".env")
    if not path.exists():
        return None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    return path


load_dotenv()
