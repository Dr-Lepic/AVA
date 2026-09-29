"""Environment configuration for AVA."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Used when AVA_MODEL is unset. A named free model with tool support.
# `openrouter/free` is the fallback if this one disappears from the free list.
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
FALLBACK_MODEL = "openrouter/free"

DEFAULT_TZ = "Asia/Dhaka"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def assert_free_model(model: str) -> str:
    """Reject any slug that is not obviously free.

    Two accepted forms:
      - anything ending in ':free'  (the always-free variant)
      - 'openrouter/free'           (router that only selects free models)

    Anything else raises, so a paid slug fails at startup instead of billing.
    """
    slug = model.strip()
    if slug.endswith(":free") or slug == FALLBACK_MODEL:
        return slug
    raise ConfigError(
        f"AVA_MODEL={slug!r} is not a free model. AVA is configured to never "
        f"spend money. Use a ':free' slug (see {OPENROUTER_MODELS_URL}) or "
        f"'{FALLBACK_MODEL}'."
    )


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str
    home: Path
    tz: str = DEFAULT_TZ

    @property
    def db_path(self) -> Path:
        return self.home / "ava.db"

    @property
    def vault_path(self) -> Path:
        return self.home / "vault"

    @property
    def state_path(self) -> Path:
        """Per-day request counter lives beside the database."""
        return self.home / "usage.json"


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Build a Config from environment variables.

    Accepts any mapping, so tests can pass a plain dict instead of mutating
    os.environ. Raises ConfigError with an actionable message when a required
    value is missing, so the CLI can print guidance instead of a traceback.
    """
    source: Mapping[str, str] = os.environ if env is None else env

    api_key = source.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            "OPENROUTER_API_KEY is not set. Create a key at "
            "https://openrouter.ai/keys then add it to your .env file."
        )

    model = assert_free_model(source.get("AVA_MODEL", "").strip() or DEFAULT_MODEL)
    tz = source.get("AVA_TZ", "").strip() or DEFAULT_TZ

    home = Path(source.get("AVA_HOME", "~/.ava")).expanduser()
    home.mkdir(parents=True, exist_ok=True)
    (home / "vault").mkdir(parents=True, exist_ok=True)

    return Config(api_key=api_key, model=model, home=home, tz=tz)
