#!/usr/bin/env python3
"""List OpenRouter models that are free AND support tool calling.

Run:
    uv run python scripts/list_free_models.py
    uv run python scripts/list_free_models.py --all   # include tool-less models

Why this exists: OpenRouter's free list rotates, so a hardcoded AVA_MODEL
eventually points at a model that is no longer free. It also includes free
models with no tool-calling support, which AVA cannot use — its whole
interface is tools, so a tool-less model answers in prose and does nothing.

The endpoint is public, so no API key is needed.
"""

from __future__ import annotations

import json
import sys

# httpx2, not urllib: the stdlib path fails SSL verification in this
# environment (no CA bundle in the venv), while httpx2 ships its own and is
# already a dependency via openai v3. Same client the SDK itself uses.
import httpx2

MODELS_URL = "https://openrouter.ai/api/v1/models"
FREE_ROUTER = "openrouter/free"


def fetch_models() -> list[dict]:
    response = httpx2.get(MODELS_URL, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    return response.json()["data"]


def is_free(model: dict) -> bool:
    """True only if both prompt and completion pricing are exactly zero.

    Missing or malformed pricing must read as not-free. Defaulting the other
    way would let a malformed entry into a list used to pick a model.
    """
    pricing = model.get("pricing") or {}
    try:
        return (
            float(pricing.get("prompt", "1")) == 0
            and float(pricing.get("completion", "1")) == 0
        )
    except (TypeError, ValueError):
        return False


def supports_tools(model: dict) -> bool:
    params = model.get("supported_parameters") or []
    return "tools" in params or "tool_choice" in params


def select_usable(models: list[dict]) -> list[dict]:
    """Free models AVA can actually use: tool support, or the free router."""
    return [
        m
        for m in models
        if is_free(m) and (supports_tools(m) or m.get("id") == FREE_ROUTER)
    ]


def main() -> int:
    show_all = "--all" in sys.argv
    models = fetch_models()

    free = [m for m in models if is_free(m)]
    usable = select_usable(models)
    blocked = [m for m in free if m not in usable]

    print(
        f"{len(models)} models total, {len(free)} free, "
        f"{len(usable)} usable by AVA\n"
    )

    print("== USABLE (free + tool calling) ==")
    for m in sorted(usable, key=lambda x: -(x.get("context_length") or 0)):
        ctx = m.get("context_length") or 0
        print(f'  {m["id"]:<52} ctx={ctx:>9,}')

    if show_all and blocked:
        print("\n== FREE BUT NO TOOL CALLING (do not use) ==")
        for m in blocked:
            print(f'  {m["id"]:<52} ctx={m.get("context_length") or 0:>9,}')

    print(
        "\nSet one in .env:\n"
        f"  AVA_MODEL={usable[0]['id'] if usable else FREE_ROUTER}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
