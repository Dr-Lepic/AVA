"""OpenRouter wiring for the Agents SDK.

OpenRouter serves the Chat Completions API only. The SDK defaults to the
Responses API, so it must be switched off the Responses path or every call
fails with an endpoint error.

Tracing is disabled because the SDK uploads traces to OpenAI, and we hold an
OpenRouter key rather than a platform.openai.com one. Leaving it on produces a
401 on every run and, per OpenRouter's guidance, a failed attempt still costs
a request against the free-tier quota.

All three settings are process-wide. Call this once, before any Runner call.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from agents import (
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

from ava.config import OPENROUTER_BASE_URL, Config


def configure_llm(config: Config) -> None:
    """Point the SDK at OpenRouter. Idempotent; call once at startup."""
    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=config.api_key,
    )
    # use_for_tracing=False: the OpenRouter key is not valid for OpenAI
    # tracing uploads, and seeding it would turn every run into a 401.
    set_default_openai_client(client=client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)
