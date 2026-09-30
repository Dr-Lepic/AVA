"""OpenRouter wiring for the Agents SDK.

The real assertions here are about global SDK state, which pytest cannot roll
back on its own. configure_llm() mutates process-wide defaults, so every test
restores them afterwards.

State is read through the SDK's own getters in agents.models._openai_shared
(get_use_responses_by_default, get_default_openai_client) rather than by
poking module globals. Those getters are what the SDK's own set_* functions
delegate to, so the tests check the same thing the SDK checks.
"""

import pytest

import ava.llm as llm_mod
from ava.config import OPENROUTER_BASE_URL, Config
from ava.llm import configure_llm

# The SDK's own state module. Private by naming, but it is the only place the
# defaults live, and it exposes getters rather than requiring us to guess at
# variable names.
from agents.models import _openai_shared as shared


@pytest.fixture(autouse=True)
def restore_sdk_defaults():
    """Undo configure_llm's global mutations so tests stay independent."""
    saved_responses = shared.get_use_responses_by_default()
    saved_client = shared.get_default_openai_client()
    yield
    shared.set_use_responses_by_default(saved_responses)
    if saved_client is not None:
        shared.set_default_openai_client(saved_client)


@pytest.fixture
def config(tmp_path):
    return Config(api_key="sk-or-v1-test", model="openrouter/free", home=tmp_path)


# --- the settings that make the SDK work on OpenRouter ------------------


def test_configure_disables_responses_api(config):
    """The load-bearing setting. OpenRouter does not serve the Responses API."""
    configure_llm(config)
    assert shared.get_use_responses_by_default() is False


def test_configure_points_client_at_openrouter(config):
    configure_llm(config)
    client = shared.get_default_openai_client()
    assert client is not None
    assert str(client.base_url).rstrip("/") == OPENROUTER_BASE_URL


def test_configure_preserves_key(config):
    configure_llm(config)
    client = shared.get_default_openai_client()
    assert client is not None
    assert client.api_key == "sk-or-v1-test"


def test_configure_disables_tracing(config):
    """Traces upload to OpenAI; with no OpenAI key every run would 401.

    The provider derives its disabled flag from _manual_disabled plus the
    OPENAI_AGENTS_DISABLE_TRACING env var, so the manual flag is what
    configure_llm sets and what this asserts.
    """
    from agents.tracing import get_trace_provider

    configure_llm(config)
    provider = get_trace_provider()
    provider._refresh_disabled_flag()
    assert provider._disabled is True


def test_client_is_not_used_for_tracing(config):
    """use_for_tracing=False must not seed the tracing export key."""
    configure_llm(config)
    assert not shared.get_default_openai_key()


def test_configure_is_idempotent(config):
    """Called once at startup, but a second call must not raise or flip state."""
    configure_llm(config)
    configure_llm(config)
    assert shared.get_use_responses_by_default() is False


def test_configure_accepts_a_frozen_config(config):
    assert isinstance(config, Config)
    configure_llm(config)


# --- guard rails against the wrong provider ------------------------------


def test_module_docstring_records_why_chat_completions():
    """If someone 'upgrades' this to the Responses API, this is the note."""
    doc = (llm_mod.__doc__ or "").lower()
    assert "chat completions" in doc
    assert "openrouter" in doc


def test_responses_api_is_not_re_enabled(config):
    """Belt and braces: nothing should flip it back behind our back."""
    configure_llm(config)
    assert shared.get_use_responses_by_default() is not True
