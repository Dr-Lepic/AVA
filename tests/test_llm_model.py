"""OpenRouter slug handling.

Every OpenRouter model slug contains a '/' (e.g.
'nvidia/nemotron-3-ultra-550b-a55b:free'). The SDK's MultiProvider treats
anything before a '/' as a provider prefix and raises "Unknown prefix:
nvidia" for a prefix it does not recognise.

That only surfaced when a real key was first used, because every test until
then used ScriptedModel, which never resolves a slug. build_model() is the
fix, and these tests pin the shape of it.
"""

import pytest

from ava.config import Config
from ava.llm import build_model, configure_llm


@pytest.fixture
def configured(tmp_path):
    return Config(api_key="sk-or-v1-test", model="openrouter/free", home=tmp_path)


def test_build_model_wraps_a_prefixed_slug(configured):
    """The slug must survive verbatim; the SDK must not parse a provider out."""
    configure_llm(configured)
    model = build_model("nvidia/nemotron-3-ultra-550b-a55b:free")
    assert model.model == "nvidia/nemotron-3-ultra-550b-a55b:free"


def test_build_model_preserves_the_free_suffix(configured):
    configure_llm(configured)
    model = build_model("qwen/qwen3.8-27b:free")
    assert model.model.endswith(":free")


def test_build_model_reuses_the_configured_client(configured):
    """A second client would double the auth surface and skip the base_url."""
    from agents.models import _openai_shared as shared

    configure_llm(configured)
    model = build_model("openrouter/free")
    assert model._client is shared.get_default_openai_client()


def test_build_model_without_any_client_raises():
    """Fails loudly rather than silently building a client with no key.

    Clears both the module-level client and the SDK default, since a previous
    test in the session may have installed one — the fallback in build_model
    is correct behaviour, not a bug, so the test must remove both to reach the
    genuinely-unconfigured state.
    """
    import ava.llm as llm_mod
    from agents.models import _openai_shared as shared

    saved_client = llm_mod._client
    saved_default = shared.get_default_openai_client()
    llm_mod._client = None
    shared.set_default_openai_client(None)
    try:
        with pytest.raises(RuntimeError, match="configure_llm"):
            build_model("openrouter/free")
    finally:
        llm_mod._client = saved_client
        if saved_default is not None:
            shared.set_default_openai_client(saved_default)


def test_build_model_returns_a_chat_completions_model(configured):
    """Chat Completions specifically: OpenRouter does not serve Responses."""
    from agents import OpenAIChatCompletionsModel

    configure_llm(configured)
    assert isinstance(build_model("openrouter/free"), OpenAIChatCompletionsModel)


def test_no_prefix_parsing_happens(configured):
    """Regression: the SDK's prefix resolver would raise on a real slug.

    This is the assertion that would have caught the original bug offline.
    """
    from agents.models.multi_provider import MultiProvider

    configure_llm(configured)
    slug = "nvidia/nemotron-3-ultra-550b-a55b:free"

    # A bare slug handed to MultiProvider fails...
    with pytest.raises(Exception) as exc:
        MultiProvider().get_model(slug)
    assert "prefix" in str(exc.value).lower()

    # ...but build_model bypasses that path entirely.
    assert build_model(slug).model == slug
