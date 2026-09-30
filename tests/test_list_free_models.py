"""Tests for the free-model discovery script.

The script is a thin wrapper over the public models endpoint, but the filtering
is the useful part: a free model with no tool calling is useless to AVA, and
picking one by hand is how a run ends up silently unable to call a tool.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "list_free_models.py"


def _load():
    """Import the script by path — it is not part of the ava package."""
    spec = importlib.util.spec_from_file_location("list_free_models", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["list_free_models"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    if not SCRIPT.exists():
        pytest.skip("script not present")
    return _load()


# --- pricing filtering ---------------------------------------------------


def test_is_free_for_zero_pricing(mod):
    assert mod.is_free({"pricing": {"prompt": "0", "completion": "0"}}) is True


def test_is_not_free_for_paid_model(mod):
    model = {"pricing": {"prompt": "0.0000025", "completion": "0.00001"}}
    assert mod.is_free(model) is False


def test_is_not_free_when_pricing_missing(mod):
    assert mod.is_free({}) is False
    assert mod.is_free({"pricing": {}}) is False


def test_is_not_free_for_garbage_pricing(mod):
    """A malformed price must not be read as free."""
    assert mod.is_free({"pricing": {"prompt": "abc", "completion": "abc"}}) is False


def test_partially_free_is_not_free(mod):
    assert mod.is_free({"pricing": {"prompt": "0", "completion": "0.5"}}) is False


# --- tool support filtering ---------------------------------------------


def test_detects_tool_support(mod):
    assert mod.supports_tools({"supported_parameters": ["tools"]}) is True
    assert mod.supports_tools({"supported_parameters": ["tool_choice"]}) is True


def test_detects_no_tool_support(mod):
    assert mod.supports_tools({"supported_parameters": ["temperature"]}) is False
    assert mod.supports_tools({}) is False


# --- selection -----------------------------------------------------------


def test_usable_excludes_free_models_without_tools(mod):
    """The point of the script: AVA cannot use a tool-less free model."""
    models = [
        {"id": "a/free:free", "pricing": {"prompt": "0", "completion": "0"},
         "supported_parameters": ["tools"], "context_length": 100},
        {"id": "b/lyrics", "pricing": {"prompt": "0", "completion": "0"},
         "supported_parameters": ["temperature"], "context_length": 999},
    ]
    usable = mod.select_usable(models)
    assert [m["id"] for m in usable] == ["a/free:free"]


def test_free_router_is_always_usable(mod):
    """openrouter/free only ever selects free models, so it always qualifies."""
    models = [
        {"id": mod.FREE_ROUTER, "pricing": {"prompt": "0.01", "completion": "0.01"},
         "supported_parameters": []},
    ]
    assert len(mod.select_usable(models)) == 1


def test_paid_models_are_never_usable(mod):
    models = [
        {"id": "openai/gpt-4o", "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
         "supported_parameters": ["tools"]},
    ]
    assert mod.select_usable(models) == []
