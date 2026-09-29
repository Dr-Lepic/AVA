import pytest

from ava.config import (
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    OPENROUTER_BASE_URL,
    ConfigError,
    assert_free_model,
    load_config,
)

PAID_SLUGS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4.1",
]


def test_load_config_reads_api_key(tmp_path):
    cfg = load_config({"OPENROUTER_API_KEY": "sk-or-v1-test", "AVA_HOME": str(tmp_path)})
    assert cfg.api_key == "sk-or-v1-test"
    assert cfg.model == DEFAULT_MODEL
    assert cfg.db_path.name == "ava.db"
    assert cfg.vault_path.name == "vault"


def test_load_config_defaults_tz(tmp_path):
    cfg = load_config({"OPENROUTER_API_KEY": "k", "AVA_HOME": str(tmp_path)})
    assert cfg.tz == "Asia/Dhaka"


def test_load_config_reads_tz_override(tmp_path):
    cfg = load_config(
        {"OPENROUTER_API_KEY": "k", "AVA_HOME": str(tmp_path), "AVA_TZ": "UTC"}
    )
    assert cfg.tz == "UTC"


def test_load_config_raises_without_api_key(tmp_path):
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        load_config({"AVA_HOME": str(tmp_path)})


@pytest.mark.parametrize("slug", PAID_SLUGS)
def test_assert_free_model_rejects_paid_slugs(slug):
    """The whole point: a paid model must fail fast, not silently bill."""
    with pytest.raises(ConfigError, match="not a free model"):
        assert_free_model(slug)


def test_assert_free_model_accepts_free_suffix():
    assert assert_free_model("nvidia/nemotron-3-super-120b-a12b:free").endswith(":free")


def test_assert_free_model_accepts_router():
    assert assert_free_model(FALLBACK_MODEL) == "openrouter/free"


def test_default_model_is_free():
    assert DEFAULT_MODEL.endswith(":free")


def test_load_config_rejects_paid_model_in_env(tmp_path):
    with pytest.raises(ConfigError, match="not a free model"):
        load_config(
            {
                "OPENROUTER_API_KEY": "k",
                "AVA_HOME": str(tmp_path),
                "AVA_MODEL": "openai/gpt-4o-mini",
            }
        )


def test_state_path_is_next_to_db(tmp_path):
    cfg = load_config({"OPENROUTER_API_KEY": "k", "AVA_HOME": str(tmp_path)})
    assert cfg.state_path.parent == cfg.home
    assert cfg.state_path.name == "usage.json"


def test_base_url_is_openrouter():
    assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


def test_load_config_creates_home_and_vault(tmp_path):
    home = tmp_path / "nested" / "ava-home"
    load_config({"OPENROUTER_API_KEY": "k", "AVA_HOME": str(home)})
    assert home.is_dir()
    assert (home / "vault").is_dir()
