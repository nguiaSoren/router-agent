"""ALLOWED_MODELS id normalization — the harness may inject bare slugs; the Fireworks API needs the
full `accounts/fireworks/models/<slug>` (a bare slug 404s → every call fails). Regression guard."""

from tokengolf import config


def test_normalize_bare_slug_gets_prefix():
    assert config._normalize_fireworks_id("kimi-k2p7-code") == "accounts/fireworks/models/kimi-k2p7-code"
    assert config._normalize_fireworks_id("minimax-m3") == "accounts/fireworks/models/minimax-m3"


def test_normalize_full_id_passes_through():
    full = "accounts/fireworks/models/kimi-k2p7-code"
    assert config._normalize_fireworks_id(full) == full


def test_allowed_models_from_env_normalizes(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", " minimax-m3 , accounts/fireworks/models/kimi-k2p7-code ")
    assert config._allowed_models_from_env() == [
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/kimi-k2p7-code",
    ]


def test_scoring_config_uses_full_ids(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "minimax-m3,kimi-k2p7-code")
    cfg = config.scoring_config_from_env()
    assert [t.model_id for t in cfg.tiers] == [
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/kimi-k2p7-code",
    ]
