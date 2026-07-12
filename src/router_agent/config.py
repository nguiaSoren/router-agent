"""Cascade configuration — model tiers as DATA, not code.

Model ids, base_urls, and the operating thresholds are config slots so launch day
(July 7, when the real local + Fireworks models are revealed) is a config edit, not
a code change. Nothing here is hard-coded from memory — ids come from env or are
filled at kickoff after verifying against each provider's /models (L6/L8).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TierConfig:
    """Declarative spec for one cascade tier. `providers.build_tier` turns this into a Tier."""
    name: str
    provider: str           # "ollama" (local) | "fireworks" | "openai" | "openrouter" | "aimlapi" | "featherless"
    model_id: str           # the provider's model id (env-overridable)
    is_local: bool
    price_in: float = 0.0   # $/1M input tok  (0.0 for local)
    price_out: float = 0.0  # $/1M output tok (0.0 for local)
    threshold: float = 1.0  # accept iff calibrated confidence >= threshold (last tier always accepts)
    env_key: str | None = None   # env var holding the API key (None for local)
    base_url_env: str | None = None  # optional env override for base_url


# Provider base_urls (verified live 2026-06-30 — re-verify before relying, L6).
BASE_URLS: dict[str, str] = {
    "ollama":      os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    "fireworks":   "https://api.fireworks.ai/inference/v1",
    "openai":      "https://api.openai.com/v1",
    "openrouter":  "https://openrouter.ai/api/v1",
    "aimlapi":     "https://api.aimlapi.com/v1",
    "featherless": "https://api.featherless.ai/v1",
}


@dataclass
class CascadeConfig:
    """The full cascade: cost-ordered tiers + the confidence signal's knobs."""
    tiers: list[TierConfig]
    self_consistency_n: int = 5         # local samples drawn per task (free)
    self_consistency_temp: float = 0.7  # sampling temperature for self-consistency
    accuracy_floor: float = 0.0         # the hidden Track-1 threshold (set after kickoff)
    accuracy_margin: float = 0.03       # clear the floor by this much (hidden test set => don't sail at it)
    budget_ceiling_usd: float | None = None  # CostTracker kill-switch
    local_max_tokens: int = 1024        # local needs CoT headroom (256 truncates GSM8K mid-reasoning)
    remote_max_tokens: int = 512        # remote is the fallback; keep bounded (cost)


# ---- DEV config: Mac + Ollama local, a cheap owned remote for plumbing (env-driven ids). ----
# Burns dev credits, NOT Fireworks. Model ids via env so nothing is guessed from memory.
DEV_CONFIG = CascadeConfig(
    tiers=[
        TierConfig(
            name="local",
            provider="ollama",
            model_id=os.environ.get("LOCAL_MODEL_ID", "qwen2.5:3b"),  # verify available via `ollama list`
            is_local=True,
            threshold=0.80,  # placeholder until calibrated on launch-day labels (no silent promotion)
        ),
        TierConfig(
            name="remote",
            provider=os.environ.get("DEV_REMOTE_PROVIDER", "openai"),
            # gpt-5.4-nano: cheapest model this key can call ($0.20/$1.25 per 1M; verified live
            # via models.list() + the pricing page, 2026-06-30, L6). Override with DEV_REMOTE_MODEL_ID.
            model_id=os.environ.get("DEV_REMOTE_MODEL_ID", "gpt-5.4-nano"),
            is_local=False,
            price_in=float(os.environ.get("DEV_REMOTE_PRICE_IN", "0.20")),
            price_out=float(os.environ.get("DEV_REMOTE_PRICE_OUT", "1.25")),
            threshold=1.0,   # last tier: always accepted
            env_key="AMD_OPENAI_API_KEY",   # the OpenAI platform key (~$42 balance) lives here
        ),
    ],
    budget_ceiling_usd=float(os.environ.get("DEV_BUDGET_CEILING", "5.0")),
)


# ---- SCORING config: built from the harness-injected env at eval time (Track 1 contract). ----
def _normalize_fireworks_id(mid: str) -> str:
    """Fireworks needs the full `accounts/fireworks/models/<slug>`; the harness may inject a bare slug
    (the launch notes list the models as `minimax-m3` / `kimi-k2p7-code`). A bare slug 404s
    (`model_not_found`) → every call fails → zero answers. Prepend the prefix when it's missing so the
    id is always callable, whether the harness passes a slug or a full path (full paths pass through)."""
    mid = mid.strip()
    return mid if "/" in mid else f"accounts/fireworks/models/{mid}"


def _allowed_models_from_env() -> list[str]:
    """Parse ALLOWED_MODELS (comma-separated), normalized to full Fireworks ids."""
    return [_normalize_fireworks_id(m) for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]


def scoring_config_from_env() -> CascadeConfig:
    """Build the submission cascade from the harness env vars (Participant Guide, Track 1).

    Reads ALLOWED_MODELS (comma-separated Fireworks model ids, published launch day) and routes
    EVERY call through FIREWORKS_BASE_URL with FIREWORKS_API_KEY — calls that bypass the proxy
    score zero tokens. One tier per allowed model, in listed order (treated cheap→strong until we
    know the pricing). Prices are 0.0 here on purpose: the leaderboard counts *tokens via the
    proxy*, not our $ estimate — the CostTracker $ is advisory in submission mode.

    Tunable via env (so we can sweep without rebuilding): ROUTER_SC_N (self-consistency samples;
    default 1 — on Fireworks each sample COSTS tokens, unlike a free local model), ROUTER_TAU
    (escalation threshold for non-final tiers), ROUTER_FLOOR (accuracy floor).
    """
    allowed = _allowed_models_from_env()
    if not allowed:
        raise ValueError(
            "ALLOWED_MODELS is empty — the harness injects it at eval time; set it "
            "(comma-separated Fireworks model ids) to test the submission path locally"
        )
    tau = float(os.environ.get("ROUTER_TAU", "0.75"))
    tiers = [
        TierConfig(
            name=f"fw{i}:{mid.split('/')[-1]}",
            provider="fireworks",
            model_id=mid,
            is_local=False,
            price_in=0.0,
            price_out=0.0,
            threshold=1.0 if i == len(allowed) - 1 else tau,  # last tier always accepts
            env_key="FIREWORKS_API_KEY",
            base_url_env="FIREWORKS_BASE_URL",  # the token-counting proxy — MUST route through it
        )
        for i, mid in enumerate(allowed)
    ]
    return CascadeConfig(
        tiers=tiers,
        self_consistency_n=int(os.environ.get("ROUTER_SC_N", "1")),
        accuracy_floor=float(os.environ.get("ROUTER_FLOOR", "0.0")),
        remote_max_tokens=int(os.environ.get("ROUTER_MAX_TOKENS", "512")),
        budget_ceiling_usd=None,  # tokens are tracked by the proxy, not our $ ceiling
    )


def submission_config_from_env() -> CascadeConfig:
    """The TOKEN-OPTIMAL submission cascade: a FREE local GGUF tier → ONE Fireworks tier.

    Binary, not N-tier: the leaderboard counts TOKENS, and escalating among Fireworks models only
    ADDS tokens — so the only token-free lever is the local model. Self-consistency runs on the free
    local tier (CPU); the single Fireworks call handles escalations. Picks `minimax-m3` from
    ALLOWED_MODELS when present (capable, concise, general), else the first allowed model.
    """
    allowed = _allowed_models_from_env()
    if not allowed:
        raise ValueError(
            "ALLOWED_MODELS is empty — set it (comma-separated Fireworks ids) to test locally; "
            "the harness injects it at eval time"
        )
    remote_id = next((m for m in allowed if "minimax-m3" in m), allowed[0])
    tau = float(os.environ.get("ROUTER_TAU", "0.75"))
    tiers = [
        TierConfig(
            name="local",
            provider="local_gguf",
            model_id=os.environ.get("LOCAL_GGUF_FILE", "qwen2.5-3b-instruct-q4_k_m.gguf"),
            is_local=True,
            threshold=tau,  # keep local when calibrated confidence >= tau, else escalate
        ),
        TierConfig(
            name=f"fw:{remote_id.split('/')[-1]}",
            provider="fireworks",
            model_id=remote_id,
            is_local=False,
            threshold=1.0,  # last tier always accepts
            env_key="FIREWORKS_API_KEY",
            base_url_env="FIREWORKS_BASE_URL",
        ),
    ]
    return CascadeConfig(
        tiers=tiers,
        self_consistency_n=int(os.environ.get("ROUTER_SC_N", "2")),  # local samples (free, but CPU — N=2 for the 30s/req margin)
        self_consistency_temp=float(os.environ.get("ROUTER_SC_TEMP", "0.7")),
        accuracy_floor=float(os.environ.get("ROUTER_FLOOR", "0.0")),
        local_max_tokens=int(os.environ.get("LOCAL_MAX_TOKENS", "400")),  # CoT needs ~400; caps rambling → faster/task
        remote_max_tokens=int(os.environ.get("ROUTER_MAX_TOKENS", "512")),
        budget_ceiling_usd=None,
    )


SCORING_CONFIG: CascadeConfig | None = None  # populated by scoring_/submission_config_from_env() at runtime
