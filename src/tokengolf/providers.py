"""Provider seam — turn a `TierConfig` into a callable `Tier`.

One OpenAI-compatible client serves BOTH the local model (Ollama @ localhost:11434/v1)
and every remote provider (Fireworks / OpenAI / OpenRouter / aimlapi / Featherless) via a
`base_url` swap; adding a provider is a `BASE_URLS` entry, not new code. Mirrors the proven
pattern in `experiments/path_b/providers.py`.

The `openai` SDK is imported LAZILY inside the functions so the core package imports without
the `serving` extra (tests inject a fake `CallFn` at the boundary — see `schema.py`).

Charging rule: realized USD = (in_tok * price_in + out_tok * price_out) / 1e6, billed to the
shared `CostTracker` per call. Local tiers have price 0.0 → they add 0.0 (free under scoring).
"""

from __future__ import annotations

import os
import time

from tokengolf.config import BASE_URLS, TierConfig
from tokengolf.schema import CostTracker, Reply, Tier

# Small exponential-backoff base (seconds); kept tiny so the default suite never sleeps on it.
_BACKOFF_BASE = 0.5
# HTTP statuses worth retrying when surfaced via a generic APIStatusError (L4).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _finalize(text: str, in_tok: int, out_tok: int, cfg: TierConfig, tracker: CostTracker) -> Reply:
    """Pure response→Reply step: charge the tracker the realized USD and return the Reply.

    Factored out for offline unit-testing (no network): local tiers (price 0.0) add 0.0; remote
    tiers add (it*price_in + ot*price_out)/1e6. `tracker.add` may raise BudgetExceeded at the ceiling.
    """
    tracker.add((in_tok * cfg.price_in + out_tok * cfg.price_out) / 1e6)
    return Reply(text=text or "", in_tok=in_tok, out_tok=out_tok)


def _resolve_base_url(cfg: TierConfig) -> str:
    """BASE_URLS[provider], overridable via os.environ[cfg.base_url_env] when that env is set."""
    if cfg.base_url_env and os.environ.get(cfg.base_url_env):
        return os.environ[cfg.base_url_env]
    try:
        return BASE_URLS[cfg.provider]
    except KeyError as e:  # pragma: no cover - guards a config typo
        raise ValueError(
            f"unknown provider {cfg.provider!r}; known: {sorted(BASE_URLS)}"
        ) from e


def _resolve_api_key(cfg: TierConfig) -> str:
    """API key from os.environ[cfg.env_key]; Ollama ignores the key, so a dummy is fine."""
    if cfg.is_local or cfg.provider == "ollama":
        # Ollama's OpenAI-compatible server ignores the key but the client requires a non-empty one.
        return os.environ.get(cfg.env_key, "ollama") if cfg.env_key else "ollama"
    if not cfg.env_key:
        raise ValueError(f"remote tier {cfg.name!r} needs an env_key for its API key")
    key = os.environ.get(cfg.env_key)
    if not key:
        raise ValueError(f"missing API key: set ${cfg.env_key} for tier {cfg.name!r}")
    return key


def build_tier(
    cfg: TierConfig,
    tracker: CostTracker,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 3,
    reasoning_effort: str | None = None,
    prompt_cache_key: str | None = None,
) -> Tier:
    """Turn a `TierConfig` into a live `Tier` over an OpenAI-compatible client.

    The returned `Tier.call(system, user) -> Reply` makes a bounded-retry chat completion, reads
    usage, charges the tracker, and returns the Reply. `temperature` is passed through (the LOCAL
    tier is built with temperature>0 for self-consistency; remotes default 0.0). If a provider
    rejects a non-default temperature (some OpenAI reasoning models do), it is dropped once and
    the call retried without it.

    `reasoning_effort` is sent for the OpenAI and Fireworks providers (both accept it). Reasoning
    models otherwise spend the token budget on hidden reasoning and can return empty text / overrun
    cost (the L-cost-01 lesson). On Fireworks, "none" collapses reasoning output ~20-34x on
    minimax-m3 / kimi-k2p7-code (measured) while keeping the answer; "low" is ignored by them. An
    unsupported value is dropped once and the call retried. Left None for non-reasoning providers.
    """
    import openai  # lazy: keep the core importable without the `serving` extra

    base_url = _resolve_base_url(cfg)
    api_key = _resolve_api_key(cfg)
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    # Native OpenAI reasoning models require `max_completion_tokens`; OpenAI-compatible providers
    # (Ollama / Fireworks / OpenRouter / aimlapi / Featherless) take `max_tokens`.
    tok_kw = (
        {"max_completion_tokens": max_tokens}
        if cfg.provider == "openai"
        else {"max_tokens": max_tokens}
    )

    # Transient exception classes for THIS SDK — enumerated, not assumed from one hierarchy (L4).
    transient = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )

    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, transient):
            return True
        if isinstance(exc, openai.APIStatusError):
            return getattr(exc, "status_code", None) in _RETRYABLE_STATUS
        return False

    def _request(messages: list[dict], temp: float | None, effort: str | None):
        kwargs: dict = dict(model=cfg.model_id, messages=messages, **tok_kw)
        if temp is not None:
            kwargs["temperature"] = temp
        if effort is not None and cfg.provider in ("openai", "fireworks"):
            # Fireworks + OpenAI both accept `reasoning_effort`. For the Fireworks reasoning models
            # (minimax-m3, kimi-k2p7-code) "none" collapses hidden-reasoning output tokens ~20-34x
            # (measured) while preserving the answer — the dominant leaderboard token lever. An
            # unsupported value is dropped once via the BadRequestError handler below.
            kwargs["reasoning_effort"] = effort
        if prompt_cache_key is not None and cfg.provider == "openai":
            # OpenAI prompt caching is automatic for prefixes ≥1024 tok (90% off cached input);
            # this key just improves routing/hit-rate. Harmless when the prefix is short.
            kwargs["prompt_cache_key"] = prompt_cache_key
        return client.chat.completions.create(**kwargs)

    def _call(system: str, user: str) -> Reply:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        temp: float | None = temperature
        effort: str | None = reasoning_effort
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                r = _request(messages, temp, effort)
                it = r.usage.prompt_tokens
                ot = r.usage.completion_tokens
                text = r.choices[0].message.content or ""
                return _finalize(text, it, ot, cfg, tracker)
            except openai.BadRequestError as e:
                # Models vary in which knobs they accept; drop an unsupported one once and retry.
                msg = str(e).lower()
                if temp is not None and "temperature" in msg:
                    temp = None
                    continue
                if effort is not None and "reasoning_effort" in msg:
                    effort = None
                    continue
                raise  # other 400s are not transient — propagate
            except Exception as e:  # noqa: BLE001 - we re-raise non-retryable below
                if not _is_retryable(e) or attempt == max_retries:
                    raise
                last_exc = e
                time.sleep(_BACKOFF_BASE * (2**attempt))
        # Loop only exits via return/raise; this guards against a logic slip.
        raise last_exc if last_exc else RuntimeError("retry loop exhausted")  # pragma: no cover

    return Tier(
        name=cfg.name,
        call=_call,
        price_in=cfg.price_in,
        price_out=cfg.price_out,
        is_local=cfg.is_local,
        threshold=cfg.threshold,
    )


def live_smoke(cfg: TierConfig, *, max_tokens: int = 64) -> Reply:
    """One real call against `cfg` IF its endpoint/key is reachable. For `__main__` / live tests.

    Charges a throwaway tracker (no ceiling). Raises whatever the SDK raises if unreachable.
    """
    tier = build_tier(cfg, CostTracker(), max_tokens=max_tokens)
    return tier.call("You are a terse assistant.", "Reply with exactly the word: ok")


if __name__ == "__main__":  # pragma: no cover - manual live probe
    import sys

    from tokengolf.config import DEV_CONFIG

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cfg = DEV_CONFIG.tiers[idx]
    print(f"live_smoke → tier={cfg.name} provider={cfg.provider} model={cfg.model_id!r}")
    reply = live_smoke(cfg)
    print(f"  text={reply.text!r}  in_tok={reply.in_tok}  out_tok={reply.out_tok}")
