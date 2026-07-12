"""Live dev-credit smoke — exercise the REAL provider seam (L3) without touching Fireworks.

Proves, against owned dev credits + a local Ollama, that the cascade's two seams work end to
end: a real local call (free) and a real remote call (charged), with token usage actually read
back and the cost rule applied. This is the run that ticks BUILD_PLAN box 0e.

It spends a tiny amount of REMOTE credit (a couple of short calls); a hard $0.25 CostTracker
ceiling caps it so a misconfig can never run away. Local (Ollama) calls are free.

Usage (see the Makefile targets):
    uv run --extra serving python scripts/live_smoke.py                 # local + remote seams
    uv run --extra serving python scripts/live_smoke.py --full          # + a tiny end-to-end cascade
    uv run --extra serving python scripts/live_smoke.py --list-models   # list remote model ids (L6)

Env it reads (via config.DEV_CONFIG):
    OPENAI_API_KEY        — your owned remote key (or set DEV_REMOTE_PROVIDER + that provider's key)
    DEV_REMOTE_MODEL_ID   — a remote model id you have access to (use --list-models to find one)
    LOCAL_MODEL_ID        — the Ollama model (default qwen2.5:3b; `ollama pull` it first)
    OLLAMA_BASE_URL       — default http://localhost:11434/v1
"""

from __future__ import annotations

import argparse

from tokengolf.config import DEV_CONFIG
from tokengolf.schema import CostTracker

_REMOTE_CEILING_USD = 0.25  # hard kill-switch for this smoke — protects dev credits


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def _bad(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}")


def _local_cfg():
    return next((t for t in DEV_CONFIG.tiers if t.is_local), DEV_CONFIG.tiers[0])


def _remote_cfg():
    return next((t for t in DEV_CONFIG.tiers if not t.is_local), DEV_CONFIG.tiers[-1])


def list_models() -> int:
    """List the remote provider's model ids so you can pick/verify DEV_REMOTE_MODEL_ID (L6)."""
    import openai

    from tokengolf.providers import _resolve_api_key, _resolve_base_url

    cfg = _remote_cfg()
    client = openai.OpenAI(base_url=_resolve_base_url(cfg), api_key=_resolve_api_key(cfg))
    print(f"remote models for provider={cfg.provider}:")
    ids = sorted(m.id for m in client.models.list().data)
    for mid in ids:
        print(f"  {mid}")
    print(f"\n({len(ids)} models) — export one as DEV_REMOTE_MODEL_ID")
    return 0


def stage_local(max_tokens: int) -> bool:
    from tokengolf.providers import build_tier

    cfg = _local_cfg()
    print(f"\n[local seam] tier={cfg.name} provider={cfg.provider} model={cfg.model_id!r}")
    tracker = CostTracker()
    try:
        tier = build_tier(cfg, tracker, max_tokens=max_tokens, temperature=0.7)
        reply = tier.call("You are a terse assistant.", "Reply with exactly the word: ok")
    except Exception as e:  # noqa: BLE001 - surface a helpful hint, then fail the stage
        _bad(f"local call failed: {type(e).__name__}: {e}")
        print("    → is Ollama running? is the model pulled?  `ollama pull "
              f"{cfg.model_id}`  (or set LOCAL_MODEL_ID)")
        return False
    _ok(f"got text={reply.text!r}  in_tok={reply.in_tok} out_tok={reply.out_tok}")
    if reply.in_tok <= 0 or reply.out_tok <= 0:
        _bad("token usage not reported (in/out should be > 0)")
        return False
    if tracker.spent != 0.0:
        _bad(f"local tier charged ${tracker.spent} — local MUST be free (price 0)")
        return False
    _ok("local is free (spent $0.00) and reports real token usage")
    return True


def stage_remote(max_tokens: int) -> bool:
    from tokengolf.providers import build_tier

    cfg = _remote_cfg()
    print(f"\n[remote seam] tier={cfg.name} provider={cfg.provider} model={cfg.model_id!r}")
    if not cfg.model_id:
        _bad("DEV_REMOTE_MODEL_ID is empty — set it (use --list-models to find one)")
        return False
    tracker = CostTracker(ceiling_usd=_REMOTE_CEILING_USD)
    # reasoning models (gpt-5.x) need completion headroom + minimal effort or they return empty.
    effort = "none" if cfg.provider == "openai" else None  # no reasoning tokens → cheapest
    try:
        tier = build_tier(cfg, tracker, max_tokens=max(max_tokens, 256), temperature=0.0,
                          reasoning_effort=effort)
        reply = tier.call("You are a terse assistant.", "Reply with exactly the word: ok")
    except Exception as e:  # noqa: BLE001
        _bad(f"remote call failed: {type(e).__name__}: {e}")
        print("    → is OPENAI_API_KEY set and the model id valid?  try --list-models")
        return False
    _ok(f"got text={reply.text!r}  in_tok={reply.in_tok} out_tok={reply.out_tok}")
    if reply.in_tok <= 0 or reply.out_tok <= 0:
        _bad("token usage not reported (in/out should be > 0)")
        return False
    cost_note = f"${tracker.spent:.6f}" if (cfg.price_in or cfg.price_out) else \
        "$0 (set DEV_REMOTE_PRICE_IN/OUT to track $; the SCORED metric is tokens, which are real)"
    _ok(f"remote seam works; charged {cost_note} (under the ${_REMOTE_CEILING_USD} ceiling)")
    return True


def stage_cascade(max_tokens: int) -> bool:
    """Tiny end-to-end: route 2 math tasks. With no calibrator the local gate is uncalibrated, so
    it escalates (no silent promotion) — this exercises local self-consistency (free) + the remote
    fallback through the real cascade."""
    from tokengolf.eval import evaluate_cascade, format_report
    from tokengolf.run import build_confidence_fns, build_tiers
    from tokengolf.schema import Task

    print("\n[end-to-end cascade] 2 math tasks, no calibrator → escalate-all (preview)")
    tasks = [
        Task(id="q1", prompt="What is 17 + 26? Reply with just the number.", gold="43", kind="math"),
        Task(id="q2", prompt="What is 8 * 7? Reply with just the number.", gold="56", kind="math"),
    ]
    tracker = CostTracker(ceiling_usd=_REMOTE_CEILING_USD)
    try:
        tiers = build_tiers(DEV_CONFIG, tracker)
        fns = build_confidence_fns(DEV_CONFIG, tiers)
        report = evaluate_cascade(tasks, tiers, fns)
    except Exception as e:  # noqa: BLE001
        _bad(f"cascade failed: {type(e).__name__}: {e}")
        return False
    print(format_report(report))
    _ok(f"routed {report['n']} tasks; remote tokens={report['total_remote_tokens']} "
        f"spent=${tracker.spent:.6f}")
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Live dev-credit smoke for the provider seam (L3).")
    p.add_argument("--list-models", action="store_true", help="list remote model ids and exit (L6)")
    p.add_argument("--full", action="store_true", help="also run a tiny end-to-end cascade")
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args(argv)

    if args.list_models:
        return list_models()

    print("=== live dev-credit smoke (NOT Fireworks) ===")
    stages = [stage_local(args.max_tokens), stage_remote(args.max_tokens)]
    if args.full:
        stages.append(stage_cascade(args.max_tokens))

    print("\n=== summary ===")
    if all(stages):
        _ok("LIVE PATH EXERCISED — both seams work on dev credits (ticks BUILD_PLAN 0e)")
        return 0
    _bad("some stage failed — see hints above; 0e stays unticked until this is green")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
