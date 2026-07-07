"""Batch-API smoke — exercise the OpenAI Batch path (batch.py) END TO END on a few dev tasks.

Proves the −50% batched-labeling lever works live: load ~8 dev tasks (GSM8K + short-QA), pack
them into one batch job at HALF price, submit → poll → collect, grade each reply, then report
the REALIZED batch cost against the ESTIMATED synchronous cost (= 2× batch, since batch is −50%).

The orchestrator runs this live (it needs a real key). The assembly is factored into the pure
`build_requests` so it is offline-testable without touching the network (see tests/test_batch_smoke.py).

Usage (live — needs AMD_OPENAI_API_KEY in the environment):
    set -a; . ../.env; set +a; uv run --extra serving --extra data python scripts/batch_smoke.py

Env it reads (via config.DEV_CONFIG remote tier):
    AMD_OPENAI_API_KEY    — the OpenAI platform key (the remote tier's env_key)
    DEV_REMOTE_MODEL_ID   — override the remote model id (default gpt-5.4-nano)
"""

from __future__ import annotations

import os

from router_agent.batch import BatchRequest, submit_and_wait
from router_agent.config import BASE_URLS, DEV_CONFIG
from router_agent.schema import CostTracker, Task

# Batch is half the synchronous price (gpt-5.4-nano std 0.20/1.25 → batch 0.10/0.625).
_BATCH_PRICE_IN = 0.10
_BATCH_PRICE_OUT = 0.625
_CEILING_USD = 0.25        # hard kill-switch for this smoke — protects dev credits
_SYSTEM = "You are a terse assistant. Answer concisely."


def _remote_cfg():
    """The non-local (remote) dev tier — provider=openai, env_key=AMD_OPENAI_API_KEY."""
    return next((t for t in DEV_CONFIG.tiers if not t.is_local), DEV_CONFIG.tiers[-1])


def build_requests(tasks: list[Task]) -> list[BatchRequest]:
    """Pure assembly: one BatchRequest per task (custom_id=task.id, shared terse system prompt).

    Offline-testable — no network, no env. The custom_id IS the task id so the unordered batch
    output can be re-keyed back onto each task for grading.
    """
    return [
        BatchRequest(custom_id=t.id, system=_SYSTEM, user=t.prompt)
        for t in tasks
    ]


def load_tasks() -> list[Task]:
    """Load ~8 dev tasks (4 GSM8K math + 4 TriviaQA short-QA). Lazy via tasks.py (datasets)."""
    from router_agent.tasks import load_gsm8k, load_short_qa

    return load_gsm8k(4) + load_short_qa(4)


def main() -> int:
    from router_agent.tasks import check

    cfg = _remote_cfg()
    env_key = cfg.env_key or "AMD_OPENAI_API_KEY"
    api_key = os.environ.get(env_key)
    if not api_key:
        print(f"✗ {env_key} is not set — export it (e.g. `set -a; . ../.env; set +a`) and re-run.")
        return 1

    base_url = BASE_URLS[cfg.provider]
    model_id = cfg.model_id
    if not model_id:
        print("✗ remote model_id is empty — set DEV_REMOTE_MODEL_ID.")
        return 1

    print("=== batch-API smoke (−50% labeling lever) ===")
    print(f"provider={cfg.provider} model={model_id!r} base_url={base_url}")
    print(f"batch price ${_BATCH_PRICE_IN}/${_BATCH_PRICE_OUT} per 1M (½ of sync $0.20/$1.25)")

    tasks = load_tasks()
    reqs = build_requests(tasks)
    by_id = {t.id: t for t in tasks}
    print(f"\nloaded {len(tasks)} dev tasks → {len(reqs)} batch requests; submitting...")

    tracker = CostTracker(ceiling_usd=_CEILING_USD)
    try:
        replies = submit_and_wait(
            reqs,
            model_id,
            api_key=api_key,
            base_url=base_url,
            price_in=_BATCH_PRICE_IN,
            price_out=_BATCH_PRICE_OUT,
            tracker=tracker,
            max_tokens=512,
            reasoning_effort="low",
            poll_interval_s=15.0,
            timeout_s=1800,
        )
    except TimeoutError as e:
        print(f"\n⏳ batch still running — re-run collect later. {e}")
        print("    (the batch was submitted; use batch.collect(batch_id, ...) once it completes)")
        return 1
    except Exception as e:  # noqa: BLE001 - surface a helpful hint, then fail
        print(f"\n✗ batch failed: {type(e).__name__}: {e}")
        return 1

    # Grade each returned reply against its task's gold; print per-task.
    print(f"\n--- per-task results ({len(replies)} returned of {len(tasks)}) ---")
    n_correct = 0
    for tid, task in by_id.items():
        rep = replies.get(tid)
        if rep is None:
            print(f"  {tid:<22} kind={task.kind:<5} MISSING (no reply returned)")
            continue
        correct = check(rep.text, task)
        n_correct += correct
        print(f"  {tid:<22} kind={task.kind:<5} correct={str(correct):<5} out_tok={rep.out_tok}")

    n_ret = len(replies)
    acc = (n_correct / n_ret) if n_ret else 0.0
    realized = tracker.spent
    est_sync = realized * 2.0  # batch is −50%, so sync would have cost double

    print("\n=== summary ===")
    print(f"  returned       : {n_ret} / {len(tasks)}")
    print(f"  accuracy       : {acc:.1%}  ({n_correct}/{n_ret})")
    print(f"  REALIZED (batch): ${realized:.6f}")
    print(f"  EST. sync (2×)  : ${est_sync:.6f}")
    print(f"  saving (−50%)   : ${est_sync - realized:.6f}")
    print("\n✓ batch path exercised end to end on dev tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
