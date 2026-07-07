"""Held-out dev rehearsal — the closest dress rehearsal to launch day we can run pre-kickoff.

Bake-off ONCE per task (local self-consistency confidence + correctness, one remote reference call
+ its token count), then do everything else as $0 CPU reanalysis:
  * deterministic train/test split (parity — both halves get math + QA),
  * fit the calibrator + pick τ on TRAIN only,
  * route the TEST split by REPLAYING the recorded confidences (no re-draw → no calibrate-vs-route
    noise), reporting held-out accuracy / local-coverage / remote-token total vs the all-local and
    all-remote baselines.
Sweeps self-consistency N (remote is independent of N, so it's called once and reused — minimal spend).

Run:  set -a; . ../.env; set +a
      uv run --extra serving --extra data python -m experiments.dev_rehearsal --n 24 --sc 3 6 --floor 0.80
"""

from __future__ import annotations

import argparse
import json

from router_agent import confidence, tasks
from router_agent.config import DEV_CONFIG
from router_agent.run import build_tiers
from router_agent.schema import CONF_KEY, CORRECT_KEY, CostTracker
from router_agent.threshold import REMOTE_CORRECT_KEY, ece_of, fit_calibrator, pick_threshold


def load_mixed(n: int) -> list:
    half = n // 2
    return tasks.load_gsm8k(n=half) + tasks.load_short_qa(n=n - half)


def remote_pass(task_list, remote_tier, check) -> dict[str, dict]:
    """One remote reference call per task (reused across all sc-N settings)."""
    ref: dict[str, dict] = {}
    for t in task_list:
        reply = remote_tier.call("", t.prompt)
        ref[t.id] = {
            REMOTE_CORRECT_KEY: check(reply.text, t),
            "remote_tokens": reply.in_tok + reply.out_tok,
        }
    return ref


def bake_rows(task_list, local_tier, sc_n: int, ref: dict[str, dict], check) -> list[dict]:
    """Local self-consistency confidence + correctness per task, joined with the remote reference."""
    conf_fn = confidence.self_consistency(n=sc_n, extract=tasks.extract_answer)
    rows: list[dict] = []
    for t in task_list:
        res = conf_fn(local_tier.call, t)
        rows.append({
            "task_id": t.id,
            CONF_KEY: res.raw,
            CORRECT_KEY: check(res.answer, t),
            REMOTE_CORRECT_KEY: ref[t.id][REMOTE_CORRECT_KEY],
            "remote_tokens": ref[t.id]["remote_tokens"],
        })
    return rows


def simulate_test(test_rows, calibrator, tau: float) -> dict:
    """Replay the cascade on the held-out rows: keep local when calibrated conf >= τ, else escalate."""
    n = len(test_rows) or 1
    correct = local = remote_tokens = 0
    for r in test_rows:
        cal = calibrator(r[CONF_KEY])
        if cal >= tau:                              # keep local (free)
            local += 1
            correct += 1 if r[CORRECT_KEY] else 0
        else:                                       # escalate (pay remote tokens)
            remote_tokens += r["remote_tokens"]
            correct += 1 if r[REMOTE_CORRECT_KEY] else 0
    all_remote_tokens = sum(r["remote_tokens"] for r in test_rows)
    return {
        "cascade_accuracy": round(correct / n, 4),
        "local_coverage": round(local / n, 4),
        "remote_tokens": remote_tokens,
        "all_local_accuracy": round(sum(1 for r in test_rows if r[CORRECT_KEY]) / n, 4),
        "all_remote_accuracy": round(sum(1 for r in test_rows if r[REMOTE_CORRECT_KEY]) / n, 4),
        "all_remote_tokens": all_remote_tokens,
        # the headline Track-1 number: % of all-remote tokens saved by routing
        "token_savings_pct": round(1 - remote_tokens / all_remote_tokens, 4) if all_remote_tokens else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Held-out dev rehearsal with a self-consistency sweep.")
    p.add_argument("--n", type=int, default=24, help="total mixed tasks (half math, half QA)")
    p.add_argument("--sc", type=int, nargs="+", default=[3, 6], help="self-consistency N values to sweep")
    p.add_argument("--floor", type=float, default=0.80)
    p.add_argument("--out", default="/tmp/dev_rehearsal.json")
    args = p.parse_args(argv)

    tracker = CostTracker(ceiling_usd=DEV_CONFIG.budget_ceiling_usd)
    tiers = build_tiers(DEV_CONFIG, tracker)
    local_tier, remote_tier = tiers[0], tiers[-1]
    check = tasks.check

    task_list = load_mixed(args.n)
    print(f"loaded {len(task_list)} mixed tasks; remote reference pass…", flush=True)
    ref = remote_pass(task_list, remote_tier, check)

    results = []
    for sc_n in args.sc:
        print(f"\n=== self-consistency N={sc_n} ===", flush=True)
        rows = bake_rows(task_list, local_tier, sc_n, ref, check)
        train, test = rows[::2], rows[1::2]          # parity split (deterministic)
        calibrator = fit_calibrator(train)
        picked = pick_threshold(train, args.floor, margin=DEV_CONFIG.accuracy_margin, calibrator=calibrator)
        test_metrics = simulate_test(test, calibrator, picked["tau"])
        rec = {
            "sc_n": sc_n,
            "n_train": len(train), "n_test": len(test),
            "tau": round(picked["tau"], 4),
            "train_clears_floor": picked["clears_floor"],
            "test_ece": round(ece_of(test, calibrator=calibrator), 4),
            **test_metrics,
        }
        results.append(rec)
        print(json.dumps(rec, indent=2), flush=True)
        m = test_metrics
        print(f"  TEST  all-local {m['all_local_accuracy']}  |  all-remote {m['all_remote_accuracy']}"
              f" ({m['all_remote_tokens']} tok)  |  CASCADE {m['cascade_accuracy']}"
              f" @ {m['local_coverage']:.0%} local, {m['remote_tokens']} remote tok"
              f"  → {m['token_savings_pct']:.0%} tokens saved", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"floor": args.floor, "n": args.n, "spent_usd": round(tracker.spent, 6),
                   "results": results}, fh, indent=2)
    print(f"\nspent ${tracker.spent:.6f}; saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
