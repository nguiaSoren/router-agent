"""Confidence-signal sweep — which self-consistency setting separates right from wrong best, for $0.

The diagnostic `experiments/confidence_separation.py` established self-consistency AUC 0.807 at N=5
(temp 0.7). This harness asks the method-level tuning question: *which N* (and would a margin signal
beat agreement?) gives the cleanest separation of the local model's correct answers from its wrong ones.

EFFICIENCY TRICK — draw once, sub-sample the rest. For each task we draw `N_max` (default 10) local
samples ONCE at DEV_CONFIG.self_consistency_temp; every smaller-N agreement is then computed from a
PREFIX of those samples — no re-drawing. Correctness is fixed ONCE per task (the modal key over all
N_max samples), so the ONLY thing that varies across N is the confidence value: a clean comparison.

Per task:
  * draw N_max samples; record each sample's `extract_answer`-normalized key (and the original text);
  * candidate = the global modal key over all N_max; `candidate_correct = check(rep_text, task)`;
  * for N in {1,3,5,8,10}: agreement_N = agreement_prefix(keys, N) = count(modal-of-first-N)/N;
  * margin (at N_max) = (top_count - second_count) / N_max.

Reports (print + JSON to --out): n, base_local_accuracy, auc_by_N, auc_margin, best_N, and the
coverage→accuracy curve for the best N. A one-line VERDICT names the best N and its AUC vs N=5.

Run:  set -a; . ../.env; set +a
      uv run --extra serving --extra data python -m experiments.confidence_sweep --n 48 --nmax 10
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from tokengolf import tasks
from tokengolf.config import DEV_CONFIG
from tokengolf.schema import CostTracker

from experiments.confidence_separation import auc, coverage_accuracy

# Candidate self-consistency budgets to compare (clamped to N_max at runtime).
SWEEP_NS = (1, 3, 5, 8, 10)
# Coverage fractions for the risk-coverage curve (the router's keep-top-X% lever).
COVS = (1.0, 0.8, 0.6, 0.4, 0.2)


# --------------------------------------------------------------- pure sub-sampling helpers
def agreement_prefix(keys: list[str], n: int) -> float:
    """Agreement among the FIRST `n` samples: count(modal key within the prefix) / len(prefix).

    The modal key is recomputed within the prefix (not inherited from the full draw), so each N is
    the honest agreement an N-sample run would have observed. `n` is clamped to len(keys); n<=0 or
    empty keys returns nan.
    """
    prefix = keys[: max(0, n)]
    if not prefix:
        return float("nan")
    modal_count = Counter(prefix).most_common(1)[0][1]
    return modal_count / len(prefix)


def margin(keys: list[str]) -> float:
    """Vote margin over all samples: (top_count - second_count) / len(keys).

    A single distinct key (or one sample) → second_count 0 → margin = top/len. Empty → nan.
    """
    if not keys:
        return float("nan")
    counts = Counter(keys).most_common()
    top = counts[0][1]
    second = counts[1][1] if len(counts) > 1 else 0
    return (top - second) / len(keys)


def _modal_representative(keys: list[str], texts: list[str]) -> str:
    """The first original text whose normalized key is the global modal key over all samples."""
    modal_key = Counter(keys).most_common(1)[0][0]
    return next(t for t, k in zip(texts, keys) if k == modal_key)


# --------------------------------------------------------------- the sweep
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Which self-consistency N / signal separates best?")
    p.add_argument("--n", type=int, default=48, help="total mixed tasks (half math, half QA)")
    p.add_argument("--nmax", type=int, default=10, help="local samples drawn ONCE per task")
    p.add_argument("--out", default="/tmp/confidence_sweep.json")
    args = p.parse_args(argv)

    sweep_ns = [n for n in SWEEP_NS if n <= args.nmax]
    if not sweep_ns:
        sweep_ns = [args.nmax]

    from tokengolf.providers import build_tier

    tracker = CostTracker()
    local_cfg = next(c for c in DEV_CONFIG.tiers if c.is_local)
    hot = build_tier(
        local_cfg, tracker,
        temperature=DEV_CONFIG.self_consistency_temp,
        max_tokens=DEV_CONFIG.local_max_tokens,
    )

    half = args.n // 2
    task_list = tasks.load_gsm8k(n=half) + tasks.load_short_qa(n=args.n - half)
    print(
        f"loaded {len(task_list)} mixed tasks; drawing {args.nmax} local samples each (free)…",
        flush=True,
    )

    # Per task: candidate_correct (fixed) + the N_max normalized keys (for prefix sub-sampling).
    records: list[tuple[bool, list[str]]] = []
    for i, t in enumerate(task_list):
        keys: list[str] = []
        texts: list[str] = []
        for _ in range(args.nmax):
            reply = hot.call("", t.prompt)
            texts.append(reply.text)
            keys.append(tasks.extract_answer(reply.text, t))
        rep = _modal_representative(keys, texts)
        candidate_correct = tasks.check(rep, t)
        records.append((candidate_correct, keys))
        if (i + 1) % 10 == 0:
            print(f"  …{i + 1}/{len(task_list)}", flush=True)

    base = sum(1 for c, _k in records if c) / len(records)

    # AUC per N from prefixes; AUC for the margin signal at N_max.
    pairs_by_n: dict[int, list[tuple[float, bool]]] = {
        n: [(agreement_prefix(keys, n), c) for c, keys in records] for n in sweep_ns
    }
    auc_by_n = {n: round(auc(pairs_by_n[n]), 4) for n in sweep_ns}
    margin_pairs = [(margin(keys), c) for c, keys in records]
    auc_margin = round(auc(margin_pairs), 4)

    # Best N = argmax AUC (ignoring nan); risk-coverage curve for that N.
    best_n = max(sweep_ns, key=lambda n: auc_by_n[n] if auc_by_n[n] == auc_by_n[n] else -1.0)
    best_cov = {
        f"{int(c * 100)}%": round(coverage_accuracy(pairs_by_n[best_n], c)[0], 4) for c in COVS
    }

    report = {
        "n": len(records),
        "base_local_accuracy": round(base, 4),
        "auc_by_N": auc_by_n,
        "auc_margin": auc_margin,
        "best_N": best_n,
        "coverage_accuracy_for_best_N": best_cov,
    }
    print(json.dumps(report, indent=2), flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nsaved {args.out} (local-only, $0 remote)", flush=True)

    baseline = auc_by_n.get(5)
    base_str = f"{baseline}" if baseline is not None else "n/a (N=5 not in sweep)"
    print(
        f"VERDICT: best N = {best_n} (AUC {auc_by_n[best_n]}); N=5 baseline AUC {base_str}; "
        f"margin-signal AUC {auc_margin}; base local acc {base:.2f}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
