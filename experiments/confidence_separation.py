"""Confidence-separation diagnostic — the make-or-break question for the router, for $0.

Token-efficient routing only works if the confidence signal SEPARATES the local model's right
answers from its wrong ones (keep confident-correct local, escalate the rest). This measures that
directly, entirely on the LOCAL model (free):

  For each task: draw a self-consistency candidate (modal answer + agreement = signal A); grade that
  SAME candidate with the local model (self-verify → P(correct) = signal B). Both predict the one
  candidate's correctness, so it's a fair head-to-head.

Reports, per signal (self-consistency / verifier / their mean):
  * AUC — does higher confidence rank correct above incorrect? (0.5 = useless, 1.0 = perfect)
  * coverage→accuracy — if we keep the top-X% most-confident LOCAL, what accuracy? (the router's lever)
  * the confident-WRONG count and whether the verifier catches what self-consistency misses.

Run:  set -a; . ../.env; set +a
      uv run --extra serving --extra data python -m experiments.confidence_separation --n 50 --sc 5
"""

from __future__ import annotations

import argparse
import json
import re

from router_agent import confidence, tasks
from router_agent.config import DEV_CONFIG
from router_agent.schema import CostTracker

_PROB = re.compile(r"(?<![\d.])(0?\.\d+|1(?:\.0+)?|0)(?![\d.])")


def parse_prob(text: str) -> float:
    m = _PROB.search(text or "")
    if not m:
        return 0.5
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return 0.5


def auc(pairs: list[tuple[float, bool]]) -> float:
    """Fraction of (correct, incorrect) pairs the score ranks correctly (ties = 0.5)."""
    pos = [s for s, c in pairs if c]
    neg = [s for s, c in pairs if not c]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def coverage_accuracy(pairs: list[tuple[float, bool]], cov: float) -> tuple[float, int]:
    """Keep the top-`cov` fraction by confidence; return (accuracy on kept, n kept)."""
    if not pairs:
        return float("nan"), 0
    ordered = sorted(pairs, key=lambda x: x[0], reverse=True)
    k = max(1, round(cov * len(ordered)))
    kept = ordered[:k]
    return sum(1 for _s, c in kept if c) / len(kept), k


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Does the confidence signal separate right from wrong?")
    p.add_argument("--n", type=int, default=50, help="total mixed tasks (half math, half QA)")
    p.add_argument("--sc", type=int, default=5, help="self-consistency samples")
    p.add_argument("--out", default="/tmp/confidence_separation.json")
    args = p.parse_args(argv)

    from router_agent.providers import build_tier

    tracker = CostTracker()
    local_cfg = next(c for c in DEV_CONFIG.tiers if c.is_local)
    hot = build_tier(local_cfg, tracker, temperature=DEV_CONFIG.self_consistency_temp,
                     max_tokens=DEV_CONFIG.local_max_tokens)      # for self-consistency
    cold = build_tier(local_cfg, tracker, temperature=0.0, max_tokens=64)  # for grading (terse)
    sc_fn = confidence.self_consistency(n=args.sc, extract=tasks.extract_answer)

    half = args.n // 2
    task_list = tasks.load_gsm8k(n=half) + tasks.load_short_qa(n=args.n - half)
    print(f"loaded {len(task_list)} mixed tasks; scoring locally (free)…", flush=True)

    sc_pairs: list[tuple[float, bool]] = []   # (agreement, candidate_correct)
    vf_pairs: list[tuple[float, bool]] = []   # (verifier_prob, candidate_correct)
    confident_wrong = []                      # candidates self-consistency was sure of but got wrong
    for i, t in enumerate(task_list):
        res = sc_fn(hot.call, t)              # modal candidate + agreement
        candidate = res.answer
        correct = tasks.check(candidate, t)
        grade = cold.call(
            "You are a strict grader. Reply with ONLY a probability between 0 and 1.",
            f"Question: {t.prompt}\n\nProposed answer: {candidate}\n\n"
            "Probability the proposed answer is correct (0-1):",
        )
        v = parse_prob(grade.text)
        sc_pairs.append((res.raw, correct))
        vf_pairs.append((v, correct))
        if res.raw >= 0.8 and not correct:
            confident_wrong.append({"sc_raw": round(res.raw, 3), "verifier": round(v, 3),
                                    "kind": t.kind})
        if (i + 1) % 10 == 0:
            print(f"  …{i + 1}/{len(task_list)}", flush=True)

    mean_pairs = [((sc + vf) / 2, c) for (sc, c), (vf, _c) in zip(sc_pairs, vf_pairs)]
    base = sum(1 for _s, c in sc_pairs if c) / len(sc_pairs)
    covs = [1.0, 0.8, 0.6, 0.4, 0.2]
    report = {
        "n": len(sc_pairs),
        "base_local_accuracy": round(base, 4),
        "auc": {
            "self_consistency": round(auc(sc_pairs), 4),
            "verifier": round(auc(vf_pairs), 4),
            "mean": round(auc(mean_pairs), 4),
        },
        "coverage_accuracy": {
            name: {f"{int(c*100)}%": round(coverage_accuracy(pairs, c)[0], 4) for c in covs}
            for name, pairs in [("self_consistency", sc_pairs), ("verifier", vf_pairs),
                                ("mean", mean_pairs)]
        },
        "confident_wrong": {
            "count": len(confident_wrong),
            "mean_verifier_on_them": round(
                sum(d["verifier"] for d in confident_wrong) / len(confident_wrong), 4
            ) if confident_wrong else None,
            "cases": confident_wrong,
        },
    }
    print(json.dumps(report, indent=2), flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nsaved {args.out} (local-only, $0 remote)", flush=True)

    # one-line verdict
    a = report["auc"]
    best = max(a, key=lambda k: a[k] if a[k] == a[k] else -1)  # ignore nan
    print(f"VERDICT: best separator = {best} (AUC {a[best]}); base acc {base:.2f}. "
          f"Confident-wrong cases: {len(confident_wrong)} "
          f"(verifier mean on them {report['confident_wrong']['mean_verifier_on_them']}).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
