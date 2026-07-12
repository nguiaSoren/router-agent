"""Calibrate the local tier on the 8-category dev set (local-only, $0, CPU-forced).

Runs the local GGUF (N=2 self-consistency) over the checkable categories, maps its confidence to
correctness, and picks the LOWEST τ whose kept-local accuracy clears a target — so we keep local
only where it's reliably right and escalate everything else. Saves a calibrator JSON the container
can bundle (`ROUTER_CALIBRATOR`). Judge-only categories (summary / code-debug / logic) have no local
checker, so they're excluded here and escalate by low confidence at runtime.

Run: uv run --extra dev --extra serving --extra local python -m experiments.calibrate_local
"""

from __future__ import annotations

import argparse
import json
import os

from tokengolf import confidence, tasks
from tokengolf import local_llm
from tokengolf.calibration.recalibrate import apply_map, fit_map
from tokengolf.schema import CONF_KEY, CORRECT_KEY, Task
from tokengolf.threshold import risk_coverage_curve


def load_checkable(n_per: int) -> list[Task]:
    """Categories with a real local checker (gold + extractable)."""
    out: list[Task] = tasks.load_gsm8k(n=n_per) + tasks.load_short_qa(n=n_per)
    for name, fn in [("sentiment", tasks.load_sentiment), ("ner", tasks.load_ner),
                     ("code_generation", tasks.load_code_generation)]:
        try:
            out += fn(n=n_per)
        except Exception as e:  # noqa: BLE001 - dataset hiccup shouldn't sink the whole run
            print(f"  (skip {name}: {type(e).__name__}: {e})", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Calibrate the local tier on the 8-category dev set.")
    p.add_argument("--n-per", type=int, default=10, help="tasks per category")
    p.add_argument("--sc", type=int, default=2, help="self-consistency samples (matches the N=2 default)")
    p.add_argument("--max-tokens", type=int, default=512, help="local output cap (short = fast → higher N fits 30s)")
    p.add_argument("--target", type=float, default=0.90, help="target kept-local accuracy")
    p.add_argument("--out", default="results/local_calibrator.json")
    args = p.parse_args(argv)

    from llama_cpp import Llama
    llm = Llama(model_path=local_llm.resolve_gguf(), n_ctx=4096,
                n_threads=os.cpu_count(), n_gpu_layers=0, verbose=False)  # CPU proxy for the judging VM
    tier = local_llm.build_local_tier(max_tokens=args.max_tokens, temperature=0.7, llm=llm)
    scfn = confidence.self_consistency(n=args.sc, extract=tasks.extract_answer)

    task_list = load_checkable(args.n_per)
    print(f"scoring {len(task_list)} tasks locally (N={args.sc}, CPU)…", flush=True)
    rows: list[dict] = []
    per_cat: dict[str, list[bool]] = {}
    for t in task_list:
        res = scfn(tier.call, t)
        correct = tasks.check(res.answer, t)
        if correct is True or correct is False:  # skip judge-only sentinels
            rows.append({CONF_KEY: res.raw, CORRECT_KEY: bool(correct)})
            per_cat.setdefault(t.kind, []).append(bool(correct))

    bp = fit_map(rows)
    calibrator = (lambda x: apply_map(bp, x))
    curve = risk_coverage_curve(rows, calibrator=calibrator)

    # lowest τ (max coverage) whose kept-local accuracy (1 - risk) clears the target
    pick = None
    for pt in sorted(curve, key=lambda q: q["tau"]):
        if pt["coverage"] > 0 and (1.0 - pt["risk"]) >= args.target:
            pick = pt
            break
    tau = pick["tau"] if pick else 1.0  # nothing safe enough → escalate everything (τ=1.0)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"breakpoints": bp, "tau": tau, "target": args.target,
                   "n": len(rows), "sc_n": args.sc}, fh, indent=2)

    base = sum(1 for r in rows if r[CORRECT_KEY]) / (len(rows) or 1)
    print("\n=== per-category local accuracy (checkable) ===", flush=True)
    for k, v in sorted(per_cat.items()):
        print(f"  {k:20} {sum(v)/len(v):.2f}  (n={len(v)})", flush=True)
    print(f"\nbase local accuracy (all): {base:.3f}  over n={len(rows)}", flush=True)
    if pick:
        print(f"τ = {tau:.3f}  → keep-local coverage {pick['coverage']:.0%} "
              f"at kept-accuracy {(1 - pick['risk']):.2f} (target {args.target}); rest escalate", flush=True)
    else:
        print(f"No τ reaches target {args.target} → τ=1.0 (escalate everything); "
              f"local too unreliable at N={args.sc} on this mix", flush=True)
    print(f"saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
