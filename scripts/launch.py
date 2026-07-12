"""LAUNCH-DAY RUNBOOK — one command that runs the whole July-7 flow.

Kickoff should be mechanical, not a scramble. This script chains the full pipeline
in a single invocation, sharing ONE `CostTracker` (so the budget kill-switch covers
both the calibration bake-off and the answering pass):

  1. load a LABELLED calibration set + the TEST set to answer (external JSONL of
     {id, prompt, gold?, kind?}, or a built-in dev rehearsal split into halves);
  2. build the cost-ordered tiers + per-tier confidence signals (default DEV_CONFIG);
  3. bake the calibration rows on the calib set (local self-consistency + one remote
     reference call per task);
  4. fit the raw->calibrated map, pick the cheapest threshold tau that clears the
     accuracy floor with margin, and PERSIST {breakpoints, tau, floor} (no silent
     promotion — tau is a held-out measurement, not an assertion);
  5. route the TEST set with that calibrator + tau injected as the local tier's gate
     (`dataclasses.replace` on the frozen Tier, exactly as `run.cmd_route` does);
  6. emit an ANSWERS file (one {id, answer, tier_used, used_remote, scored_tokens}
     per test task) + a REPORT JSON, and print the human-readable report + spend.

Why a capturing checker (step 5/6): `eval.evaluate_cascade` routes each task exactly
once and is the canonical scorer, but its `per_task` rows omit the answer TEXT the
answers file needs. Routing a second time to recover the text would DOUBLE launch-day
remote spend — the one thing this project exists to avoid. So we hand `evaluate_cascade`
a `check` closure that records each routed answer as a side effect of grading it, then
join those answers onto the canonical report. One routing pass, no edits to eval.

Heavy deps (`openai`, `datasets`) stay lazy — they are pulled only by the modules this
calls (`run.build_tiers`, the dev loaders), so `--help` and the offline tests need only
the stdlib core.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from functools import partial

from tokengolf.calibration.recalibrate import apply_map, fit_map
from tokengolf.config import DEV_CONFIG, CascadeConfig
from tokengolf.eval import build_calibration_rows, evaluate_cascade, format_report
from tokengolf.run import build_confidence_fns, build_tiers, save_calibrator
from tokengolf.schema import CONF_KEY, CORRECT_KEY, CostTracker, Task
from tokengolf.threshold import pick_threshold

logger = logging.getLogger("launch")


# ----------------------------------------------------------------- task loading
def read_jsonl_tasks(path: str) -> list[Task]:
    """Read a JSONL of {id, prompt, gold?, kind?} into Tasks (run.py's reader pattern).

    Missing `gold` -> "" (an UNLABELLED test task — accuracy is then N/A); missing
    `kind` -> "qa" (the default verifier); missing `id` -> the line index."""
    tasks: list[Task] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tasks.append(
                Task(
                    id=str(obj.get("id", i)),
                    prompt=obj["prompt"],
                    gold=str(obj.get("gold", "")),
                    kind=obj.get("kind", "qa"),
                    meta=obj.get("meta", {}),
                )
            )
    return tasks


def _halve(xs: list[Task]) -> tuple[list[Task], list[Task]]:
    """Split a list at the midpoint: (first half -> calib, second half -> test)."""
    mid = len(xs) // 2
    return xs[:mid], xs[mid:]


def load_calib_and_test(args: argparse.Namespace) -> tuple[list[Task], list[Task]]:
    """Resolve the (calibration, test) task pair from the CLI.

    Two modes: external JSONL (`--calib-tasks` + `--test-tasks`), or a built-in dev
    rehearsal (`--dev gsm8k|qa|mix --n N`) loaded then split into halves. For `mix`
    each kind is split independently so BOTH halves carry both kinds. Loaded/sampled
    counts are logged (no silent caps)."""
    if args.dev:
        from tokengolf import tasks as tasks_mod  # lazy: pulls `datasets`

        which, n = args.dev, args.n
        if which == "gsm8k":
            calib, test = _halve(tasks_mod.load_gsm8k(n=n))
        elif which == "qa":
            calib, test = _halve(tasks_mod.load_short_qa(n=n))
        else:  # mix — split each kind so both halves stay representative
            half = max(1, n // 2)
            gc, gt = _halve(tasks_mod.load_gsm8k(n=half))
            qc, qt = _halve(tasks_mod.load_short_qa(n=n - half))
            calib, test = gc + qc, gt + qt
        logger.info("dev rehearsal %r n=%d -> calib=%d test=%d", which, n, len(calib), len(test))
        return calib, test

    if not (args.calib_tasks and args.test_tasks):
        raise SystemExit("launch needs --calib-tasks AND --test-tasks, or --dev <gsm8k|qa|mix>")
    calib = read_jsonl_tasks(args.calib_tasks)
    test = read_jsonl_tasks(args.test_tasks)
    logger.info("loaded calib=%d (%s) test=%d (%s)", len(calib), args.calib_tasks, len(test), args.test_tasks)
    return calib, test


# ----------------------------------------------------------------- the flow
def run_launch(args: argparse.Namespace, config: CascadeConfig | None = None) -> dict:
    """The end-to-end launch flow (steps 1-6). Returns the emitted report dict."""
    config = config or DEV_CONFIG
    floor = args.floor if getattr(args, "floor", None) is not None else config.accuracy_floor

    # ONE tracker for the whole flow: the bake-off AND the answering pass both bill it,
    # so the budget ceiling is a hard kill-switch over the entire launch.
    tracker = CostTracker(ceiling_usd=config.budget_ceiling_usd)

    calib_tasks, test_tasks = load_calib_and_test(args)

    # --- build tiers + signals ONCE (shared across calibration and routing).
    tiers = build_tiers(config, tracker)
    if len(tiers) < 2:
        raise SystemExit("launch needs at least a local + a remote tier")
    fns = build_confidence_fns(config, tiers)
    local_tier, remote_tier = tiers[0], tiers[-1]
    local_conf_fn = fns[0]

    # --- (3) bake calibration rows on the LABELLED calib set.
    logger.info("baking calibration rows on %d calib tasks ...", len(calib_tasks))
    rows = build_calibration_rows(
        calib_tasks, local_tier=local_tier, local_conf_fn=local_conf_fn, remote_tier=remote_tier
    )

    # --- (4) fit the calibrator + pick the cheapest tau that clears the floor; persist.
    bp = fit_map([{CONF_KEY: r[CONF_KEY], CORRECT_KEY: r[CORRECT_KEY]} for r in rows])
    calibrator = partial(apply_map, bp)
    picked = pick_threshold(rows, floor, margin=config.accuracy_margin, calibrator=calibrator)
    save_calibrator(
        args.calibrator_out, bp, picked["tau"], floor,
        {
            "n_calib": len(rows),
            "clears_floor": picked["clears_floor"],
            "projected_accuracy": round(picked["projected_accuracy"], 4),
            "coverage": round(picked["coverage"], 4),
            "spent_usd_after_calib": round(tracker.spent, 6),
        },
    )
    logger.info(
        "picked tau=%.4f clears_floor=%s (projected acc=%.4f, local coverage=%.4f) -> %s",
        picked["tau"], picked["clears_floor"], picked["projected_accuracy"], picked["coverage"],
        args.calibrator_out,
    )
    if not picked["clears_floor"]:
        print(
            "WARNING: no threshold clears the accuracy floor with margin — the local model "
            "may be too weak, or the floor demands more remote coverage than is available.",
            file=sys.stderr,
        )

    # --- (5) inject tau as the local tier's gate (frozen Tier -> replace), then route.
    tiers[0] = dataclasses.replace(tiers[0], threshold=picked["tau"])
    calibrators = {tiers[0].name: calibrator}

    # Capture each routed answer as a side effect of grading (single routing pass).
    from tokengolf import tasks as tasks_mod

    captured: dict[str, str] = {}

    def capturing_check(pred_text: str, task: Task) -> bool:
        captured[task.id] = pred_text
        return tasks_mod.check(pred_text, task)

    logger.info("routing %d test tasks through the cascade ...", len(test_tasks))
    report = evaluate_cascade(test_tasks, tiers, fns, calibrators=calibrators, check=capturing_check)

    # --- (6) emit answers + report.
    with open(args.out_answers, "w", encoding="utf-8") as fh:
        for row in report["per_task"]:
            fh.write(json.dumps({
                "id": row["task_id"],
                "answer": captured.get(row["task_id"], ""),
                "tier_used": row["tier_used"],
                "used_remote": row["used_remote"],
                "scored_tokens": row["scored_tokens"],
            }) + "\n")

    # accuracy only when every test task carried a gold label (else it's meaningless).
    has_gold = bool(test_tasks) and all(t.gold for t in test_tasks)
    report_out = {
        "n": report["n"],
        "coverage_local": round(report["coverage"], 4),
        "total_remote_tokens": report["total_remote_tokens"],
        "spent_usd": round(tracker.spent, 6),
        "chosen_tau": picked["tau"],
        "clears_floor": picked["clears_floor"],
    }
    if has_gold:
        report_out["accuracy"] = round(report["accuracy"], 4)
    with open(args.out_report, "w", encoding="utf-8") as fh:
        json.dump(report_out, fh, indent=2)

    print(format_report(report))
    if not has_gold:
        print("  accuracy             : N/A (test tasks carry no gold)")
    print(f"  chosen tau           : {picked['tau']:.4f}  (clears_floor={picked['clears_floor']})")
    print(f"  spent (usd)          : {tracker.spent:.4f}")
    print(f"  answers -> {args.out_answers}")
    print(f"  report  -> {args.out_report}")
    return report_out


# ----------------------------------------------------------------- argparse
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="launch",
        description="LAUNCH-DAY RUNBOOK — calibrate + route in one command.",
    )
    p.add_argument("--calib-tasks", dest="calib_tasks", help="LABELLED calib JSONL {id,prompt,gold,kind?}")
    p.add_argument("--test-tasks", dest="test_tasks", help="TEST JSONL to answer {id,prompt,gold?,kind?}")
    p.add_argument("--dev", choices=["gsm8k", "qa", "mix"],
                   help="built-in dev rehearsal (loaded then split into calib/test halves)")
    p.add_argument("--n", type=int, default=20, help="number of dev tasks (with --dev)")
    p.add_argument("--floor", type=float, default=None,
                   help="accuracy floor to clear (default: config.accuracy_floor)")
    p.add_argument("--out-answers", dest="out_answers", default="answers.jsonl",
                   help="where to write per-task answers (JSONL)")
    p.add_argument("--out-report", dest="out_report", default="report.json",
                   help="where to write the leaderboard-shaped report (JSON)")
    p.add_argument("--calibrator-out", dest="calibrator_out", default="calibrator.json",
                   help="where to persist the fitted calibrator + tau")
    return p


def main(argv: list[str] | None = None, config: CascadeConfig | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    run_launch(args, config=config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
