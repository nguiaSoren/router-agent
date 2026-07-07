"""Eval + labeling harness — the instrument that turns the cascade into numbers.

Two jobs, one module:

  1. LABELING (`build_calibration_rows`): run the full local-vs-remote bake-off
     per task to produce the rows `threshold.py` consumes — each row carries the
     local tier's RAW confidence (`CONF_KEY`), whether the LOCAL answer was right
     (`CORRECT_KEY`), and whether REMOTE would have been right (`remote_correct`,
     read per-task by `pick_threshold`). Local sampling is free (the confidence
     fn may draw many local samples); remote is called exactly once per task.

  2. SCORING (`evaluate_cascade`): run the real `route(...)` per task and report
     the leaderboard-shaped pair — end-to-end `accuracy` and `total_remote_tokens`
     (remote-only, since local is free). `coverage` is the fraction answered
     locally (never escalated).

`calibrate_and_pick` is the honest convenience wrapper: it fits the calibrator on
a TRAIN split and scores τ / ECE / the risk-coverage curve on a disjoint TEST
split — never fit-and-scored on the same rows.

The correctness checker is a parameter (`check`, defaulting to `tasks.check`) so
the whole module stays offline-testable with a fake checker. Importing this module
does NOT pull `datasets` (`tasks`' loaders import it lazily).

Pure stdlib on purpose.
"""

from __future__ import annotations

from typing import Callable

from . import tasks as tasks_mod
from .cascade import route
from .schema import (
    CONF_KEY,
    CORRECT_KEY,
    CheckFn,
    ConfidenceFn,
    Task,
    Tier,
)
from .threshold import (
    REMOTE_CORRECT_KEY,
    ece_of,
    fit_calibrator,
    pick_threshold,
    risk_coverage_curve,
)


def _resolve_check(check: CheckFn | None) -> CheckFn:
    """The checker to use — the injected one, else the `tasks.check` dispatcher."""
    return check if check is not None else tasks_mod.check


# ----------------------------------------------------------------- 1. scoring
def evaluate_cascade(
    tasks: list[Task],
    tiers: list[Tier],
    confidence_fns: list[ConfidenceFn],
    *,
    calibrators: dict[str, Callable[[float], float]] | None = None,
    check: CheckFn | None = None,
) -> dict:
    """Run the real cascade per task and report accuracy + remote-token total.

    For each task: `route(...)` produces an answer; correctness is decided by
    `check`. Local-answered tasks contribute 0 to `total_remote_tokens` (local is
    free); escalated tasks contribute their remote tiers' scored tokens.

    Returns the leaderboard-shaped report:
        {n, accuracy, coverage, total_remote_tokens, mean_remote_tokens, per_task}
    where `coverage` is the fraction answered LOCALLY (used_remote is False) and
    `per_task` is a list of {task_id, tier_used, used_remote, scored_tokens, correct}.
    Empty input returns zeros (no division by zero)."""
    checker = _resolve_check(check)
    n = len(tasks)

    per_task: list[dict] = []
    n_correct = 0
    n_local = 0
    total_remote_tokens = 0

    for task in tasks:
        result = route(task, tiers, confidence_fns, calibrators=calibrators)
        correct = checker(result.answer, task)
        scored = result.scored_tokens

        n_correct += 1 if correct else 0
        n_local += 0 if result.used_remote else 1
        total_remote_tokens += scored

        per_task.append({
            "task_id": result.task_id,
            "tier_used": result.tier_used,
            "used_remote": result.used_remote,
            "scored_tokens": scored,
            "correct": correct,
        })

    return {
        "n": n,
        "accuracy": (n_correct / n) if n else 0.0,
        "coverage": (n_local / n) if n else 0.0,
        "total_remote_tokens": total_remote_tokens,
        "mean_remote_tokens": (total_remote_tokens / n) if n else 0.0,
        "per_task": per_task,
    }


# ----------------------------------------------------------------- 2. labeling
def build_calibration_rows(
    tasks: list[Task],
    *,
    local_tier: Tier,
    local_conf_fn: ConfidenceFn,
    remote_tier: Tier,
    check: CheckFn | None = None,
) -> list[dict]:
    """Full bake-off per task → the rows `fit_calibrator` / `pick_threshold` consume.

    For each task:
      * run `local_conf_fn(local_tier.call, task)` → ConfidenceResult; the local
        answer is `res.answer`, the RAW confidence is `res.raw`. The confidence fn
        may sample the local model many times (self-consistency) — free, local.
      * call `remote_tier.call("", task.prompt)` exactly ONCE → Reply.
      * grade both with `check`.

    Row = {task_id, CONF_KEY: raw, CORRECT_KEY: local_correct,
           "remote_correct": remote_correct}. `pick_threshold` reads the per-task
    `remote_correct` to project end-to-end accuracy on the escalated slice."""
    checker = _resolve_check(check)

    rows: list[dict] = []
    for task in tasks:
        res = local_conf_fn(local_tier.call, task)
        local_correct = checker(res.answer, task)

        reply = remote_tier.call("", task.prompt)
        remote_correct = checker(reply.text, task)

        rows.append({
            "task_id": task.id,
            CONF_KEY: res.raw,
            CORRECT_KEY: local_correct,
            REMOTE_CORRECT_KEY: remote_correct,
        })
    return rows


# ----------------------------------------------------------------- 3. calibrate + pick
def calibrate_and_pick(
    rows: list[dict],
    accuracy_floor: float,
    *,
    margin: float = 0.03,
) -> dict:
    """Fit the calibrator on TRAIN, score τ / ECE / curve on a disjoint TEST.

    Honesty: the calibrator is never fit and evaluated on the same rows. The split
    is deterministic by index parity (even → train, odd → test), which keeps both
    halves distributionally representative regardless of input ordering. With fewer
    than 2 rows the (degenerate) split puts everything in train and the test
    metrics fall back to their empty-input defaults.

    Returns {chosen_tau, projected_accuracy, coverage, clears_floor, test_ece, curve}."""
    train = rows[::2]
    test = rows[1::2]

    calibrator = fit_calibrator(train)
    picked = pick_threshold(test, accuracy_floor, margin=margin, calibrator=calibrator)

    return {
        "chosen_tau": picked["tau"],
        "projected_accuracy": picked["projected_accuracy"],
        "coverage": picked["coverage"],
        "clears_floor": picked["clears_floor"],
        "test_ece": ece_of(test, calibrator=calibrator),
        "curve": risk_coverage_curve(test, calibrator=calibrator),
    }


# ----------------------------------------------------------------- 4. report
def format_report(report: dict) -> str:
    """Human-readable one-block summary of an `evaluate_cascade` report."""
    lines = [
        "=== cascade eval ===",
        f"  tasks (n)            : {report['n']}",
        f"  accuracy             : {report['accuracy']:.4f}",
        f"  coverage (local frac): {report['coverage']:.4f}",
        f"  total remote tokens  : {report['total_remote_tokens']}",
        f"  mean remote tokens   : {report['mean_remote_tokens']:.2f}",
    ]
    return "\n".join(lines)
