"""Risk–coverage operating-point picker — choose the calibrated-confidence
threshold τ at which the LOCAL tier answers (auto) versus ESCALATES to remote.

This is the binary local-vs-escalate case (the dominant one): below τ we pay for
a remote call, at/above τ we answer locally for free. The picker turns a target
end-to-end accuracy into the *cheapest* τ that still hits it — and is honest when
no τ can (`clears_floor: False`).

It reuses the copied calibration package verbatim:
  * `recalibrate.fit_map(rows) -> breakpoints` + `recalibrate.apply_map(bp, x)`
    for the raw→calibrated map (isotonic / PAV),
  * `estimators.ece_debiased(rows, key)` for the promotion-gate calibration error.

Rows are the calibration package's shape: each carries `CONF_KEY`
("judge_confidence", the RAW pre-calibration score) and `CORRECT_KEY`
("correct", whether the LOCAL answer was right). For end-to-end projection a row
may also carry `"remote_correct"` (bool: would remote have been right on this
task).

Pure stdlib on purpose.
"""

from __future__ import annotations

from typing import Callable

from .calibration import estimators
from .calibration import recalibrate
from .schema import CONF_KEY, CORRECT_KEY

REMOTE_CORRECT_KEY = "remote_correct"


def fit_calibrator(rows: list[dict]) -> Callable[[float], float]:
    """Fit a raw→calibrated map from labeled rows and return it as a callable.

    Wraps `recalibrate.fit_map` (isotonic/PAV over {CONF_KEY, CORRECT_KEY}) and
    closes over the breakpoints. Empty / degenerate input yields the identity map
    (`fit_map` returns `[]`, and `apply_map([], x) == x`)."""
    bp = recalibrate.fit_map(rows)

    def _calibrator(x: float) -> float:
        return recalibrate.apply_map(bp, x)

    return _calibrator


def _cal_confidences(rows: list[dict], calibrator: Callable[[float], float] | None) -> list[float]:
    """Calibrated confidence per row (raw passed through if no calibrator)."""
    if calibrator is None:
        return [r[CONF_KEY] for r in rows]
    return [calibrator(r[CONF_KEY]) for r in rows]


def _tau_grid(cal_confs: list[float]) -> list[float]:
    """Sorted unique calibrated confidences, plus the 0.0 and 1.0 endpoints."""
    return sorted(set(cal_confs) | {0.0, 1.0})


def risk_coverage_curve(
    rows: list[dict],
    calibrator: Callable[[float], float] | None = None,
) -> list[dict]:
    """Risk–coverage curve over a grid of thresholds τ.

    At each τ the LOCAL tier answers the rows with `cal_conf >= τ` (the "auto"
    slice); the rest escalate. Each point reports:
        - `tau`:      the threshold,
        - `coverage`: fraction answered locally (n_auto / N),
        - `risk`:     LOCAL error rate on the auto slice (0.0 if the slice is empty),
        - `n_auto`:   size of the auto slice.

    As τ rises coverage is non-increasing (the slice shrinks) and risk is
    typically non-increasing (the retained answers are higher-confidence).
    Returns `[]` on empty input."""
    if not rows:
        return []
    cal_confs = _cal_confidences(rows, calibrator)
    correct = [bool(r[CORRECT_KEY]) for r in rows]
    n = len(rows)

    curve: list[dict] = []
    for tau in _tau_grid(cal_confs):
        auto = [c for cc, c in zip(cal_confs, correct) if cc >= tau]
        n_auto = len(auto)
        n_wrong = sum(1 for c in auto if not c)
        risk = (n_wrong / n_auto) if n_auto else 0.0
        curve.append({
            "tau": tau,
            "coverage": n_auto / n,
            "risk": risk,
            "n_auto": n_auto,
        })
    return curve


def _projected_accuracy(
    cal_confs: list[float],
    rows: list[dict],
    tau: float,
    remote_accuracy: float | None,
) -> tuple[float, int]:
    """Projected end-to-end accuracy at τ + the auto-slice size.

    Auto slice (cal_conf >= τ): contributes LOCAL correctness. Escalated slice:
    contributes per-task `remote_correct` when present, else the scalar
    `remote_accuracy` (0.0 if neither is available — we never invent remote skill)."""
    n = len(rows)
    hits = 0.0
    n_auto = 0
    for cc, r in zip(cal_confs, rows):
        if cc >= tau:
            n_auto += 1
            hits += 1.0 if r[CORRECT_KEY] else 0.0
        else:
            if REMOTE_CORRECT_KEY in r:
                hits += 1.0 if r[REMOTE_CORRECT_KEY] else 0.0
            elif remote_accuracy is not None:
                hits += remote_accuracy
            # else: no remote signal → contributes 0.0
    return hits / n, n_auto


def pick_threshold(
    rows: list[dict],
    accuracy_floor: float,
    margin: float = 0.03,
    remote_accuracy: float | None = None,
    calibrator: Callable[[float], float] | None = None,
) -> dict:
    """Pick the CHEAPEST τ whose projected end-to-end accuracy clears the floor.

    Projected accuracy at τ = [local-correct on the auto slice (cal_conf >= τ)]
    + [remote-correct on the escalated slice] all / N, where the escalated
    contribution is per-task `remote_correct` if present else the scalar
    `remote_accuracy`.

    Returns the LOWEST τ with projected accuracy >= `accuracy_floor + margin`
    (lowest τ = most local coverage = cheapest), as:
        {tau, projected_accuracy, coverage, escalation_rate, clears_floor: True}.
    If no τ clears the bar, returns the MOST ACCURATE τ with
    `clears_floor: False` (honest — no pretending). Empty input returns a
    degenerate τ=1.0 record with `clears_floor: False`."""
    target = accuracy_floor + margin
    if not rows:
        return {
            "tau": 1.0,
            "projected_accuracy": 0.0,
            "coverage": 0.0,
            "escalation_rate": 1.0,
            "clears_floor": False,
        }

    cal_confs = _cal_confidences(rows, calibrator)
    n = len(rows)
    grid = _tau_grid(cal_confs)

    best: dict | None = None  # highest projected accuracy seen (fallback)
    for tau in grid:  # ascending → first clearing τ is the lowest/cheapest
        acc, n_auto = _projected_accuracy(cal_confs, rows, tau, remote_accuracy)
        coverage = n_auto / n
        record = {
            "tau": tau,
            "projected_accuracy": acc,
            "coverage": coverage,
            "escalation_rate": 1.0 - coverage,
        }
        if acc >= target:
            return {**record, "clears_floor": True}
        if best is None or acc > best["projected_accuracy"]:
            best = record

    return {**best, "clears_floor": False}


def ece_of(rows: list[dict], calibrator: Callable[[float], float] | None = None) -> float:
    """Debiased calibration error of the (calibrated) confidences — the number
    the promotion gate checks before a confidence may be called `calibrated`.

    Builds `{cal, correct}` rows and defers to `estimators.ece_debiased`
    (Kumar et al. bias-corrected, equal-mass bins). Returns 0.0 on empty input
    (no calibration error to report)."""
    if not rows:
        return 0.0
    cal_confs = _cal_confidences(rows, calibrator)
    built = [{"cal": cc, "correct": bool(r[CORRECT_KEY])} for cc, r in zip(cal_confs, rows)]
    e = estimators.ece_debiased(built, "cal")
    return 0.0 if e is None else e


# ----------------------------------------------------------------- TODO: N-tier
# This picker is BINARY (one local-vs-escalate threshold). The N-tier
# generalization — jointly selecting a threshold per intermediate tier so the
# whole cost-ordered cascade hits a target accuracy at minimum expected spend —
# is a multi-dimensional operating-point search (one τ per non-final tier, the
# escalated slice of tier i becoming the input population of tier i+1). Not built
# yet: it needs per-tier correctness labels on the *escalated* sub-population, and
# a cost model to break ties among accuracy-equivalent threshold vectors. Add when
# the cascade has >2 tiers with measured per-tier outcomes.
