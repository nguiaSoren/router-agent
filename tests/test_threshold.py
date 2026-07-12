"""Offline tests for the risk–coverage threshold picker — synthetic labeled rows.

Higher confidence ⇒ higher P(correct), so: the fitted calibrator is monotone,
the risk–coverage coverage is non-increasing, and the picker returns the cheapest
τ that meets a reachable floor (and is honest when none does).
"""

from __future__ import annotations

from tokengolf.schema import CONF_KEY, CORRECT_KEY
from tokengolf.threshold import (
    REMOTE_CORRECT_KEY,
    ece_of,
    fit_calibrator,
    pick_threshold,
    risk_coverage_curve,
)


def _rows() -> list[dict]:
    """Confidence bands with monotonically increasing empirical accuracy.

    conf 0.5→2/10, 0.6→4/10, 0.7→6/10, 0.8→8/10, 0.9→10/10 correct.
    Remote is strong (remote_correct=True everywhere) so escalating a low-conf
    row strictly helps projected end-to-end accuracy."""
    rows: list[dict] = []
    plan = [(0.5, 2), (0.6, 4), (0.7, 6), (0.8, 8), (0.9, 10)]
    for conf, n_correct in plan:
        for i in range(10):
            rows.append({
                CONF_KEY: conf,
                CORRECT_KEY: i < n_correct,
                REMOTE_CORRECT_KEY: True,
            })
    return rows


# --------------------------------------------------------------- calibrator is monotone
def test_fit_calibrator_is_monotone():
    cal = fit_calibrator(_rows())
    xs = [0.5, 0.6, 0.7, 0.8, 0.9]
    ys = [cal(x) for x in xs]
    assert ys == sorted(ys), f"calibrated map must be non-decreasing: {ys}"
    # the 0.9 band was all-correct, the 0.5 band mostly wrong → ordering preserved
    assert cal(0.9) > cal(0.5)


def test_fit_calibrator_empty_is_identity():
    cal = fit_calibrator([])
    assert cal(0.42) == 0.42


# --------------------------------------------------------------- risk–coverage curve
def test_risk_coverage_coverage_non_increasing():
    cal = fit_calibrator(_rows())
    curve = risk_coverage_curve(_rows(), calibrator=cal)
    covs = [pt["coverage"] for pt in curve]
    assert covs == sorted(covs, reverse=True), f"coverage must be non-increasing in τ: {covs}"
    # risk should (typically) also fall as we keep only higher-confidence answers
    risks = [pt["risk"] for pt in curve if pt["n_auto"] > 0]
    assert risks == sorted(risks, reverse=True), f"risk should be non-increasing in τ: {risks}"
    # endpoints present
    assert curve[0]["tau"] == 0.0 and curve[0]["coverage"] == 1.0


def test_risk_coverage_empty():
    assert risk_coverage_curve([]) == []


# --------------------------------------------------------------- pick_threshold
def test_pick_threshold_returns_lowest_clearing_tau():
    rows = _rows()
    cal = fit_calibrator(rows)
    # local-only accuracy (τ=0) is 30/50 = 0.6; remote is perfect, so escalation
    # lifts accuracy. Floor 0.85 + margin 0.03 = 0.88 is reachable only by escalating.
    out = pick_threshold(rows, accuracy_floor=0.85, margin=0.03, calibrator=cal)

    target = 0.85 + 0.03
    assert out["clears_floor"] is True
    assert out["projected_accuracy"] >= target - 1e-9
    # it is NOT the trivial all-local τ=0.0 (that one fails the floor)
    assert out["tau"] > 0.0

    # confirm "lowest": recompute projected accuracy at every grid τ strictly
    # below the chosen one and check none of them clears the target.
    grid = sorted({pt["tau"] for pt in risk_coverage_curve(rows, calibrator=cal)})
    cal_confs = [cal(r[CONF_KEY]) for r in rows]
    n = len(rows)
    for tau in grid:
        if tau >= out["tau"]:
            continue
        hits = sum(
            (1.0 if r[CORRECT_KEY] else 0.0) if cc >= tau else (1.0 if r[REMOTE_CORRECT_KEY] else 0.0)
            for cc, r in zip(cal_confs, rows)
        )
        assert hits / n < target, f"τ={tau} should NOT clear the target but did"


def test_pick_threshold_unreachable_floor_is_honest():
    rows = _rows()
    cal = fit_calibrator(rows)
    out = pick_threshold(rows, accuracy_floor=2.0, margin=0.03, calibrator=cal)
    assert out["clears_floor"] is False
    # best-effort: returns the most accurate τ achievable
    curve_accs = []
    for tau in sorted({pt["tau"] for pt in risk_coverage_curve(rows, calibrator=cal)}):
        single = pick_threshold(rows, accuracy_floor=tau, margin=999.0, calibrator=cal)
        curve_accs.append(single["projected_accuracy"])
    assert out["projected_accuracy"] >= max(curve_accs) - 1e-9


def test_pick_threshold_uses_scalar_remote_accuracy():
    # drop per-task remote labels → fall back to the scalar remote_accuracy.
    rows = [{CONF_KEY: r[CONF_KEY], CORRECT_KEY: r[CORRECT_KEY]} for r in _rows()]
    cal = fit_calibrator(rows)
    out = pick_threshold(rows, accuracy_floor=0.7, margin=0.0, remote_accuracy=1.0, calibrator=cal)
    assert "projected_accuracy" in out
    assert 0.0 <= out["projected_accuracy"] <= 1.0


def test_pick_threshold_empty():
    out = pick_threshold([], accuracy_floor=0.5)
    assert out["clears_floor"] is False
    assert out["tau"] == 1.0


# --------------------------------------------------------------- ece_of
def test_ece_of_runs():
    rows = _rows()
    cal = fit_calibrator(rows)
    e = ece_of(rows, calibrator=cal)
    assert isinstance(e, float)
    assert e >= 0.0


def test_ece_of_empty():
    assert ece_of([]) == 0.0
