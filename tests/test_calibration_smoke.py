"""Smoke test — the reused PARALLAX calibration package still runs intact here,
on router-shaped rows. If this fails after a copy, the import-path fix regressed.

Run:  PYTHONPATH=src python3 tests/test_calibration_smoke.py   (or: uv run pytest -q)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tokengolf.calibration import estimators as E
from tokengolf.calibration import nested
from tokengolf.calibration import recalibrate as R


def _rows(n: int = 60):
    # router-shaped: judge_confidence=route conf, correct=pick won, modality=group
    return [{"judge_confidence": round(0.5 + 0.45 * (i / (n - 1)), 3),
             "correct": (i % 4 != 0), "modality": "route"} for i in range(n)]


def test_report_improves_ece():
    rep = R.report(_rows(), by_modality=False)
    assert rep["improved"] is True
    assert rep["calibrated_loo_ece"] <= rep["raw_ece"] + 1e-9


def test_selective_picks_a_method():
    sel = R.selective(_rows())
    assert sel["chosen"]["route"]["method"] in {"platt", "isotonic", "raw", "none"}


def test_debiased_estimator_runs():
    assert E.ece_debiased(_rows(), "judge_confidence") is not None


def test_nested_cv_runs_noncircular():
    res = nested.nested_cv(_rows(), by_modality=False)
    assert "_caveat" in res and res["pairs"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} calibration smoke tests passed")
