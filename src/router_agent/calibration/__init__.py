"""Confidence calibration — a small, pure-stdlib package that turns raw scores into
calibrated probabilities plus an honest abstain policy.

Modules:
  * `recalibrate` — Platt + isotonic fits, `ece`, leave-one-out, and `selective`
    (per-group method choice + the abstain/identity policy). The core.
  * `estimators`  — binning-robust ECE: equal-mass, debiased (Kumar et al. 2019),
    sweep, bootstrap CI.
  * `analyze`     — a 3-number credibility report + promotion threshold.
  * `nested`      — nested cross-validation, the non-circular generalization estimate.
  * `apply`       — load a fitted model from `data/calibration/` and remap at the
    output edge (raw stays stored; calibration is applied only here).

Pure stdlib (no numpy/sklearn), MIT.
"""

__all__ = ["recalibrate", "estimators", "analyze", "nested", "apply"]
