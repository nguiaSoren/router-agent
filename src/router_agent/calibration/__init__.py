"""Confidence calibration — REUSED VERBATIM from PARALLAX's verifier.

This package is the load-bearing reuse behind GAUGE (see `README.md`, rung 3). The
five compute modules are copied verbatim from GAUGE's `gauge.calibration`, which in
turn copied them verbatim from PARALLAX's `parallax.calibration`; only the
intra-package import path differs on each hop (`parallax.` → `gauge.` →
`router_agent.`). Nothing here is reimplemented — that honesty is the point: the
calibrated/abstaining machinery already existed for verdicts, and GAUGE re-aims it
at routes.

Modules:
  * `recalibrate` — Platt + isotonic fits, `ece`, leave-one-out, and `selective`
    (per-group method choice + the abstain/identity policy). The core.
  * `estimators`  — binning-robust ECE: equal-mass, debiased (Kumar et al. 2019),
    sweep, bootstrap CI.
  * `analyze`     — the "3-number credibility" report + promotion threshold.
  * `nested`      — nested-CV, the non-circular generalization estimate.
  * `apply`       — load a fitted model from `data/calibration/` and remap at the
    output edge (raw stays stored; calibration is applied only here).

NOTE on what was *dropped* in the copy: PARALLAX's `PredictionItem` (a pydantic
card that packages judge verdicts for human labeling) is verifier-domain, not
generic calibration, so it did not come over. GAUGE's analogue — packaging routing
decisions for outcome labeling — is Phase 1 work (`BUILD_PLAN.md`). Keeping it out
is why the v0 core stays pure-stdlib.
"""

__all__ = ["recalibrate", "estimators", "analyze", "nested", "apply"]
