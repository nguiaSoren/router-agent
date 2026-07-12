"""Binning-robust calibration-error estimators.

The shipped headline ECE (``recalibrate.ece``) uses fixed EQUAL-WIDTH bins
(``int(x * n_bins)``). That estimator is known to be (i) binning-SENSITIVE — the
number declines as you change the bin count and edges (Nixon et al. 2019) — and
(ii) upward-BIASED — even a perfectly calibrated model scores a positive ECE
because each bin's empirical accuracy is a noisy estimate of its true accuracy
(Kumar, Liang, Ma, "Verified Uncertainty Calibration", NeurIPS 2019).

This module provides alternatives that make the calibration story binning-robust:

* ``ece_equal_mass`` — adaptive EQUAL-MASS binning (each bin holds ~N/n_bins
  rows), which removes the empty/near-empty high-confidence bins that destabilize
  equal-width ECE.
* ``ece_debiased`` — Kumar et al.'s bias-corrected plug-in estimator of the
  (root) calibration error, computed on equal-mass bins (their recommendation).
* ``ece_sweep`` — runs equal-width vs equal-mass across several bin counts so a
  reviewer can see the *ranking* survives the choice of bin count.
* ``bootstrap_ci`` — a deterministic nonparametric percentile bootstrap CI around
  any of the above.

Data shape (mirrors ``recalibrate.ece``): ``rows: list[dict]``, each row carries a
probability under string ``key`` (e.g. "cal", "raw", "v") and a boolean
``correct``. Pure Python stdlib only — no numpy/sklearn/scipy.
"""

from __future__ import annotations

import math
import random
from typing import Callable

from tokengolf.calibration.recalibrate import ece


# ---------------------------------------------------------------- equal-mass bins
def _equal_mass_bins(rows: list[dict], key: str, n_bins: int) -> list[list[dict]]:
    """Sort rows by ``row[key]`` and split into ``n_bins`` groups of as-equal-as-
    possible COUNT. The first ``N % n_bins`` bins get one extra row (remainder
    spread, not dumped in the last bin). If ``N < n_bins`` we make ``N`` singleton
    bins (degenerate-but-graceful: never an empty partition, never a crash)."""
    srt = sorted(rows, key=lambda r: r[key])
    n = len(srt)
    k = min(n_bins, n)  # can't have more non-empty bins than rows
    base, rem = divmod(n, k)
    bins: list[list[dict]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        bins.append(srt[start : start + size])
        start += size
    return bins


def ece_equal_mass(rows: list[dict], key: str, n_bins: int = 10) -> float | None:
    """Adaptive EQUAL-MASS Expected Calibration Error.

    Partition rows into ``n_bins`` bins of (nearly) equal COUNT after sorting by
    confidence, then return the standard L1 ECE:

        ECE = Σ_b (n_b / N) · | conf_b − acc_b |

    where ``conf_b`` is the mean confidence and ``acc_b`` the empirical accuracy in
    bin b. Equal-mass binning avoids the empty / sparse high-confidence bins that
    make equal-width ECE jumpy and bin-count-sensitive. Returns ``None`` on empty
    input."""
    if not rows:
        return None
    n = len(rows)
    e = 0.0
    for b in _equal_mass_bins(rows, key, n_bins):
        if not b:
            continue
        conf = sum(x[key] for x in b) / len(b)
        acc = sum(1 for x in b if x["correct"]) / len(b)
        e += len(b) / n * abs(conf - acc)
    return e


# ---------------------------------------------------------------- debiased CE
def ece_debiased(rows: list[dict], key: str, n_bins: int = 10) -> float | None:
    """Bias-corrected calibration error (Kumar, Liang, Ma, "Verified Uncertainty
    Calibration", NeurIPS 2019).

    The naive plug-in *squared* CE over bins,

        CE²_plugin = Σ_b p_b · ( conf_b − acc_b )² ,    p_b = n_b / N

    is upward-biased: ``acc_b`` estimates the bin's true accuracy with sampling
    noise, and that noise inflates the squared term even under perfect
    calibration. Kumar et al. give the debiased (unbiased-in-the-square) estimator
    by subtracting each bin's label-mean variance (their ``unbiased_square_ce``,
    reference impl github.com/p-lambda/verified_calibration):

        CE²_debiased = Σ_b p_b · [ ( conf_b − acc_b )²
                                   − acc_b·(1 − acc_b) / (n_b − 1) ]

    i.e. bin error  | conf_b − acc_b |²  minus the Bernoulli-variance correction
    term  acc_b·(1−acc_b)/(n_b−1)  for the empirical accuracy (the per-bin label
    mean). Bins with n_b < 2 contribute 0 (no variance estimate). Computed on
    EQUAL-MASS bins, as Kumar et al. recommend.

    We return the ROOT calibration error sqrt(CE²_debiased) so it lives on the same
    scale as the L1/L2 ECE numbers it's compared against. The debiased squared sum
    can go slightly NEGATIVE on a well-calibrated sample (the whole point — the
    correction removes the positive bias); we clamp at 0 before the square root.
    Returns ``None`` on empty input."""
    if not rows:
        return None
    n = len(rows)
    sq = 0.0
    for b in _equal_mass_bins(rows, key, n_bins):
        nb = len(b)
        if nb < 2:
            continue  # no variance estimate from a singleton bin
        conf = sum(x[key] for x in b) / nb
        acc = sum(1 for x in b if x["correct"]) / nb
        biased = (conf - acc) ** 2
        variance = acc * (1.0 - acc) / (nb - 1.0)
        sq += (nb / n) * (biased - variance)
    return math.sqrt(sq) if sq > 0.0 else 0.0


# ---------------------------------------------------------------- bin-count sweep
def ece_sweep(rows: list[dict], key: str, bin_counts: tuple[int, ...] = (5, 10, 15, 20)) -> dict:
    """For each bin count, report equal-width ECE (the shipped ``recalibrate.ece``)
    alongside equal-mass ECE. The point is to show the *ranking* of methods is
    bin-insensitive: if equal-width ECE wobbles across bin counts but equal-mass
    stays stable and the comparison's sign doesn't flip, the binning objection is
    answered. Returns ``{n_bins: {"binned_equal_width": ..., "equal_mass": ...}}``;
    values are ``None`` on empty input."""
    return {
        nb: {
            "binned_equal_width": ece(rows, key, n_bins=nb),
            "equal_mass": ece_equal_mass(rows, key, n_bins=nb),
        }
        for nb in bin_counts
    }


# ---------------------------------------------------------------- bootstrap CI
def bootstrap_ci(
    rows: list[dict],
    stat_fn: Callable[[list[dict]], float | None],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> dict:
    """Deterministic nonparametric percentile bootstrap CI for any row-level
    statistic.

    ``stat_fn`` is a callable that already closes over ``key`` — e.g.
    ``lambda r: ece_equal_mass(r, "cal")``. We resample ``rows`` WITH replacement
    ``n_resamples`` times using a seeded ``random.Random(seed)`` (so two calls with
    the same seed return identical bounds), drop any resample where ``stat_fn``
    returns ``None``, and read the percentile interval.

    Returns ``{"point": stat_fn(rows), "lo": <alpha/2 pct>, "hi": <1−alpha/2 pct>,
    "n_resamples": <# kept>}``. On empty input (or if every resample is dropped) the
    point and bounds are ``None``."""
    point = stat_fn(rows)
    if not rows:
        return {"point": point, "lo": None, "hi": None, "n_resamples": 0}
    rng = random.Random(seed)
    n = len(rows)
    stats: list[float] = []
    for _ in range(n_resamples):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        s = stat_fn(sample)
        if s is not None:
            stats.append(s)
    if not stats:
        return {"point": point, "lo": None, "hi": None, "n_resamples": 0}
    stats.sort()
    lo = _percentile(stats, alpha / 2.0)
    hi = _percentile(stats, 1.0 - alpha / 2.0)
    return {"point": point, "lo": lo, "hi": hi, "n_resamples": len(stats)}


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0,1]) over an already-sorted list."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo_i = int(math.floor(pos))
    hi_i = int(math.ceil(pos))
    if lo_i == hi_i:
        return sorted_vals[lo_i]
    frac = pos - lo_i
    return sorted_vals[lo_i] * (1.0 - frac) + sorted_vals[hi_i] * frac
