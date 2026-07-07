"""Confidence recalibration — map the judge's RAW confidence to its empirical
accuracy, so a stated "0.85" actually means 85% right.

The hard gold batch showed the raw confidence is non-monotone (more accurate at
0.6–0.7 than 0.8–0.9), so we fit an **isotonic** map (Pool-Adjacent-Violators) —
it produces the monotone calibration that best fits the data, pooling the
overconfident dip. Fitted per modality, since review and vision differ.

HONESTY: fitting *and* scoring on the same points reports a fake ~0 ECE
(isotonic interpolates its own training data). So the headline recalibrated ECE
is **leave-one-out cross-validated** — fit on the other N-1, predict the held-out
one. That's the number we'd actually see on unseen listings.

Pure Python (no numpy/sklearn). The fitted model serializes to plain breakpoints.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------- isotonic (PAV)
def fit_map(rows: list[dict]) -> list[list[float]]:
    """Fit an isotonic calibration map from rows ({judge_confidence, correct}).

    Isotonic regression WITH TIES, done correctly: group rows by confidence value,
    take each group's empirical accuracy (its mean) weighted by group size, then run
    weighted Pool-Adjacent-Violators on those per-confidence means. (Running PAV on
    the raw interleaved 0/1 labels mis-pools across confidence boundaries — it must be
    grouped first.) Returns monotone breakpoints [[x, y], ...]: raw confidence → the
    empirical accuracy actually observed at/around it."""
    groups: dict[float, list[float]] = {}  # x -> [sum_correct, count]
    for r in rows:
        x = r["judge_confidence"]
        g = groups.setdefault(x, [0.0, 0.0])
        g[0] += 1.0 if r["correct"] else 0.0
        g[1] += 1.0
    xs = sorted(groups)
    if not xs:
        return []
    # blocks over the per-x means: [weighted_sum_correct, total_weight, n_x_groups]
    blocks: list[list[float]] = [[groups[x][0], groups[x][1], 1.0] for x in xs]
    k = 0
    while k < len(blocks) - 1:
        if blocks[k][0] / blocks[k][1] > blocks[k + 1][0] / blocks[k + 1][1] + 1e-12:
            blocks[k][0] += blocks[k + 1][0]
            blocks[k][1] += blocks[k + 1][1]
            blocks[k][2] += blocks[k + 1][2]
            del blocks[k + 1]
            if k > 0:
                k -= 1
        else:
            k += 1
    pooled: list[float] = []
    for csum, w, ng in blocks:
        pooled.extend([csum / w] * int(ng))
    return [[xs[i], pooled[i]] for i in range(len(xs))]


def apply_map(bp: list[list[float]], x: float) -> float:
    """Calibrated confidence for raw ``x`` via linear interpolation over the
    breakpoints (clamped at the ends). Identity if the map is empty."""
    if not bp:
        return x
    if x <= bp[0][0]:
        return bp[0][1]
    if x >= bp[-1][0]:
        return bp[-1][1]
    for i in range(len(bp) - 1):
        x0, y0 = bp[i]
        x1, y1 = bp[i + 1]
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return bp[-1][1]


# ---------------------------------------------------------------- Platt (logistic)
def _sigmoid(z: float) -> float:
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def fit_platt(rows: list[dict], *, iters: int = 600, lr: float = 0.3, l2: float = 0.02) -> dict:
    """Platt scaling — a smooth monotone logistic fit P(correct)=sigmoid(a·conf+b).
    Unlike isotonic it PRESERVES the raw ordering (so 0.71/0.85/0.95 map to *distinct*
    calibrated values, restoring high-end discrimination isotonic pools away) and never
    saturates to exactly 1.0. Mild L2 keeps the slope from blowing up on clean data."""
    xs = [r["judge_confidence"] for r in rows]
    ys = [1.0 if r["correct"] else 0.0 for r in rows]
    n = len(xs)
    if n == 0:
        return {"type": "platt", "params": [0.0, 0.0]}
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga, gb = l2 * a, 0.0
        for x, y in zip(xs, ys):
            p = _sigmoid(a * x + b)
            ga += (p - y) * x
            gb += (p - y)
        a -= lr * ga / n
        b -= lr * gb / n
    return {"type": "platt", "params": [a, b]}


def fit_isotonic(rows: list[dict]) -> dict:
    return {"type": "isotonic", "params": fit_map(rows)}


def apply_cal(cal, x: float) -> float:
    """Apply a calibrator. Handles a calibrator dict ({type,params}), a bare isotonic
    breakpoint list (legacy), or None/identity → raw."""
    if not cal:
        return x
    if isinstance(cal, list):  # legacy bare isotonic breakpoints
        return apply_map(cal, x)
    t = cal.get("type")
    if t == "isotonic":
        return apply_map(cal.get("params") or [], x)
    if t == "platt":
        a, b = cal["params"]
        return _sigmoid(a * x + b)
    return x  # identity / unknown


# ---------------------------------------------------------------- model + ECE
def fit_model(joined: list[dict], *, by_modality: bool = True) -> dict:
    """Full-data ISOTONIC model (bare breakpoints) — used by report()/self-test.
    The shippable per-modality model is chosen by selective() (isotonic vs Platt)."""
    model = {"_global": fit_map(joined)}
    if by_modality:
        mods = {r["modality"] for r in joined}
        for m in mods:
            rows = [r for r in joined if r["modality"] == m]
            if len(rows) >= 8:
                model[m] = fit_map(rows)
    return model


def calibrate(model: dict, raw_conf: float, modality: str) -> float:
    """Apply the model's calibrator for ``modality`` (dict or legacy list), else raw."""
    return apply_cal(model.get(modality), raw_conf)


def ece(items: list[dict], key: str, n_bins: int = 10) -> float | None:
    """Expected Calibration Error over [0,1] (calibrated confidences can fall below
    0.5, so we bin the full range, unlike the 0.5–1.0 report bins)."""
    if not items:
        return None
    bins: list[list[dict]] = [[] for _ in range(n_bins)]
    for it in items:
        b = min(n_bins - 1, max(0, int(it[key] * n_bins)))
        bins[b].append(it)
    n = len(items)
    e = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(x[key] for x in b) / len(b)
        acc = sum(1 for x in b if x["correct"]) / len(b)
        e += len(b) / n * abs(conf - acc)
    return e


# ---------------------------------------------------------------- LOO + report
def loo_calibrated(joined: list[dict], *, by_modality: bool = True) -> list[dict]:
    """Leave-one-out calibrated confidences — fit on the other N-1 (same modality
    when it has support, else global), predict the held-out one. The honest score."""
    out: list[dict] = []
    for i, r in enumerate(joined):
        if by_modality:
            rest = [x for j, x in enumerate(joined) if j != i and x["modality"] == r["modality"]]
            if len(rest) < 5:  # too thin → global fallback
                rest = [x for j, x in enumerate(joined) if j != i]
        else:
            rest = [x for j, x in enumerate(joined) if j != i]
        cal = apply_map(fit_map(rest), r["judge_confidence"])
        out.append({"cal": cal, "raw": r["judge_confidence"], "correct": r["correct"], "modality": r["modality"]})
    return out


def report(joined: list[dict], *, by_modality: bool = True) -> dict:
    """Raw vs LOO-calibrated ECE, overall and per modality."""
    raw_rows = [{"v": r["judge_confidence"], "correct": r["correct"], "modality": r["modality"]} for r in joined]
    raw_ece = ece([{**r, "raw": r["v"]} for r in raw_rows], key="raw")

    loo = loo_calibrated(joined, by_modality=by_modality)
    cal_ece = ece(loo, key="cal")

    per_modality = {}
    for m in sorted({r["modality"] for r in joined}):
        rraw = [r for r in raw_rows if r["modality"] == m]
        rcal = [r for r in loo if r["modality"] == m]
        per_modality[m] = {
            "n": len(rcal),
            "raw_ece": ece([{**r, "raw": r["v"]} for r in rraw], key="raw"),
            "cal_ece": ece(rcal, key="cal"),
        }
    return {
        "n": len(joined),
        "raw_ece": raw_ece,
        "calibrated_loo_ece": cal_ece,
        "improved": (cal_ece is not None and raw_ece is not None and cal_ece < raw_ece),
        "by_modality": per_modality,
    }


def _loo_method(rows: list[dict], fit_fn) -> list[dict]:
    """Leave-one-out calibrated values for one modality's rows under a given fit fn."""
    out = []
    for i, r in enumerate(rows):
        rest = [x for j, x in enumerate(rows) if j != i]
        out.append({"cal": apply_cal(fit_fn(rest), r["judge_confidence"]), "correct": r["correct"]})
    return out


def selective(joined: list[dict], *, threshold: float = 0.10, platt_pref: float = 0.005) -> dict:
    """Per modality, fit BOTH isotonic and Platt, score each by leave-one-out ECE, and
    pick the better — preferring Platt on a near-tie because it preserves high-end
    discrimination and never saturates (the isotonic-only map collapsed review's top
    band to 1.0). A modality is CALIBRATED if the chosen method beats raw and clears
    ``threshold``, OR if raw already clears it (identity, no remap). Else: preview.

    Returns {recalibrate, calibrated, chosen, model, effective_loo_ece}."""
    mods = sorted({r["modality"] for r in joined})
    recal: dict[str, bool] = {}
    calibrated: dict[str, bool] = {}
    chosen: dict[str, dict] = {}
    model: dict[str, dict] = {}
    eff: list[dict] = []
    for m in mods:
        rows = [r for r in joined if r["modality"] == m]
        raw_ece = ece([{"cal": r["judge_confidence"], "correct": r["correct"]} for r in rows], key="cal")
        iso_e = ece(_loo_method(rows, fit_isotonic), key="cal")
        platt_e = ece(_loo_method(rows, fit_platt), key="cal")
        # choose: Platt unless isotonic is clearly better (by more than platt_pref)
        if platt_e is not None and (iso_e is None or platt_e <= iso_e + platt_pref):
            best_e, best_fit, best_name = platt_e, fit_platt, "platt"
        else:
            best_e, best_fit, best_name = iso_e, fit_isotonic, "isotonic"

        improves = best_e is not None and raw_ece is not None and best_e < raw_ece
        recal[m] = improves
        if improves:
            model[m] = best_fit(rows)
            calibrated[m] = best_e <= threshold
            chosen[m] = {"method": best_name, "loo_ece": best_e, "raw_ece": raw_ece, "iso_ece": iso_e, "platt_ece": platt_e}
            eff += [{"cal": r["cal"], "correct": r["correct"]} for r in _loo_method(rows, best_fit)]
        elif raw_ece is not None and raw_ece <= threshold:
            model[m] = {"type": "identity", "params": None}
            calibrated[m] = True
            chosen[m] = {"method": "raw", "loo_ece": raw_ece, "raw_ece": raw_ece, "iso_ece": iso_e, "platt_ece": platt_e}
            eff += [{"cal": r["judge_confidence"], "correct": r["correct"]} for r in rows]
        else:
            calibrated[m] = False
            chosen[m] = {"method": "none", "loo_ece": best_e, "raw_ece": raw_ece, "iso_ece": iso_e, "platt_ece": platt_e}
            eff += [{"cal": r["judge_confidence"], "correct": r["correct"]} for r in rows]

    return {"recalibrate": recal, "calibrated": calibrated, "chosen": chosen,
            "model": model, "effective_loo_ece": ece(eff, key="cal")}


if __name__ == "__main__":  # offline self-test, no files
    # 1) MAP INVARIANT — an overconfident bin must be pulled DOWN; map stays monotone.
    over = [
        *[{"judge_confidence": 0.62, "correct": True} for _ in range(18)],   # 0.62 -> 90%
        *[{"judge_confidence": 0.62, "correct": False} for _ in range(2)],
        *[{"judge_confidence": 0.85, "correct": True} for _ in range(6)],    # 0.85 -> 60% (overconfident)
        *[{"judge_confidence": 0.85, "correct": False} for _ in range(4)],
    ]
    bp = fit_map(over)
    ys = [y for _, y in bp]
    assert ys == sorted(ys), f"map must be monotone non-decreasing: {bp}"
    assert apply_map(bp, 0.85) < 0.85, "overconfident 0.85 must calibrate downward"
    print(f"map invariant OK — 0.85 -> {apply_map(bp, 0.85):.2f} (pulled down), monotone")

    # 2) LOO IMPROVEMENT — spread confidences but a FLAT true accuracy (~0.75): every
    #    stated confidence is wrong, so recalibration collapses them and ECE drops.
    spread = [
        {"judge_confidence": round(0.50 + 0.45 * (i / 59), 3), "correct": (i % 4 != 0), "modality": "review"}
        for i in range(60)
    ]
    rep = report(spread, by_modality=False)
    print(f"raw ECE {rep['raw_ece']:.3f}  ->  calibrated LOO ECE {rep['calibrated_loo_ece']:.3f}  improved={rep['improved']}")
    assert rep["improved"], "recalibration should reduce ECE on a spread miscalibrated set"
    print("self-test passed")
