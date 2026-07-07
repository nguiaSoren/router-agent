"""Nested cross-validation for confidence calibration — the honest, non-circular
estimate of how the *calibrator-selection procedure* generalizes to unseen points.

WHY THIS EXISTS (the circularity it fixes): ``recalibrate.selective()`` picks each
modality's method (isotonic vs Platt vs raw/identity) by scoring inner LOO-ECE over
ALL points, and then reports ``effective_loo_ece`` on those SAME points. The method
choice for scoring point i therefore SEES point i — an EACL reviewer reads that as
circular/optimistic. Nested CV makes the method choice for point i WITHOUT ever
seeing point i: an OUTER leave-one-out loop holds out i, an INNER selection (the
faithfully-replicated ``selective()`` 3-branch policy) runs on the other N-1 rows,
the chosen method is fit on those N-1, and only then is it applied to held-out i.

_caveat: Nested CV estimates the ECE of the SELECTION PROCEDURE over unseen points,
not the ECE of a single fixed deployed calibrator.

Reuses the fitting primitives from ``recalibrate`` (``fit_isotonic``, ``fit_platt``,
``apply_cal``, ``ece``, ``_loo_method``) — nothing is reimplemented here.
"""

from __future__ import annotations

from router_agent.calibration.recalibrate import (
    _loo_method,
    apply_cal,
    ece,
    fit_isotonic,
    fit_platt,
)

_CAVEAT = (
    "Nested CV estimates the ECE of the SELECTION PROCEDURE over unseen points, "
    "not the ECE of a single fixed deployed calibrator."
)


def _select_method(inner: list[dict], *, threshold: float, platt_pref: float) -> str:
    """Replicate ``recalibrate.selective()``'s per-modality 3-branch policy on the
    INNER set ONLY, returning the chosen method name: "isotonic" | "platt" | "identity".

    Faithful mirror of selective():
      - inner raw-ECE     = ece(inner judge_confidence vs correct)
      - inner isotonic    = ece(_loo_method(inner, fit_isotonic), key="cal")
      - inner Platt       = ece(_loo_method(inner, fit_platt),    key="cal")
      - choose Platt unless isotonic beats it by MORE than platt_pref
      - branch 1: if best LOO-ECE improves over inner raw-ECE -> use best (iso/platt)
      - branch 2: elif inner raw-ECE <= threshold            -> identity (raw, no remap)
      - branch 3: else                                       -> keep raw (identity)
    """
    raw_ece = ece(
        [{"cal": r["judge_confidence"], "correct": r["correct"]} for r in inner],
        key="cal",
    )
    iso_e = ece(_loo_method(inner, fit_isotonic), key="cal")
    platt_e = ece(_loo_method(inner, fit_platt), key="cal")

    # choose: Platt unless isotonic is clearly better (by more than platt_pref)
    if platt_e is not None and (iso_e is None or platt_e <= iso_e + platt_pref):
        best_e, best_name = platt_e, "platt"
    else:
        best_e, best_name = iso_e, "isotonic"

    improves = best_e is not None and raw_ece is not None and best_e < raw_ece
    if improves:
        return best_name  # branch 1: chosen remap beats raw
    if raw_ece is not None and raw_ece <= threshold:
        return "identity"  # branch 2: raw already clears threshold
    return "identity"  # branch 3: keep raw (no remap helped)


def _held_out_cal(inner: list[dict], method: str, x: float) -> float:
    """Fit ``method`` on the FULL inner set and apply it to held-out raw conf ``x``.
    identity -> cal = raw (no remap)."""
    if method == "isotonic":
        return apply_cal(fit_isotonic(inner), x)
    if method == "platt":
        return apply_cal(fit_platt(inner), x)
    return x  # identity


def nested_cv(
    joined: list[dict],
    *,
    by_modality: bool = True,
    threshold: float = 0.10,
    platt_pref: float = 0.005,
) -> dict:
    """Nested cross-validated calibration estimate — non-circular by construction.

    Per modality (or all rows as one group when ``by_modality`` is False), run an
    OUTER leave-one-out loop. n is small (e.g. 42 review / 18 vision) so outer LOO is
    correct. For each held-out index i:

      - inner = all OTHER rows of that modality (row i is absent by construction).
      - SELECT the method on ``inner`` ONLY via the replicated selective() 3-branch
        policy (see ``_select_method``).
      - FIT the chosen method on the FULL inner set and apply it to row i's
        ``judge_confidence`` -> held-out ``cal`` (identity -> cal == raw).
      - record the held-out pair {cal, raw, correct, modality, chosen_method}.

    Returns::

        {
          "by_modality": {m: {"n", "pairs", "method_counts", "ece"}},
          "pairs": [all pairs across modalities],
          "_caveat": <see module docstring>,
        }

    The per-modality ``ece`` is a convenience binned sanity number over the held-out
    pairs; the reproduce script scores the pairs downstream — final headline ECEs are
    NOT computed here.

    _caveat: Nested CV estimates the ECE of the SELECTION PROCEDURE over unseen
    points, not the ECE of a single fixed deployed calibrator.
    """
    if by_modality:
        mods = sorted({r["modality"] for r in joined})
        groups = {m: [r for r in joined if r["modality"] == m] for m in mods}
    else:
        groups = {"_all": list(joined)}

    by_mod: dict[str, dict] = {}
    all_pairs: list[dict] = []

    for m, rows in groups.items():
        pairs: list[dict] = []
        method_counts: dict[str, int] = {"isotonic": 0, "platt": 0, "identity": 0}
        for i, row_i in enumerate(rows):
            inner = [r for j, r in enumerate(rows) if j != i]
            method = _select_method(inner, threshold=threshold, platt_pref=platt_pref)
            cal = _held_out_cal(inner, method, row_i["judge_confidence"])
            pair = {
                "cal": cal,
                "raw": row_i["judge_confidence"],
                "correct": row_i["correct"],
                "modality": row_i["modality"],
                "chosen_method": method,
            }
            method_counts[method] += 1
            pairs.append(pair)
        all_pairs.extend(pairs)
        by_mod[m] = {
            "n": len(rows),
            "pairs": pairs,
            "method_counts": method_counts,
            "ece": ece(pairs, key="cal"),  # convenience binned sanity number
        }

    return {"by_modality": by_mod, "pairs": all_pairs, "_caveat": _CAVEAT}


if __name__ == "__main__":  # offline self-test, no files
    spread = [
        {"judge_confidence": round(0.50 + 0.45 * (i / 59), 3), "correct": (i % 4 != 0), "modality": "review"}
        for i in range(60)
    ]
    res = nested_cv(spread, by_modality=True)
    rmod = res["by_modality"]["review"]
    print(f"nested CV review n={rmod['n']} ece={rmod['ece']:.3f} counts={rmod['method_counts']}")
    print(res["_caveat"])
