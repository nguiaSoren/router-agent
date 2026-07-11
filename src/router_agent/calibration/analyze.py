"""Calibration analysis — measure how trustworthy the judge's confidence is.

Given the predictions the judge already made (verdict + a 0-1 confidence) and the
human gold labels produced by the labeling page (agree / disagree / unsure), this
module answers one question per modality: *when the judge says it's this sure, is
it actually right that often?*

This is PARALLAX's "3-number credibility" check, run per modality (review vs
vision). The three headline numbers are:

  1. **accuracy**            — how often the human agreed with the judge,
  2. **ECE**                 — expected calibration error: the gap between the
                               judge's stated confidence and its empirical
                               accuracy, averaged over confidence bins,
  3. **diverges-precision**  — when we *flag a divergence*, how often it's real.

A prediction is **correct** iff the human AGREES with the judge's verdict.
`disagree` = incorrect. `unsure` is dropped from every metric (but counted and
reported, so the sample is never silently shrunk).

The suggested **promotion threshold** — the lowest confidence at/above which
empirical accuracy clears 0.9 — is what later flips a verdict's
``ConfidenceState`` from ``UNCALIBRATED_PREVIEW`` to ``CALIBRATED``.

Pure module: no file or network I/O. The only output is the synthetic
``__main__`` self-test below, which fabricates predictions+labels in memory.
"""

from __future__ import annotations

__all__ = ["join", "metrics", "verdict", "BIN_EDGES"]

# Reliability-diagram bins. The judge's confidences are always >= 0.5 (it reports
# confidence in the verdict it chose, not in the rejected one), so the lowest bin
# starts at 0.5. The top bin is closed on the right to include exactly 1.0.
BIN_EDGES: list[tuple[float, float]] = [
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
]

# Promotion / SHIP thresholds (kept as module constants so the report and verdict
# agree on the same numbers).
_PROMOTE_ACC = 0.9          # a bin "qualifies" for promotion at >= this accuracy
_MIN_BIN_N = 5              # below this a single bin is too small to trust
_SHIP_ECE = 0.1            # ship only if ECE is at or below this
_SHIP_DIV_PRECISION = 0.8   # ship only if a diverges-flag is real at least this often


def _bin_index(conf: float) -> int | None:
    """Return the index into ``BIN_EDGES`` for a confidence, or None if < 0.5.

    The top edge (1.0) lands in the last bin; values are clamped sensibly so a
    stray 1.0 or a tiny float overshoot does not fall off the end.
    """
    if conf < BIN_EDGES[0][0]:
        return None
    for i, (lo, hi) in enumerate(BIN_EDGES):
        # closed-left, open-right, except the final bin is closed-right
        if lo <= conf < hi:
            return i
        if i == len(BIN_EDGES) - 1 and conf >= lo:
            return i
    return None


def join(predictions: list[dict], gold: list[dict]) -> list[dict]:
    """Inner-join predictions to gold on ``item_id``; keep only graded rows.

    For each prediction that has a gold label of ``agree`` or ``disagree``, emit a
    compact row carrying just what the metrics need: ``judge_confidence``,
    ``modality``, ``judge_verdict``, and ``correct`` (agree -> True, disagree ->
    False). Rows labeled ``unsure`` are dropped here; the caller learns how many
    were dropped by diffing against the gold count (the report does this).

    A prediction with no gold entry, or a gold entry with no matching prediction,
    is silently skipped (that's what an inner join means).
    """
    # Map item_id -> normalized human label. Later duplicates overwrite earlier
    # ones (last write wins) — a labeling page that lets you revise a card should
    # have already de-duplicated, but we don't crash if it didn't.
    label_by_id: dict[str, str] = {}
    for g in gold:
        gid = g.get("item_id")
        if gid is None:
            continue
        label = str(g.get("label", "")).strip().lower()
        label_by_id[gid] = label

    joined: list[dict] = []
    for p in predictions:
        pid = p.get("item_id")
        if pid is None or pid not in label_by_id:
            continue
        label = label_by_id[pid]
        if label == "agree":
            correct = True
        elif label == "disagree":
            correct = False
        else:
            # "unsure" (or any non-binary label) is excluded from metrics.
            continue
        joined.append(
            {
                "item_id": pid,
                "judge_confidence": float(p.get("judge_confidence", 0.0)),
                "modality": p.get("modality", "unknown"),
                "judge_verdict": p.get("judge_verdict", "unknown"),
                "correct": correct,
            }
        )
    return joined


def _reliability_bins(rows: list[dict]) -> list[dict]:
    """Bin rows by confidence; per bin report n, mean confidence, empirical acc.

    Every bin in ``BIN_EDGES`` is emitted even when empty (n=0), so the reliability
    diagram has a stable shape. ``low_n`` flags a bin with fewer than _MIN_BIN_N
    graded items — its accuracy is too noisy to lean on.
    """
    buckets: list[list[dict]] = [[] for _ in BIN_EDGES]
    for r in rows:
        idx = _bin_index(r["judge_confidence"])
        if idx is not None:
            buckets[idx].append(r)

    out: list[dict] = []
    for (lo, hi), bucket in zip(BIN_EDGES, buckets):
        n = len(bucket)
        if n:
            mean_conf = sum(r["judge_confidence"] for r in bucket) / n
            acc = sum(1 for r in bucket if r["correct"]) / n
        else:
            mean_conf = None
            acc = None
        out.append(
            {
                "lo": lo,
                "hi": hi,
                "n": n,
                "mean_confidence": mean_conf,
                "accuracy": acc,
                "low_n": 0 < n < _MIN_BIN_N,
            }
        )
    return out


def _ece(bins: list[dict], total: int) -> float | None:
    """Expected calibration error over the confidence bins.

    ECE = sum_b (n_b / N) * |mean_conf_b - accuracy_b|, summing only non-empty
    bins. Returns None when there is nothing graded.
    """
    if total <= 0:
        return None
    err = 0.0
    for b in bins:
        if b["n"] and b["mean_confidence"] is not None and b["accuracy"] is not None:
            err += (b["n"] / total) * abs(b["mean_confidence"] - b["accuracy"])
    return err


def _promotion_threshold(bins: list[dict]) -> dict:
    """Lowest confidence at/above which empirical accuracy clears _PROMOTE_ACC.

    Walks the bins low->high and takes the first one that both has enough data
    (>= _MIN_BIN_N) and reaches the accuracy bar. Once a qualifying bin is found
    we require every *higher* bin that has data to also clear the bar — otherwise
    the curve isn't actually monotone-enough to trust a single floor, and we say
    so rather than promoting on a lucky low bin.

    Returns a dict with ``threshold`` (the bin's lower edge) or ``threshold=None``
    plus a human ``reason`` when no bin qualifies / data is too thin.
    """
    graded = sum(b["n"] for b in bins)
    if graded < _MIN_BIN_N:
        return {"threshold": None, "reason": "insufficient data (too few graded items)"}

    candidate: dict | None = None
    for b in bins:
        if b["n"] >= _MIN_BIN_N and b["accuracy"] is not None and b["accuracy"] >= _PROMOTE_ACC:
            candidate = b
            break

    if candidate is None:
        return {
            "threshold": None,
            "reason": "insufficient data — no confidence bin (n>=%d) reaches %.0f%% accuracy"
            % (_MIN_BIN_N, _PROMOTE_ACC * 100),
        }

    # Require higher bins with real data not to dip back below the bar.
    for b in bins:
        if b["lo"] > candidate["lo"] and b["n"] >= _MIN_BIN_N and b["accuracy"] is not None:
            if b["accuracy"] < _PROMOTE_ACC:
                return {
                    "threshold": None,
                    "reason": "insufficient data — accuracy is non-monotone above %.2f "
                    "(bin %.1f–%.1f drops to %.0f%%)"
                    % (candidate["lo"], b["lo"], b["hi"], b["accuracy"] * 100),
                }

    return {
        "threshold": candidate["lo"],
        "reason": "empirical accuracy >= %.0f%% at confidence >= %.2f (bin n=%d)"
        % (_PROMOTE_ACC * 100, candidate["lo"], candidate["n"]),
    }


def _group_metrics(rows: list[dict]) -> dict:
    """All per-group numbers for one slice of joined rows."""
    n = len(rows)
    accuracy = (sum(1 for r in rows if r["correct"]) / n) if n else None
    bins = _reliability_bins(rows)
    return {
        "n": n,
        "accuracy": accuracy,
        "reliability_bins": bins,
        "ece": _ece(bins, n),
        "promotion_threshold": _promotion_threshold(bins),
    }


def _diverges_precision(rows: list[dict]) -> dict:
    """Precision of a 'diverges' flag: among predicted-diverges, fraction correct.

    This is the product-facing headline — "when we flag a divergence, it's real
    X%." Returns ``precision=None`` when the judge never flagged a divergence in
    this slice (no flags -> precision is undefined, not 0 or 1).
    """
    flagged = [r for r in rows if r["judge_verdict"] == "diverges"]
    n = len(flagged)
    precision = (sum(1 for r in flagged if r["correct"]) / n) if n else None
    return {"n_flagged": n, "precision": precision}


def metrics(joined: list[dict]) -> dict:
    """Compute calibration metrics overall, per modality, and per verdict.

    Structure::

        {
          "n_total": int,
          "overall": {n, accuracy, reliability_bins, ece, promotion_threshold,
                      diverges_precision},
          "by_modality": {"review": {...}, "vision": {...}, ...},
          "by_verdict":  {"holds": {...}, "diverges": {...}},
        }

    Each leaf carries enough to draw a reliability diagram and judge whether the
    confidence is trustworthy. ``by_verdict`` additionally exposes that the
    ``diverges`` slice IS the flag-precision population.
    """
    overall = _group_metrics(joined)
    overall["diverges_precision"] = _diverges_precision(joined)

    by_modality: dict[str, dict] = {}
    for mod in sorted({r["modality"] for r in joined}):
        sub = [r for r in joined if r["modality"] == mod]
        g = _group_metrics(sub)
        g["diverges_precision"] = _diverges_precision(sub)
        by_modality[mod] = g

    by_verdict: dict[str, dict] = {}
    for verd in ("holds", "diverges"):
        sub = [r for r in joined if r["judge_verdict"] == verd]
        g = _group_metrics(sub)
        g["diverges_precision"] = _diverges_precision(sub)
        by_verdict[verd] = g

    return {
        "n_total": len(joined),
        "overall": overall,
        "by_modality": by_modality,
        "by_verdict": by_verdict,
    }


def verdict(metrics: dict, *, min_n: int = 20) -> dict:
    """SHIP / REFINE / INSUFFICIENT-DATA call on the calibration.

    Rule:
      * **INSUFFICIENT-DATA** if overall n < ``min_n`` — never claim calibration
        on a sample too small to mean anything.
      * **SHIP** if n >= min_n AND ECE <= 0.1 AND diverges-precision >= 0.8 — the
        confidence tracks reality and a flag is usually real.
      * **REFINE** otherwise — enough data, but the numbers don't clear the bar.

    A ``diverges_precision`` of None (the judge never flagged anything) cannot
    clear the SHIP bar, so it falls through to REFINE — we won't ship a divergence
    detector that never fired on the calibration set.
    """
    overall = metrics.get("overall", {})
    n = overall.get("n", 0)
    ece = overall.get("ece")
    div = overall.get("diverges_precision", {}) or {}
    div_precision = div.get("precision")

    if n < min_n:
        return {
            "verdict": "INSUFFICIENT-DATA",
            "n": n,
            "min_n": min_n,
            "ece": ece,
            "diverges_precision": div_precision,
            "summary": "INSUFFICIENT-DATA: only %d graded predictions (need >=%d) — "
            "label more before trusting the calibration." % (n, min_n),
        }

    ship = (
        ece is not None
        and ece <= _SHIP_ECE
        and div_precision is not None
        and div_precision >= _SHIP_DIV_PRECISION
    )
    call = "SHIP" if ship else "REFINE"

    div_str = "n/a" if div_precision is None else "%.0f%%" % (div_precision * 100)
    ece_str = "n/a" if ece is None else "%.3f" % ece
    if call == "SHIP":
        summary = (
            "SHIP: n=%d, ECE=%s (<=%.2f), diverges-precision=%s (>=%.0f%%) — "
            "confidence is calibrated; promote." % (n, ece_str, _SHIP_ECE, div_str, _SHIP_DIV_PRECISION * 100)
        )
    else:
        reasons = []
        if ece is None or ece > _SHIP_ECE:
            reasons.append("ECE=%s > %.2f" % (ece_str, _SHIP_ECE))
        if div_precision is None or div_precision < _SHIP_DIV_PRECISION:
            reasons.append("diverges-precision=%s < %.0f%%" % (div_str, _SHIP_DIV_PRECISION * 100))
        summary = "REFINE: n=%d but %s — keep preview/uncalibrated." % (n, "; ".join(reasons))

    return {
        "verdict": call,
        "n": n,
        "min_n": min_n,
        "ece": ece,
        "diverges_precision": div_precision,
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
# Offline self-test: synthetic predictions + labels, no files touched.
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    import random

    random.seed(7)

    # Build a synthetic set where the judge is *well calibrated*: a prediction at
    # confidence c is correct with probability ~ c. This should yield low ECE and,
    # at high confidence, accuracy clearing the promotion bar.
    preds: list[dict] = []
    gold: list[dict] = []
    n_items = 240
    for i in range(n_items):
        modality = "review" if i % 2 == 0 else "vision"
        conf = round(random.uniform(0.5, 1.0), 3)
        verd = "diverges" if random.random() < 0.4 else "holds"
        # human agrees (judge correct) with probability ~= conf
        roll = random.random()
        if roll < conf:
            label = "agree"
        elif roll < conf + 0.08:
            label = "unsure"  # a slice the human couldn't call
        else:
            label = "disagree"
        item_id = "item-%03d" % i
        preds.append(
            {
                "item_id": item_id,
                "modality": modality,
                "attribute": "sofa_bed" if modality == "review" else "sea_view",
                "judge_verdict": verd,
                "judge_confidence": conf,
                "claimed": True,
            }
        )
        gold.append({"item_id": item_id, "label": label})

    # A prediction with no gold (should be dropped) and gold with no prediction.
    preds.append(
        {
            "item_id": "orphan-pred",
            "modality": "review",
            "attribute": "x",
            "judge_verdict": "holds",
            "judge_confidence": 0.9,
            "claimed": False,
        }
    )
    gold.append({"item_id": "orphan-gold", "label": "agree"})

    joined = join(preds, gold)
    n_unsure = sum(1 for g in gold if g["label"] == "unsure")

    # --- assertions: prove the join + metric plumbing ---
    assert all(set(r) == {"item_id", "judge_confidence", "modality", "judge_verdict", "correct"} for r in joined)
    assert not any(r["item_id"] in ("orphan-pred", "orphan-gold") for r in joined), "orphans must be dropped"
    expected_kept = sum(1 for g in gold if g["label"] in ("agree", "disagree") and g["item_id"] != "orphan-gold")
    assert len(joined) == expected_kept, (len(joined), expected_kept)

    m = metrics(joined)
    assert m["n_total"] == len(joined)
    assert set(m["by_verdict"]) == {"holds", "diverges"}
    # bin n's sum back to the group n (every graded row lands in exactly one bin)
    for grp in [m["overall"], *m["by_modality"].values(), *m["by_verdict"].values()]:
        assert sum(b["n"] for b in grp["reliability_bins"]) == grp["n"], grp["n"]
        if grp["n"]:
            assert 0.0 <= grp["accuracy"] <= 1.0
            assert grp["ece"] is None or grp["ece"] >= 0.0

    v = verdict(m, min_n=20)

    # --- report ---
    print("=== analyze.py synthetic self-test ===")
    print("graded (after dropping unsure/orphans): %d   unsure dropped: %d" % (m["n_total"], n_unsure))
    o = m["overall"]
    print(
        "OVERALL  n=%d  acc=%.3f  ECE=%.3f  diverges-precision=%s (n_flagged=%d)"
        % (
            o["n"],
            o["accuracy"],
            o["ece"],
            "%.3f" % o["diverges_precision"]["precision"] if o["diverges_precision"]["precision"] is not None else "n/a",
            o["diverges_precision"]["n_flagged"],
        )
    )
    print("  reliability bins (lo–hi : n  mean_conf  emp_acc  [flags]):")
    for b in o["reliability_bins"]:
        mc = "  -  " if b["mean_confidence"] is None else "%.3f" % b["mean_confidence"]
        ac = "  -  " if b["accuracy"] is None else "%.3f" % b["accuracy"]
        flag = " low-n(<5)" if b["low_n"] else ""
        print("    %.1f–%.1f : n=%-3d  conf=%s  acc=%s%s" % (b["lo"], b["hi"], b["n"], mc, ac, flag))
    pt = o["promotion_threshold"]
    print("  promotion threshold: %s  (%s)" % (pt["threshold"], pt["reason"]))
    for mod, g in m["by_modality"].items():
        dp = g["diverges_precision"]["precision"]
        print(
            "  modality %-7s n=%-3d acc=%.3f ECE=%s div-prec=%s"
            % (
                mod,
                g["n"],
                g["accuracy"],
                "%.3f" % g["ece"] if g["ece"] is not None else "n/a",
                "%.3f" % dp if dp is not None else "n/a",
            )
        )
    print("VERDICT:", v["summary"])

    # The calibrated synthetic set should land in a sane place.
    assert v["verdict"] in ("SHIP", "REFINE"), v
    assert o["ece"] < 0.15, "calibrated synthetic data should have small ECE, got %s" % o["ece"]

    # An empty / tiny join must report INSUFFICIENT-DATA.
    tiny = verdict(metrics(joined[:3]), min_n=20)
    assert tiny["verdict"] == "INSUFFICIENT-DATA", tiny
    print("tiny-sample check:", tiny["summary"])
    print("=== self-test OK ===")


if __name__ == "__main__":
    _selftest()
