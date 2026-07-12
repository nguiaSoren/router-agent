"""Apply the fitted calibration model at inference — remap a judge's RAW confidence
to its validated value and report whether that modality is calibrated.

Loaded once from ``data/calibration/calibration_model.json`` (written by
``scripts/recalibrate.py``). Only modalities the held-out validation actually
recalibrated carry a map; everything else is returned RAW with is_calibrated=False
(so an un-validated modality is never silently "promoted").

Important: profiles/predictions always store RAW confidence — calibration is applied
only here, at the output edge — so the next calibration round still fits raw→empirical
and there's no feedback drift.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .recalibrate import calibrate as _apply_model

_PATH = Path(__file__).resolve().parents[3] / "data" / "calibration" / "calibration_model.json"


@lru_cache(maxsize=1)
def _data() -> dict:
    try:
        return json.loads(_PATH.read_text())
    except Exception:  # noqa: BLE001 — no model yet → everything stays raw/preview
        return {}


def calibrate_confidence(raw: float, modality: str) -> tuple[float, bool]:
    """(calibrated_confidence, is_calibrated). is_calibrated is True only when the
    validated model holds a map for ``modality``; otherwise (raw, False)."""
    model = _data().get("model", {})
    if modality in model and model[modality]:
        return float(_apply_model(model, raw, modality)), True
    return float(raw), False
