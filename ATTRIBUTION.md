# Attribution & reuse (honest accounting)

This project is a **recombination, not an invention**. The original seam is narrow: retargeting calibrated, abstaining routing onto the *local-vs-remote model* decision. The load-bearing machinery is reused, and that is the point.

## Reused verbatim

- **`src/router_agent/calibration/`** — the confidence-calibration package (Platt + isotonic fits, debiased ECE, nested-CV, the selective/abstain policy). Copied verbatim from the GAUGE project's `gauge.calibration`, which in turn copied it verbatim from PARALLAX's `parallax.calibration`. The **only** edit on each hop is the intra-package import path. Pure stdlib (no numpy/sklearn). MIT throughout.

## Mechanisms we build on (not ours)

- **Calibrated-abstaining cascade routing** — the "try-cheap, escalate-when-unsure, abstain-honestly" pattern is established (e.g. model-cascade routing à la UCCI). We apply it to its native problem (which model to call), with a calibrated confidence gate.
- **Self-consistency** — sampling a model multiple times and measuring answer agreement as a confidence signal is a known technique.
- **RouterBench** (`withmartian/routerbench`, MIT) — used only as a free, precomputed stand-in to dry-run the calibration plumbing offline. Catalogued + fetched, never rehosted.

## What is ours

- The retarget: wiring the calibration/abstention machinery into an N-tier, cost-ordered local↔remote cascade scored on remote tokens.
- The free-local-compute lever: spending unlimited (free) local samples on the *escalation decision* itself.
- The launch-day calibration protocol: label local-correctness, fit, pick the lowest threshold that clears the accuracy floor with margin.

No "first" or "novel" claims. Every number will state its `n` and held-out protocol.
