# Acknowledgments

TokenGolf builds on a few established ideas and one open dataset, credited honestly:

- **Calibrated, abstaining cascade routing** — the "try the cheap option, escalate when unsure, abstain honestly" pattern for model cascades is established prior art (e.g. UCCI-style uncertainty-calibrated cascade routing). TokenGolf applies it to the local-vs-remote decision through a calibrated confidence gate.
- **Self-consistency** — sampling a model several times and measuring answer agreement as a confidence signal is a known technique; here it runs on the free local tier, so the escalation decision costs no scored tokens.
- **RouterBench** (`withmartian/routerbench`, MIT) — used only as a free, precomputed dataset to dry-run the calibration plumbing offline. Catalogued and fetched, never rehosted.

The confidence-calibration package (`src/router_agent/calibration/`) is pure stdlib (Platt + isotonic fits, debiased ECE per Kumar et al. 2019, nested cross-validation, a selective/abstain policy), MIT.

## What TokenGolf does

- Wires the calibration/abstention machinery into a binary, cost-ordered local↔remote cascade scored on remote tokens.
- Spends unlimited (free) local samples on the *escalation decision* itself.
- Disables the Fireworks reasoning models' hidden reasoning (`reasoning_effort="none"`), the dominant cost on a token-ranked leaderboard.
- Calibration protocol: label local-correctness, fit, pick the lowest threshold that clears the accuracy floor with margin.

No "first" or "novel" claims. Every reported number states its `n` and held-out protocol.
