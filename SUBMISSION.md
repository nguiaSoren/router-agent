# lablab submission: draft fields (Track 1)

Copy-paste into the lablab.ai submission form. Update the image/repo URLs once pushed.

## Title
TokenGolf

## Short description (224 chars)
Token-efficient Track-1 agent handling all 8 categories via one Fireworks call with hidden reasoning disabled (the dominant token cost), plus an optional free local tier and a calibrated gate that escalates only when unsure.

## Long description
TokenGolf answers all eight Track-1 categories while minimizing the Fireworks tokens the leaderboard scores. Its key optimization: both allowed Fireworks models (minimax-m3, kimi-k2p7-code) are reasoning models whose hidden reasoning dominates token cost, so the agent disables it (reasoning_effort="none"), which measurably cuts tokens ~46% with no accuracy loss. It implements the harness contract exactly (/input/tasks.json → /output/results.json, all inference through FIREWORKS_BASE_URL, models read from ALLOWED_MODELS), builds linux/amd64, and stays well under the 10 GB / 60 s / 30 s-per-request limits. An optional free local tier (Qwen2.5-3B via llama.cpp, zero scored tokens) answers sentiment and NER, and a self-consistency confidence signal, calibrated on held-out data so the escalation threshold is measured rather than hand-tuned, decides when to spend a single Fireworks call. The cascade is binary, never a chain, since each call only adds tokens.

## Technologies Used
Fireworks AI, Docker, Python, llama.cpp, Qwen, AMD Developer Cloud

## Public artifacts
- **Docker image (submit this):** `docker.io/soren19/router-agent:smartlocal` (amd64, PUBLIC ✓) = **most token-efficient**. Reasoning off (kimi) for the hard categories, PLUS four $0 paths (each exact-or-abstain — never a wrong answer): **sentiment + NER → free local Qwen**, **arithmetic/percentages → a deterministic calculator**, and **"evaluate/output of this code" → a sandboxed executor** (AST-locked, injection-safe). On a mixed test: **4/6 answered free, 176 remote tokens**. NER prompt fixed (no example-parroting hallucination); zero-Fireworks-call DQ guard included. **Robustness:** `results.json` is written incrementally + atomically (complete/valid at all times → partial credit on any crash/timeout, never zero) and a **time watchdog** skips the slow local model near the 10-min cap, flushing the rest to fast Fireworks. **Verified under grading constraints (4 GB RAM / 2 vCPU): 2.2 GB peak, <30 s/request, exit 0.**

### Hardening pass — measured decisions (2026-07-12)
- **`:smartlocal` fits 4 GB / 2 vCPU** — 2229 MiB peak (≈1.8 GB headroom), local calls well under 30 s/request. The 3B tier is viable on the grading box.
- **Math-via-Python (local model writes Python → execute) — BUILT, MEASURED, NOT ENABLED.** On GSM8K the 3B mistranslates word problems *stably* (both agreement samples produce the same wrong setup), so the agreement gate didn't filter errors — ~1/5 fired-correct. Enabling it would ship wrong answers at $0 and drag accuracy, so it stays **dormant** (`ROUTER_MATH_PY`, off). Math word problems keep going to Fireworks (kimi reasoning-off was 100% on the dev bench). Kept in code for a stronger local model. *(A measured negative, kept out — same discipline as the reverted grammar-constrained decoding.)*
  - Alternates: `:kimi` (lean all-remote, ~58 MB — the safe accuracy fallback if local ever misses the gate), `:latest` (minimax).
- **GitHub repo:** https://github.com/nguiaSoren/router-agent (public, README, MIT) ✓
- Cover image: TODO
- Video presentation: **DONE** → `submission/video/TokenGolf_demo.mp4` (114 s: animated deck + live terminal A/B + Soren's voiceover + ducked ambient music + word-by-word animated captions). Upload pending [you]
- Slide deck: TODO (problem → route-by-confidence → token efficiency; note the measured model bench)

## A/B plan (leaderboard is the only ground truth for the real gate)
Submit **`:smartlocal`** (most token-efficient) as the primary. Also submit **`:kimi`** (all-remote, accuracy-safe) as the fallback — if `:smartlocal` clears the gate it ranks lower on tokens (~22%+ fewer) and wins; if local NER ever drags accuracy below the gate on the hidden set, `:kimi` is the safe net. Rate limit is 10/hr, so A/B both.

## Pre-submit checklist
- [ ] Final image rebuilt with the winning config (post-P1) + pushed + **PUBLIC** + amd64 manifest
- [ ] `docker run` on a clean pull produces valid `/output/results.json` (verified)
- [ ] GitHub repo public, README present, MIT license, no secrets
- [ ] Cover image + video + slides uploaded
- [ ] Form fields filled; submitted before **July 12** deadline (check event schedule in local tz)
