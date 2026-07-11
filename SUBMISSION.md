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
- **Docker image (submit this):** `docker.io/soren19/router-agent:smartlocal` (amd64, PUBLIC ✓) = **most token-efficient**. Reasoning off (kimi) for the hard categories, PLUS four $0 paths (each exact-or-abstain — never a wrong answer): **sentiment + NER → free local Qwen**, **arithmetic/percentages → a deterministic calculator**, and **"evaluate/output of this code" → a sandboxed executor** (AST-locked, injection-safe). On a mixed test: **4/6 answered free, 176 remote tokens**. NER prompt fixed (no example-parroting hallucination); zero-Fireworks-call DQ guard included.
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
