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
- **Docker image (submit this):** `docker.io/soren19/router-agent:kimi` (lean, kimi-k2p7-code, reasoning off, amd64, 58 MB, PUBLIC ✓) = measured-best
  - Alternates: `:latest` (minimax fallback), `:smartlocal` (adds the free-local tier)
- **GitHub repo:** https://github.com/nguiaSoren/router-agent (public, README, MIT) ✓
- Cover image: TODO
- Video presentation: TODO (short demo: `docker run` → results.json + token count)
- Slide deck: TODO (problem → route-by-confidence → token efficiency; note the measured model bench)

## A/B plan (leaderboard is the only ground truth for the real gate)
Submit `:latest` (minimax, safe) first → confirm it clears the gate + note its token rank. Then submit `:kimi` → if it clears the gate too, it should rank higher on tokens. Keep whichever ranks best. Rate limit is 10/hr.

## Pre-submit checklist
- [ ] Final image rebuilt with the winning config (post-P1) + pushed + **PUBLIC** + amd64 manifest
- [ ] `docker run` on a clean pull produces valid `/output/results.json` (verified)
- [ ] GitHub repo public, README present, MIT license, no secrets
- [ ] Cover image + video + slides uploaded
- [ ] Form fields filled; submitted before **July 12** deadline (check event schedule in local tz)
