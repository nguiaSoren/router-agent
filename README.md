![TokenGolf](submission/deck/out/github_card.png)

# TokenGolf: Token-Efficient Routing Agent (AMD ACT II, Track 1)

A Track-1 agent that answers tasks across 8 capability categories (factual, math, sentiment, summarisation, NER, code-debugging, logic, code-generation) using **as few Fireworks tokens as possible**. It reads `/input/tasks.json`, writes `/output/results.json`, and routes each task through a **cost-ordered cascade**: free deterministic + local paths answer what they can prove they're right on, and only the rest escalate to a single Fireworks model. Scored on an accuracy gate, then ranked by total tokens — so the design maximises free coverage while staying above the gate.

## How it works
- **Reasoning off is the token win.** Both allowed Fireworks models are reasoning models whose hidden reasoning dominates token cost. Setting `reasoning_effort="none"` cuts scored tokens ~46% with no measured accuracy loss (it was the dominant cost, not the answers).
- **$0 exact-or-abstain tiers.** Before any Fireworks token is spent, deterministic paths answer for free — but only when they can be *proven* correct; otherwise they abstain and escalate, so they can never drag accuracy down:
  - **sentiment + NER → a free local Qwen2.5-3B** (llama.cpp, CPU; 0 scored tokens). The NER prompt copies entities verbatim and never invents (no in-prompt example to parrot).
  - **arithmetic & percentages → a calculator** (`heuristics.solve_math`): exact on `47 * 13` / `15% of 240`; abstains on word problems.
  - **"evaluate / output of this code" → an AST-locked sandbox** (`heuristics.solve_code`): runs the snippet (imports, dunder access, and dangerous builtins rejected; SIGALRM timeout) and returns the result; abstains on *generative* debug/gen (runs ≠ correct) and anything unsafe.
- **Confidence-gated routing.** When the calibrated cascade is enabled, the local model is sampled with self-consistency; high agreement keeps the local answer (free), low agreement escalates. The keep/escalate threshold τ is **calibrated on held-out data** (no silent promotion).
- **Token-minimal by construction.** The leaderboard counts tokens, so the cascade is binary (free/local → one Fireworks call, never a chain), the Fireworks prompt is terse and output-capped, and a guard forces one remote call if a whole batch is answered for free (a zero-API-call run is disqualified).
- **Config, not code.** Models, mode, and thresholds are read from the environment at runtime.

## The harness contract
- Reads **`/input/tasks.json`**: `[{ "task_id", "prompt" }]`
- Writes **`/output/results.json`**: `[{ "task_id", "answer" }]`
- Environment (injected by the eval harness; do not hardcode): `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL` (all Fireworks calls route through it), `ALLOWED_MODELS` (comma-separated ids).

## Build & run
```bash
# Judging VM is linux/amd64. From the repo root — :smartlocal is the submit image (bakes the local GGUF):
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t <registry>/router-agent:smartlocal --push .

# Local smoke:
mkdir -p input output && echo '[{"task_id":"t1","prompt":"What is 2+2?"}]' > input/tasks.json
docker run --rm \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
  -e ALLOWED_MODELS="accounts/fireworks/models/kimi-k2p7-code" \
  -v "$PWD/input:/input" -v "$PWD/output:/output" \
  <registry>/router-agent:smartlocal
cat output/results.json
```
The lean all-remote fallback (no local tier) is `docker/Dockerfile.baseline` → `:kimi`.

### Tuning knobs (env, no rebuild)
- `ROUTER_SMARTLOCAL=1`: **the submit mode** — sentiment/NER → free local, arithmetic & code-eval → $0 solvers, everything else → one Fireworks call.
- `ROUTER_REASONING_EFFORT`: reasoning_effort for the Fireworks call (default `none`; set `""` to restore full reasoning). The headline token lever.
- `ROUTER_FW_MODEL`: which allowed model to call by name (e.g. `kimi`, `minimax-m3`).
- `ROUTER_NO_LOCAL=1`: Fireworks-only baseline (skip the local tier).
- `ROUTER_CALIBRATOR`: path to a bundled calibrator JSON, enabling the confidence cascade with the calibrated τ.
- `ROUTER_SC_N`, `ROUTER_MAX_TOKENS`, `ROUTER_TAU`: self-consistency samples / output cap / escalation threshold.

## Layout
`src/router_agent/`: `run.py` (`submit` entrypoint + smartlocal routing), `heuristics.py` (the $0 deterministic tiers: NER regex, `solve_math`, `solve_code`), `cascade.py` (confidence cascade), `confidence.py` (self-consistency), `local_llm.py` (CPU GGUF free tier), `providers.py` (Fireworks seam), `threshold.py`/`calibration/` (τ + calibration), `tasks.py` (8-category dev sets + checkers). `experiments/`: eval + calibration harnesses. `docker/`: the amd64 images. `submission/deck/`: the pitch deck + brand assets.

## Development
```bash
uv run --extra dev pytest -q          # offline test suite (207 tests)
uv run --extra dev ruff check src/ tests/
```

## License
MIT (`LICENSE`). See `ATTRIBUTION.md` for acknowledgments: the established techniques this builds on and the open dataset used.
