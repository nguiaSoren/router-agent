# router-agent — Track 1 submission image (linux/amd64, CPU)

The judging VM runs **linux/amd64** (Participant Guide) — no GPU is promised, image ≤10 GB, ready in 60 s, 10 min total / <30 s per request. So this image is **CPU-only and lean**: a `python:3.11-slim` base + the OpenAI-compatible client (`.[serving]`), no ROCm and no bundled model runtime. v1 is **Fireworks-only** — every answer is a Fireworks call routed through the harness proxy. (A small in-container local model can be added later *iff* the rules confirm local answering is allowed and there's CPU headroom under the 30 s limit — earlier we mistakenly assumed an AMD/ROCm GPU box; the guide says otherwise.)

## The contract it implements
- reads **`/input/tasks.json`** = `[{ "task_id", "prompt" }]`
- writes **`/output/results.json`** = `[{ "task_id", "answer" }]`
- reads env injected by the harness: **`FIREWORKS_API_KEY`**, **`FIREWORKS_BASE_URL`** (token-counting proxy — all calls MUST go through it), **`ALLOWED_MODELS`** (comma-separated ids). Never hardcode these.
- exit 0 on success.

## Build & push (judging VM is amd64)
```bash
# from the repo root (contains pyproject.toml). On Apple Silicon, cross-build for amd64:
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t <registry>/router-agent:latest --push .
# On an Intel/AMD host, a plain build already targets amd64:
docker build -f docker/Dockerfile -t <registry>/router-agent:latest .
```

## Run locally (smoke)
```bash
mkdir -p input output
echo '[{"task_id":"t1","prompt":"What is 2+2?"}]' > input/tasks.json
docker run --rm \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
  -e ALLOWED_MODELS="<id1>,<id2>" \
  -v "$PWD/input:/input" -v "$PWD/output:/output" \
  <registry>/router-agent:latest
cat output/results.json
```

**Tuning knobs (env, no rebuild):** `ROUTER_MODEL_INDEX` (which allowed model in single-model mode; default -1 = strongest), `ROUTER_CALIBRATOR` (path to a bundled calibrator → full cascade), `ROUTER_SC_N`, `ROUTER_TAU`, `ROUTER_MAX_TOKENS`.

**Status:** entrypoint is live (`python -m router_agent.run submit`). The image needs `ALLOWED_MODELS` + the real `FIREWORKS_BASE_URL`/key (published at launch) to run end to end; the wiring + submit path are unit-tested offline.
