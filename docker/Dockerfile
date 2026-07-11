# syntax=docker/dockerfile:1
#
# Token Golf — Track 1 submission image (Participant Guide contract).
# =====================================================================
# Judging VM = linux/amd64, CPU (no GPU): image <=10GB, ready in 60s, 10min total / <30s per request.
# Container reads /input/tasks.json -> writes /output/results.json. Binary cascade:
#   FREE local GGUF (llama.cpp, CPU) answers the confident slice; escalations go to ONE Fireworks
#   model via the harness-injected FIREWORKS_BASE_URL (token-counting proxy) using ALLOWED_MODELS.
#   The leaderboard counts TOKENS, so local answers = 0 tokens = the ranking lever.
#
# Build on Apple Silicon (judging VM is amd64 — cross-build + push):
#   docker buildx build --platform linux/amd64 -f docker/Dockerfile -t <registry>/<img>:latest --push .
#
# The `--platform=linux/amd64` constant is INTENTIONAL (buildx warns; ignore it): it guarantees an
# amd64 image even from a plain `docker build` on Apple Silicon — the "wrong arch = score zero" footgun.
FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Build tools for llama-cpp-python (used only if pip can't find a prebuilt CPU wheel).
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Runtime deps: OpenAI-compatible client (Fireworks) + llama.cpp (local free tier) + hf download.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serving,local]"

# Bake the local GGUF into the image (no runtime download → fast startup, no network needed).
# ~2GB Q4 model; total image stays well under the 10GB cap. Verified filename 2026-07-07.
ENV LOCAL_GGUF_PATH=/app/models/qwen2.5-3b-instruct-q4_k_m.gguf
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-3B-Instruct-GGUF', 'qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='/app/models')"

# Non-root + pre-created harness mount points.
RUN useradd --create-home --uid 1001 runner \
    && mkdir -p /input /output && chown -R runner:runner /app /input /output
USER runner

# :smartlocal mode — category-gated: sentiment/NER prompts → the FREE baked local model (Qwen owns
# them 1.00/0.88 in our bench), everything else → one Fireworks call. Token edge: ~2/8 categories
# cost $0. (Override ROUTER_SMARTLOCAL="" to fall back to the confidence cascade.)
#   ROUTER_FW_MODEL=kimi — measured best of the two allowed models with reasoning off (higher acc,
#     fewer tokens than minimax-m3). A/B against minimax on the live leaderboard if desired.
#   Reasoning is disabled by CODE default (run.build_tiers → reasoning_effort="none"): the Fireworks
#     models are reasoning models whose hidden reasoning was ~46% of scored tokens; "none" cuts it
#     with no measured accuracy loss. Set ROUTER_REASONING_EFFORT="" to restore full reasoning.
ENV ROUTER_SMARTLOCAL=1 \
    ROUTER_FW_MODEL=kimi

# Entrypoint = the harness contract; config (ALLOWED_MODELS / FIREWORKS_*) read from env at runtime.
ENTRYPOINT ["python", "-m", "router_agent.run", "submit"]
