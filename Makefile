# router-agent — common tasks. Run from hackathon_track1/.
# Live targets load API keys from an optional env file (defaults to the parent ../.env during dev;
# override with `make live-smoke ENV_FILE=path`). Standalone, just export the keys yourself.

ENV_FILE ?= ../.env
LOADENV = set -a; [ -f $(ENV_FILE) ] && . $(ENV_FILE); set +a

.PHONY: test lint live-models live-smoke live-smoke-full

test:                  ## offline suite (deterministic, no network)
	uv run --extra dev pytest -q

lint:                  ## ruff over all code
	uv run --extra dev ruff check src/ tests/ experiments/ scripts/

live-models:           ## list remote model ids so you can pick DEV_REMOTE_MODEL_ID (L6)
	$(LOADENV); uv run --extra serving python scripts/live_smoke.py --list-models

live-smoke:            ## exercise the real local + remote seams on dev credits (ticks 0e)
	$(LOADENV); uv run --extra serving python scripts/live_smoke.py

live-smoke-full:       ## live-smoke + a tiny end-to-end cascade run
	$(LOADENV); uv run --extra serving python scripts/live_smoke.py --full
