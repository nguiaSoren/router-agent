#!/usr/bin/env bash
# Build + push Token Golf submission images to Docker Hub (soren19/tokengolf), amd64.
#
# Usage (run from the repo root — the dir with pyproject.toml):
#   bash scripts/build_push.sh              # all tags, in dependency-safe order
#   bash scripts/build_push.sh local zero   # only the named tags
#
# Requires `docker login` first. The --platform=linux/amd64 cross-builds on Apple Silicon.
# Dependency chain (buildx PULLs the base, so the base must already be pushed):
#   smartlocal → local (FROM :smartlocal) → zero (FROM :local)
#   kimi / latest are independent (FROM python:3.11-slim via Dockerfile.baseline).
set -euo pipefail

REPO="soren19/tokengolf"
PLATFORM="linux/amd64"
cd "$(dirname "$0")/.."   # repo root regardless of where it's invoked from

_build() {   # _build <tag> <extra buildx args...>
  local tag="$1"; shift
  echo ">>> building + pushing ${REPO}:${tag}"
  docker buildx build --platform "$PLATFORM" "$@" -t "${REPO}:${tag}" --push .
}

build_kimi()       { _build kimi       -f docker/Dockerfile.baseline --build-arg FW_MODEL=kimi; }
build_latest()     { _build latest     -f docker/Dockerfile.baseline; }                 # default minimax-m3
build_smartlocal() { _build smartlocal -f docker/Dockerfile; }                          # bakes the ~2GB GGUF
build_local()      { _build local      -f docker/Dockerfile.local; }                    # FROM :smartlocal
build_zero()       { _build zero       -f docker/Dockerfile.zero; }                     # FROM :local

# No args → all tags, in FROM-chain order (smartlocal before local before zero).
if [ "$#" -eq 0 ]; then
  set -- smartlocal local zero kimi latest
fi
for t in "$@"; do
  case "$t" in
    kimi)       build_kimi ;;
    latest)     build_latest ;;
    smartlocal) build_smartlocal ;;
    local)      build_local ;;
    zero)       build_zero ;;
    *) echo "!! unknown tag: $t (known: kimi latest smartlocal local zero)" >&2; exit 2 ;;
  esac
done
echo ">>> done. Verify: https://hub.docker.com/r/${REPO}/tags"
