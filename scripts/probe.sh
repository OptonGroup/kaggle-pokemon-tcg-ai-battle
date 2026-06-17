#!/usr/bin/env bash
# probe.sh — build the linux/amd64 Docker image and run the Q-SEARCH probe.
#
# Usage:
#   ./scripts/probe.sh [n_rollouts] [seed]
#
# Defaults: 30 rollouts, seed=7.
#
# Measures engine search cost (search_begin / search_step) to size the search
# budget for v003. Runs inside Docker (libcg.so is Linux x86_64 only).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="pokemon-tcg-sim"
N_ROLLOUTS="${1:-30}"
SEED="${2:-7}"

# Ensure Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    echo "=== Docker not running — launching Docker Desktop ==="
    open -a Docker
    echo "Waiting for Docker daemon..."
    until docker info >/dev/null 2>&1; do
        sleep 2
    done
    echo "Docker is ready."
fi

echo "=== Building Docker image (linux/amd64) ==="
docker build \
    --platform linux/amd64 \
    -t "$IMAGE_NAME" \
    -f "$REPO_ROOT/docker/Dockerfile" \
    "$REPO_ROOT"

echo ""
echo "=== Running Q-SEARCH probe: $N_ROLLOUTS rollouts, seed=$SEED ==="
docker run --rm \
    --platform linux/amd64 \
    "$IMAGE_NAME" \
    python src/eval/search_probe.py "$N_ROLLOUTS" "$SEED"
