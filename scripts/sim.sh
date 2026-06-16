#!/usr/bin/env bash
# sim.sh — build the linux/amd64 Docker image and run the simulation harness.
#
# Usage:
#   ./scripts/sim.sh [n_games] [seed]
#
# Defaults: 20 games, seed=42.
#
# The image is built with --platform=linux/amd64 so libcg.so (Linux x86_64)
# loads correctly even on macOS ARM via emulation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="pokemon-tcg-sim"
N_GAMES="${1:-20}"
SEED="${2:-42}"

echo "=== Building Docker image (linux/amd64) ==="
docker build \
    --platform linux/amd64 \
    -t "$IMAGE_NAME" \
    -f "$REPO_ROOT/docker/Dockerfile" \
    "$REPO_ROOT"

echo ""
echo "=== Running simulation harness: $N_GAMES games, seed=$SEED ==="
docker run --rm \
    --platform linux/amd64 \
    -e SIM_N_GAMES="$N_GAMES" \
    -e SIM_SEED="$SEED" \
    "$IMAGE_NAME" \
    python src/sim_harness.py "$N_GAMES" "$SEED"
