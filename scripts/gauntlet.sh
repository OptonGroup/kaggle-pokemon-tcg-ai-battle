#!/usr/bin/env bash
# gauntlet.sh — run the round-robin deck gauntlet inside Docker (linux/amd64).
#
# Usage:
#   ./scripts/gauntlet.sh [decks_dir] [n_games] [seed]
#
# Defaults:
#   decks_dir = decks/gauntlet
#   n_games   = 40
#   seed      = 42
#
# Example:
#   ./scripts/gauntlet.sh decks/gauntlet 40 42   # smoke: 5 decks × 10 pairs × 40 = 400 games

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="pokemon-tcg-sim"
DECKS_DIR="${1:-decks/gauntlet}"
N_GAMES="${2:-40}"
SEED="${3:-42}"

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
echo "=== Running gauntlet: decks_dir=$DECKS_DIR n_games=$N_GAMES seed=$SEED ==="
docker run --rm \
    --platform linux/amd64 \
    "$IMAGE_NAME" \
    python src/eval/gauntlet.py "$DECKS_DIR" "$N_GAMES" "$SEED"
