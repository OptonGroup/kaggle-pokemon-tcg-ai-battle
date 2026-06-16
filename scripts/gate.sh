#!/usr/bin/env bash
# gate.sh — run the Submission Gate for a candidate version against the champion.
#
# Usage:
#   ./scripts/gate.sh <candidate_version> [n_games] [seed]
#
# Example:
#   ./scripts/gate.sh v002_lucario 50       # smoke test (50 games)
#   ./scripts/gate.sh v002_lucario 300 42   # full gate run
#
# Reads the current champion from results/champion.txt.
# Builds the linux/amd64 Docker image (shared with sim.sh) and runs gate_runner.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="pokemon-tcg-sim"
CANDIDATE="${1:-}"
N_GAMES="${2:-50}"
SEED="${3:-42}"

if [[ -z "$CANDIDATE" ]]; then
    echo "Usage: $0 <candidate_version> [n_games] [seed]"
    exit 1
fi

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
echo "=== Running gate: candidate=$CANDIDATE n_games=$N_GAMES seed=$SEED ==="
docker run --rm \
    --platform linux/amd64 \
    "$IMAGE_NAME" \
    python src/eval/gate_runner.py "$CANDIDATE" "$N_GAMES" "$SEED"
