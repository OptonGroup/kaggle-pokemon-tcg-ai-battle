#!/usr/bin/env bash
# run_gate_safe.sh — mirror gate with hard Docker timeout to prevent hangs.
# Usage: ./scripts/run_gate_safe.sh <candidate> <opponent> <n_games> <seed> <out_json>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="pokemon-tcg-sim"
CANDIDATE="${1:-}"
OPPONENT="${2:-}"
N_GAMES="${3:-100}"
SEED="${4:-0}"
OUT_JSON="${5:-}"

if [[ -z "$CANDIDATE" || -z "$OPPONENT" || -z "$OUT_JSON" ]]; then
    echo "Usage: $0 <candidate> <opponent> <n_games> <seed> <out_json>"
    exit 1
fi

# Per-game timeout budget: ~60s per game * N_GAMES + 120s startup overhead
# For 100 games: 6120s; for 400 games: 24120s — use a flat generous ceiling
DOCKER_TIMEOUT=$(( N_GAMES * 60 + 300 ))

echo "=== Mirror Gate: $CANDIDATE vs $OPPONENT ==="
echo "  n_games=$N_GAMES  seed=$SEED"
echo "  Docker timeout: ${DOCKER_TIMEOUT}s"
echo "  Output: $OUT_JSON"
echo ""

LOG_TMP="$(mktemp /tmp/gate_XXXXXX.log)"

# Run docker with a hard timeout; capture output
if timeout "$DOCKER_TIMEOUT" docker run --rm \
    --platform linux/amd64 \
    --memory=4g \
    "$IMAGE_NAME" \
    python src/eval/gate_runner.py "$CANDIDATE" "$OPPONENT" "$N_GAMES" "$SEED" \
    2>&1 | tee "$LOG_TMP"; then
    EXIT_CODE=0
else
    EXIT_CODE=$?
    if [[ $EXIT_CODE -eq 124 ]]; then
        echo ""
        echo "ERROR: Docker run timed out after ${DOCKER_TIMEOUT}s" | tee -a "$LOG_TMP"
    fi
fi

echo ""
echo "=== Parsing results from log ==="

# Parse the log to extract numbers
WINS_A=$(grep -oP "Candidate wins\s*:\s*\K[0-9]+" "$LOG_TMP" || echo "")
WINS_B=$(grep -oP "Champion wins\s*:\s*\K[0-9]+" "$LOG_TMP" || echo "")
DRAWS=$(grep -oP "Draws\s*:\s*\K[0-9]+" "$LOG_TMP" || echo "")
WINRATE=$(grep -oP "Winrate \(excl\. draws\):\s*\K[0-9.]+" "$LOG_TMP" || echo "")
WILSON_LO=$(grep -oP "Wilson 95% CI\s*:\s*\[\K[0-9.]+" "$LOG_TMP" || echo "")
WILSON_HI=$(grep -oP "Wilson 95% CI\s*:.*?,\s*\K[0-9.]+" "$LOG_TMP" || echo "")
VERDICT=$(grep -oP "VERDICT: \K(PASS|FAIL)" "$LOG_TMP" || echo "UNKNOWN")

echo "  wins_A=$WINS_A  wins_B=$WINS_B  draws=$DRAWS"
echo "  winrate=$WINRATE  wilson_lo=$WILSON_LO  wilson_hi=$WILSON_HI"
echo "  verdict=$VERDICT"

# Write JSON result
python3 - <<PYEOF
import json, datetime

data = {
    "candidate": "$CANDIDATE",
    "opponent": "$OPPONENT",
    "n_games": $N_GAMES,
    "seed": $SEED,
    "wins_A": int("$WINS_A") if "$WINS_A" else None,
    "wins_B": int("$WINS_B") if "$WINS_B" else None,
    "draws": int("$DRAWS") if "$DRAWS" else None,
    "winrate_A": float("$WINRATE") if "$WINRATE" else None,
    "wilson_lo": float("$WILSON_LO") if "$WILSON_LO" else None,
    "wilson_hi": float("$WILSON_HI") if "$WILSON_HI" else None,
    "verdict": "$VERDICT",
    "exit_code": $EXIT_CODE,
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
}

with open("$OUT_JSON", "w") as f:
    json.dump(data, f, indent=2)

print(f"Written to $OUT_JSON")
PYEOF

rm -f "$LOG_TMP"
exit $EXIT_CODE
