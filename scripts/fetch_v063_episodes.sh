#!/usr/bin/env bash
# Download all episode replays + own-agent logs for v063_forensic86 A/B
# (tracked-2 as of 2026-07-22).
# Serial, small pause, explicit 429 abort (lesson from project_kaggle_kgat_token:
# 429 is account-wide and cumulative).
#
# Usage: ./scripts/fetch_v063_episodes.sh
set -uo pipefail

COMP="pokemon-tcg-ai-battle"
EPISODES_DIR="data/episodes"
SLEEP=0.4
mkdir -p "$EPISODES_DIR"

SUB_IDS=(54896368 54896381)  # v063_forensic86 A, B

for SUB_ID in "${SUB_IDS[@]}"; do
  echo "=== Submission $SUB_ID ==="
  EPISODE_IDS=$(.venv/bin/kaggle competitions episodes "$SUB_ID" -v --quiet 2>/dev/null \
    | tail -n +2 | cut -d',' -f1)
  n_total=$(echo "$EPISODE_IDS" | grep -c .)
  echo "Found $n_total episodes."

  n_ok=0
  n_skip=0
  n_missing=0

  for EP_ID in $EPISODE_IDS; do
    REPLAY="$EPISODES_DIR/episode-${EP_ID}-replay.json"

    if [ ! -f "$REPLAY" ]; then
      OUT=$(.venv/bin/kaggle competitions replay "$EP_ID" -p "$EPISODES_DIR" -q 2>&1)
      RC=$?
      if echo "$OUT" | grep -qiE "429|too many requests"; then
        echo "!!! 429 hit downloading replay $EP_ID. STOPPING."
        break 2
      fi
      if [ $RC -ne 0 ] || [ ! -f "$REPLAY" ]; then
        echo "  WARNING: episode $EP_ID replay failed (rc=$RC): $OUT"
        n_missing=$((n_missing+1))
        sleep "$SLEEP"
        continue
      fi
      sleep "$SLEEP"
    fi

    # Determine our agent index from the replay (team name == "Maxim Simakov")
    OUR_IDX=$(python3 -c "
import json
with open('$REPLAY') as f:
    d = json.load(f)
names = d['info']['TeamNames']
print(names.index('Maxim Simakov') if 'Maxim Simakov' in names else -1)
" 2>/dev/null)

    if [ "$OUR_IDX" == "-1" ] || [ -z "$OUR_IDX" ]; then
      echo "  WARNING: could not determine our agent index for episode $EP_ID, skipping logs"
      n_missing=$((n_missing+1))
      continue
    fi

    LOGFILE="$EPISODES_DIR/episode-${EP_ID}-agent-${OUR_IDX}-logs.json"
    if [ -f "$LOGFILE" ]; then
      n_skip=$((n_skip+1))
      continue
    fi

    LOGOUT=$(.venv/bin/kaggle competitions logs "$EP_ID" "$OUR_IDX" -p "$EPISODES_DIR" -q 2>&1)
    if echo "$LOGOUT" | grep -qiE "429|too many requests"; then
      echo "!!! 429 hit downloading logs for episode $EP_ID. STOPPING."
      break 2
    fi
    # kaggle CLI names the log file "{episode_id}-{agent_index}.json" — rename to our convention
    DEFAULT_NAME="$EPISODES_DIR/${EP_ID}-${OUR_IDX}.json"
    if [ -f "$DEFAULT_NAME" ]; then
      mv "$DEFAULT_NAME" "$LOGFILE"
    fi

    n_ok=$((n_ok+1))
    echo "  [$n_ok/$n_total] episode $EP_ID (our idx=$OUR_IDX)"
    sleep "$SLEEP"
  done

  echo "  Downloaded this run: $n_ok | already had: $n_skip | missing: $n_missing"
done

echo ""
echo "Total replay files now in $EPISODES_DIR: $(ls "$EPISODES_DIR"/episode-*-replay.json 2>/dev/null | wc -l | tr -d ' ')"
