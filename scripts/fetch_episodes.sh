#!/usr/bin/env bash
# Download all episode replays for all submissions
# Usage: ./scripts/fetch_episodes.sh
set -euo pipefail

COMP="pokemon-tcg-ai-battle"
EPISODES_DIR="data/episodes"
mkdir -p "$EPISODES_DIR"

echo "Fetching submission list..."
SUBMISSION_IDS=$(.venv/bin/kaggle competitions submissions -c "$COMP" -v --quiet 2>/dev/null \
  | tail -n +2 | cut -d',' -f1)

for SUB_ID in $SUBMISSION_IDS; do
  echo "Episodes for submission $SUB_ID:"
  EPISODE_IDS=$(.venv/bin/kaggle competitions episodes "$SUB_ID" -v --quiet 2>/dev/null \
    | tail -n +2 | cut -d',' -f1)

  for EP_ID in $EPISODE_IDS; do
    REPLAY="$EPISODES_DIR/episode-${EP_ID}-replay.json"
    if [ -f "$REPLAY" ]; then
      echo "  $EP_ID already downloaded, skipping"
      continue
    fi
    echo "  Downloading episode $EP_ID..."
    .venv/bin/kaggle competitions replay "$EP_ID" -p "$EPISODES_DIR" -q || echo "  WARNING: replay $EP_ID failed"
    .venv/bin/kaggle competitions logs "$EP_ID" 0 -p "$EPISODES_DIR" -q || true
  done
done

echo ""
echo "Episodes in $EPISODES_DIR:"
ls "$EPISODES_DIR"/*.json 2>/dev/null | wc -l
