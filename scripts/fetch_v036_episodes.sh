#!/usr/bin/env bash
# Download all episode replays for a SINGLE submission (v036 forensic analysis).
# Serial (no threads), small pause between requests, explicit 429 detection ->
# abort immediately instead of silently continuing (lesson from
# project_kaggle_kgat_token: 429 is account-wide and cumulative; retrying
# extends the block, and a silent scanner just looks "dead").
#
# Usage: ./scripts/fetch_v036_episodes.sh [SUBMISSION_ID]
set -uo pipefail

SUB_ID="${1:-54768157}"  # v036_archaludon_patched
COMP="pokemon-tcg-ai-battle"
EPISODES_DIR="data/episodes"
SLEEP=0.4
mkdir -p "$EPISODES_DIR"

echo "Fetching episode list for submission $SUB_ID..."
EPISODE_IDS=$(.venv/bin/kaggle competitions episodes "$SUB_ID" -v --quiet 2>/dev/null \
  | tail -n +2 | cut -d',' -f1)

n_total=$(echo "$EPISODE_IDS" | grep -c .)
echo "Found $n_total episodes."

n_ok=0
n_skip=0
n_missing=0
missing_ids=()

for EP_ID in $EPISODE_IDS; do
  REPLAY="$EPISODES_DIR/episode-${EP_ID}-replay.json"
  if [ -f "$REPLAY" ]; then
    n_skip=$((n_skip+1))
    continue
  fi

  OUT=$(.venv/bin/kaggle competitions replay "$EP_ID" -p "$EPISODES_DIR" -q 2>&1)
  RC=$?

  if echo "$OUT" | grep -qiE "429|too many requests"; then
    echo "!!! 429 Too Many Requests hit at episode $EP_ID. STOPPING (not retrying)."
    echo "Output: $OUT"
    break
  fi

  if [ $RC -ne 0 ] || [ ! -f "$REPLAY" ]; then
    echo "  WARNING: episode $EP_ID replay failed (rc=$RC): $OUT"
    n_missing=$((n_missing+1))
    missing_ids+=("$EP_ID")
    sleep "$SLEEP"
    continue
  fi

  LOGOUT=$(.venv/bin/kaggle competitions logs "$EP_ID" 0 -p "$EPISODES_DIR" -q 2>&1)
  if echo "$LOGOUT" | grep -qiE "429|too many requests"; then
    echo "!!! 429 Too Many Requests hit on logs for episode $EP_ID. STOPPING."
    break
  fi

  n_ok=$((n_ok+1))
  echo "  [$n_ok/$n_total] downloaded episode $EP_ID"
  sleep "$SLEEP"
done

echo ""
echo "=== Summary ==="
echo "Downloaded this run: $n_ok"
echo "Already had (skipped): $n_skip"
echo "Failed/missing: $n_missing"
if [ "${#missing_ids[@]}" -gt 0 ]; then
  echo "Missing episode IDs: ${missing_ids[*]}"
fi
echo "Total replay files now in $EPISODES_DIR: $(ls "$EPISODES_DIR"/episode-*-replay.json 2>/dev/null | wc -l | tr -d ' ')"
