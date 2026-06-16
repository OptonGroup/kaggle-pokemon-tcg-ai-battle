#!/usr/bin/env bash
# Usage: ./scripts/submit.sh <version> [message]
# Example: ./scripts/submit.sh v002_heuristic "heuristic agent"
set -euo pipefail

VERSION=${1:?Usage: $0 <version> [message]}
MESSAGE=${2:-"$VERSION submission"}
COMP="pokemon-tcg-ai-battle"

SUBMIT_DIR="submissions/$VERSION"
if [ ! -d "$SUBMIT_DIR" ]; then
  echo "ERROR: $SUBMIT_DIR not found"
  exit 1
fi

ARCHIVE="results/${VERSION}.tar.gz"
echo "Packing $SUBMIT_DIR -> $ARCHIVE ..."
mkdir -p results
(cd "$SUBMIT_DIR" && tar -czvf "../../$ARCHIVE" \
  --exclude="*.pyc" \
  --exclude="__pycache__" \
  --exclude=".DS_Store" \
  .)

echo "Submitting to Kaggle..."
.venv/bin/kaggle competitions submit -c "$COMP" -f "$ARCHIVE" -m "$MESSAGE"

echo ""
echo "Done. Track your submission:"
echo "  .venv/bin/kaggle competitions submissions -c $COMP"
