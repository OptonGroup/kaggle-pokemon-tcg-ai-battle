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

echo "Zipping $SUBMIT_DIR..."
cd "$SUBMIT_DIR"
zip -r "../../results/${VERSION}.zip" . -x "*.pyc" -x "__pycache__/*"
cd - > /dev/null

echo "Submitting to Kaggle..."
kaggle competitions submit -c "$COMP" -f "results/${VERSION}.zip" -m "$MESSAGE"

echo "Done. Check scores:"
echo "  kaggle competitions submissions -c $COMP"
