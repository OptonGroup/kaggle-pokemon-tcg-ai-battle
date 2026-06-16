#!/usr/bin/env bash
# promote.sh — promote a version to champion after a confirmed ladder win.
#
# Usage:
#   ./scripts/promote.sh <version>
#
# Writes <version> into results/champion.txt after verifying the submission
# directory exists. Echoes old → new.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-}"

if [[ -z "$VERSION" ]]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 v002_lucario"
    exit 1
fi

SUB_DIR="$REPO_ROOT/submissions/$VERSION"
if [[ ! -d "$SUB_DIR" ]]; then
    echo "ERROR: submission directory does not exist: $SUB_DIR"
    echo "Create the submission first before promoting."
    exit 1
fi

CHAMP_FILE="$REPO_ROOT/results/champion.txt"
OLD_CHAMP="(none)"
if [[ -f "$CHAMP_FILE" ]]; then
    OLD_CHAMP="$(cat "$CHAMP_FILE")"
fi

echo "$VERSION" > "$CHAMP_FILE"
echo "Promoted: $OLD_CHAMP → $VERSION"
echo "results/champion.txt updated."
