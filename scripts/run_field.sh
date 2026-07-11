#!/usr/bin/env bash
# run_field.sh — запуск WP2 харнеса массовых прогонов в Docker (Linux x86_64).
#
# Использование:
#   ./scripts/run_field.sh [candidate_version] [n_games] [workers] [extra_flags]
#
# Примеры:
#   ./scripts/run_field.sh v012_ref 100 4
#   ./scripts/run_field.sh v014_pimc_pilot 200 4 --no-trajectories
#   ./scripts/run_field.sh v012_ref 20 2 --opponents random sample_random
#
# Аргументы:
#   candidate_version — версия из submissions/ (по умолчанию: v012_ref)
#   n_games           — партий на каждого оппонента (по умолчанию: 100)
#   workers           — число параллельных процессов (по умолчанию: 4)
#   extra_flags       — передаются напрямую в harness.py (напр. --no-trajectories)
#
# Результаты (winrates + JSONL датасет) сохраняются в results/.
# Требует: Docker с поддержкой linux/amd64 (macOS ARM работает через эмуляцию,
# но ЗНАЧИТЕЛЬНО медленнее — ~52 rollout/сек vs ~500+ на Kaggle).
#
# Для полного прогона (300 партий на оппонента) используйте Kaggle notebook.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="pokemon-tcg-sim"

CANDIDATE="${1:-v012_ref}"
N_GAMES="${2:-100}"
WORKERS="${3:-4}"
shift 3 2>/dev/null || true   # убираем первые 3 аргумента; остальное — extra_flags
EXTRA="${*:-}"

echo "=== WP2 run_field: candidate=${CANDIDATE}, n=${N_GAMES}, workers=${WORKERS} ==="
echo ""

echo "--- Сборка Docker-образа (linux/amd64) ---"
docker build \
    --platform linux/amd64 \
    -t "$IMAGE_NAME" \
    -f "$REPO_ROOT/docker/Dockerfile" \
    "$REPO_ROOT"

echo ""
echo "--- Запуск харнеса ---"
docker run --rm \
    --platform linux/amd64 \
    -v "$REPO_ROOT":/workspace \
    -e PYTHONPATH=/workspace/src \
    "$IMAGE_NAME" \
    python /workspace/src/exit/harness.py \
        --candidate "$CANDIDATE" \
        --n "$N_GAMES" \
        --workers "$WORKERS" \
        $EXTRA

echo ""
echo "Готово. Результаты в /workspace/results/"
