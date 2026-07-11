#!/usr/bin/env bash
# run_exit.sh — Запуск ExIt-цикла (expert-iteration против замороженного поля) в Docker.
#
# Требует: Docker с поддержкой linux/amd64 (native или qemu-эмуляция).
# Движок libcg.so — Linux x86_64 only; реальные партии — только в Docker.
#
# Использование:
#   ./scripts/run_exit.sh [n_iters] [games_per_opp] [workers] [pimc_budget] [filter_mode] [init_model]
#
# Параметры (все опциональные):
#   n_iters       — число итераций ExIt               (по умолчанию: 5)
#   games_per_opp — партий против каждого оппонента   (по умолчанию: 300)
#   workers       — параллельных процессов             (по умолчанию: 4)
#   pimc_budget   — бюджет PIMC на ход, сек            (по умолчанию: 3.0)
#   filter_mode   — "win" | "dagger"                   (по умолчанию: win)
#   init_model    — путь к стартовой модели .txt       (по умолчанию: models/il_iono_a.txt)
#
# Примеры:
#   ./scripts/run_exit.sh                              # production defaults
#   ./scripts/run_exit.sh 1 20 2 1.0 win               # smoke-тест (быстро)
#   ./scripts/run_exit.sh 10 300 8 3.0 win             # extended run
#   ./scripts/run_exit.sh 5 300 4 3.0 win /workspace/models/il_iono_a.txt
#
# Выходные файлы (в ./models/ и ./results/):
#   models/pilot_iter{N}.txt  — чекпойнт после каждой успешной итерации
#   models/pilot_best.txt     — лучший pilot по winrate против поля
#   results/exit_log.csv      — per-iteration лог (winrate, wilson_lo, promoted, ...)
#   results/trajectories_*.jsonl — собранные траектории (каждый run_games)
#
# СЕВЕРНАЯ ЗВЕЗДА: aggregate winrate против замороженного поля (не self-play WR).
# Гейт: winrate>=55% AND wilson_lo>50% AND n_games>=300.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="pokemon-tcg-sim"

N_ITERS="${1:-5}"
GAMES_PER_OPP="${2:-300}"
WORKERS="${3:-4}"
PIMC_BUDGET="${4:-3.0}"
FILTER_MODE="${5:-win}"
INIT_MODEL="${6:-}"

echo "============================================================"
echo " ExIt-цикл: expert-iteration против замороженного поля"
echo "============================================================"
echo "  Итераций      : $N_ITERS"
echo "  Партий/оппон  : $GAMES_PER_OPP"
echo "  Workers       : $WORKERS"
echo "  PIMC budget   : ${PIMC_BUDGET}s"
echo "  Filter mode   : $FILTER_MODE"
if [ -n "$INIT_MODEL" ]; then
    echo "  Init model    : $INIT_MODEL"
else
    echo "  Init model    : auto (models/il_iono_a.txt или pilot_best.txt)"
fi
echo "============================================================"
echo ""

# ── Сборка Docker-образа ──────────────────────────────────────────────────────
echo "=== Сборка Docker-образа (linux/amd64) ==="
docker build \
    --platform linux/amd64 \
    -t "$IMAGE_NAME" \
    -f "$REPO_ROOT/docker/Dockerfile" \
    "$REPO_ROOT"
echo ""

# ── Формируем аргументы для CLI ───────────────────────────────────────────────
EXTRA_ARGS=""
if [ -n "$INIT_MODEL" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --init-model $INIT_MODEL"
fi

# ── Запуск ExIt в Docker ──────────────────────────────────────────────────────
echo "=== Запуск ExIt-цикла ==="
docker run --rm \
    --platform linux/amd64 \
    -v "$REPO_ROOT":/workspace \
    -e PYTHONPATH=/workspace/src \
    "$IMAGE_NAME" \
    python /workspace/src/exit/exit_loop.py \
        --n-iters       "$N_ITERS" \
        --games-per-opp "$GAMES_PER_OPP" \
        --workers       "$WORKERS" \
        --pimc-budget   "$PIMC_BUDGET" \
        --filter-mode   "$FILTER_MODE" \
        --log-path      "/workspace/results/exit_log.csv" \
        --out-dir       "/workspace/models" \
        $EXTRA_ARGS

echo ""
echo "=== Готово ==="
echo "Лучший pilot : $REPO_ROOT/models/pilot_best.txt"
echo "Лог          : $REPO_ROOT/results/exit_log.csv"
