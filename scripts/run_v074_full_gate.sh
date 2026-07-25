#!/bin/bash
# Полный гейт для абляционной лестницы v074 (форензик v073, 98 live-игр).
#
# Арма-лестница (каждая сборка = предыдущая + одна правка, поэтому дельты аддитивны):
#   v074_fullmetal  = P3 Full Metal Lab (действующий измеренный кандидат)  -> метки v074_r5/r6
#   v074c_promofix  = + P1 Rock-Inn demote без гейта threat>=hp            -> v074c_r1/r2
#   v074d_pppobs    = + P0 Premium Power Pro по наблюдению в сбросе        -> v074d_r1/r2
#   v074e_evogate   = + P2 гейт на эволюцию в ex в Активе                  -> v074e_r1/r2
#
# Почему 2 повтора на арму: стенд НЕВОСПРОИЗВОДИМ (движок не принимает сид), шумовой пол
# ±1-2 пп на пару при N=1000 (ловушка 2 из docs/gold/PLAN_v074.md). Вердикт даётся только
# по пулу повторов через scripts/v074_gate_aggregate.py.
#
# Логи фильтруются на лету: полные [dec]/[alt]-трейсы дают ~300-500 МБ на прогон, а нужны
# из них только счётчики срабатываний новых веток.
set -u
M=/Users/max/kaggle-pokemon-tcg-ai-battle
W=$M/.claude/worktrees/v074-live-forensic
PY=$M/.venv/bin/python
LOGDIR=$W/out/v074_gate
mkdir -p "$LOGDIR"

run_full() {
  local cand=$1 label=$2 seed=$3
  echo "=== [$(date +%H:%M:%S)] $label  (candidate=$cand, seed=$seed) ===" >&2
  "$PY" "$M/scripts/gate_v074_fullmetal.py" \
      --n 1000 --n-proxy 1000 --workers 8 --seed "$seed" \
      --candidate "$W/submissions/$cand" --label "$label" 2>&1 \
    | grep -v -E '^\[alt\]|^\[dec\] t[0-9]|^\[dbg\] guard=|^\[galt\]' \
    > "$LOGDIR/$label.log"
  echo "=== [$(date +%H:%M:%S)] $label done, log $(du -h "$LOGDIR/$label.log" | cut -f1) ===" >&2
}

run_full v074_fullmetal  v074_r5   30260725
run_full v074_fullmetal  v074_r6   30260726
run_full v074c_promofix  v074c_r1  30260731
run_full v074c_promofix  v074c_r2  30260732
run_full v074d_pppobs    v074d_r1  30260741
run_full v074d_pppobs    v074d_r2  30260742
run_full v074e_evogate   v074e_r1  30260751
run_full v074e_evogate   v074e_r2  30260752

echo "=== ВСЕ ПРОГОНЫ ЗАВЕРШЕНЫ [$(date +%H:%M:%S)] ===" >&2
