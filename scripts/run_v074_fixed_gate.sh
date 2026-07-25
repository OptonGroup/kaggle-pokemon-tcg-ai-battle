#!/bin/bash
# Полный гейт для ИСПРАВЛЕННОЙ лестницы v074 (после состязательной проверки правок,
# out/v074_gate2/REVIEW_findings.md — 6 подтверждённых дефектов в первой версии сборок).
#
#   v074f_promofix2 = P1 с двумя исправлениями: (a) Rock-Inn-демоут работает и на CTX_SWITCH,
#                     (b) unstick-ветка score_retreat не отступает в тело, которому Rock Inn
#                     запрещает наносить урон. Без них демоут откатывался через ход и сжигал
#                     {D}-энергию — измеренная причина нулевой дельты у v074c.
#   v074g_pppfix    = + P0 с самосбрасывающимся трекером (reset_game_state недостижим офлайн,
#                     скорер кэшируется в sys.modules -> стейт тёк между играми и давал
#                     фантомный +60).
#   v074h_evofix    = + P2 с двумя исправлениями: гейт больше не обходится второй копией ex в
#                     руке, и проверка летала не требует энергии ДО эволюции (Punk Up
#                     прикрепляет энергию напрямую, свежая ex бьёт 180 в тот же ход).
#
# База для сравнения — v074_r5/v074_r6 (v074_fullmetal, тот же пул, 07-25).
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

run_full v074f_promofix2 v074f_r1 30260761
run_full v074g_pppfix    v074g_r1 30260771
run_full v074h_evofix    v074h_r1 30260781
run_full v074f_promofix2 v074f_r2 30260762
run_full v074g_pppfix    v074g_r2 30260772
run_full v074h_evofix    v074h_r2 30260782

echo "=== ИСПРАВЛЕННАЯ ЛЕСТНИЦА ЗАВЕРШЕНА [$(date +%H:%M:%S)] ===" >&2
