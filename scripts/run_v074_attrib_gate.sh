#!/bin/bash
# Атрибуция P1: какая из двух половин фикса даёт замеренные -5.64 пп на крустл-страте.
#   v074i_switchonly  = v074c + ТОЛЬКО (a) демоут работает и на CTX_SWITCH
#   v074j_unstickonly = v074c + ТОЛЬКО (b) unstick не отступает в тело под Rock Inn
# Вместе с уже посчитанными v074c (ни (a), ни (b)) и v074f (обе) это полный дизайн 2x2.
set -u
M=/Users/max/kaggle-pokemon-tcg-ai-battle
W=$M/.claude/worktrees/v074-live-forensic
PY=$M/.venv/bin/python
LOGDIR=$W/out/v074_gate
run_full() {
  local cand=$1 label=$2 seed=$3
  echo "=== [$(date +%H:%M:%S)] $label  (candidate=$cand, seed=$seed) ===" >&2
  "$PY" "$M/scripts/gate_v074_fullmetal.py" --n 1000 --n-proxy 1000 --workers 8 --seed "$seed" \
      --candidate "$W/submissions/$cand" --label "$label" 2>&1 \
    | grep -v -E '^\[alt\]|^\[dec\] t[0-9]|^\[dbg\] guard=|^\[galt\]' > "$LOGDIR/$label.log"
  echo "=== [$(date +%H:%M:%S)] $label done ===" >&2
}
run_full v074i_switchonly  v074i_r1 30260791
run_full v074j_unstickonly v074j_r1 30260801
run_full v074i_switchonly  v074i_r2 30260792
run_full v074j_unstickonly v074j_r2 30260802
echo "=== АТРИБУЦИЯ ЗАВЕРШЕНА [$(date +%H:%M:%S)] ===" >&2
