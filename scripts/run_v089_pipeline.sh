#!/usr/bin/env bash
# run_v089_pipeline.sh — остаток конвейера v089 после фичеризации:
#   1) паритет фич (единственное отличие от v088 = БЛОК 8) — ЖЁСТКИЙ ГЕЙТ, дальше не идём;
#   2) обучение v089 = рецепт v088 (--retreat-weight 1.6, graded-таргет, ctx 2.0, deep-turn 1.5)
#      на НОВЫХ фичах; ровно одно отличие от v088 — набор фич;
#   3) замер на остром холдауте: подкласс "своп повреждённого 648" + контроль attack.
# Сабмит НЕ делается — только офлайн-результат (по правилу проекта нужен явный "ок").
set -euo pipefail
cd /Users/max/kaggle-pokemon-tcg-ai-battle

V=out/corpus_2b/versions/v089_block8
PY=.venv/bin/python

echo "=== [1/3] паритет фич ==="
$PY scripts/verify_v089_features_parity.py \
  --old out/corpus_2b/versions/v088_dragapult/features_v088 \
  --new "$V/features" --out "$V/features_parity.json"

echo
echo "=== [2/3] обучение v089 (вес 1.6 + БЛОК 8) ==="
$PY scripts/train_bc_2b_v088.py \
  --features-dir "$V/features" \
  --out-dir "$V/models_w16_block8" \
  --retreat-weight 1.6 \
  --semantic-target --semantic-target-mode graded \
  --ctx-weight 2.0 --deep-turn-weight 1.5 --deep-turn-threshold 15.0 \
  --high-weight 3.0 --seed 42 --n-jobs 6 \
  2>&1 | tail -25

echo
echo "=== [3/3] замер на остром холдауте ==="
$PY scripts/eval_holdout_v089.py \
  --old-features out/corpus_2b/versions/v088_dragapult/holdout_features \
  --new-features "$V/holdout_features" \
  --model-base out/corpus_2b/versions/v086_targeted/models_v087_bestiter \
  --model-v088 out/corpus_2b/versions/v088_dragapult/final/B_k1k2_16 \
  --model-v089 "$V/models_w16_block8" \
  --out "$V/holdout_report_v089.json"

echo "=== КОНВЕЙЕР v089 ЗАВЕРШЁН ==="
