#!/usr/bin/env bash
# rollback_v088_scheduled.sh — отложенная заливка v088 ДВУМЯ копиями после сброса
# дневного лимита Kaggle (сброс в UTC-полночь).
#
# Контекст: 2026-08-15 лимит 5/5 был исчерпан (2 копии v088 в 03:07 + 2 копии v089 +
# 1 копия v088-отката), вторая копия отката получила 400. v089 опровергнут live
# (зеркало 62%->27%), на конец 08-15 в tracked-2 оставалась одна копия v089.
# Цель: вернуть tracked-2 = {v088, v088} до дедлайна 08-16.
#
# Заливает ПОСЛЕДОВАТЕЛЬНО с проверкой: если первая копия не зарегистрировалась,
# вторая НЕ отправляется (иначе можно сжечь слот впустую).
set -uo pipefail
cd /Users/max/kaggle-pokemon-tcg-ai-battle

KAGGLE=.venv/bin/kaggle
START_DAY=$(date -u +%Y-%m-%d)

echo "[rollback] старт $(date -u +%H:%M) UTC, текущий UTC-день $START_DAY"
echo "[rollback] жду смены UTC-суток (сброс дневного лимита)..."
while [ "$(date -u +%Y-%m-%d)" = "$START_DAY" ]; do sleep 30; done
echo "[rollback] UTC-сутки сменились: $(date -u +%Y-%m-%d %H:%M) — лимит сброшен"
sleep 60   # небольшой запас, чтобы счётчик на стороне Kaggle точно провернулся

count_today() {
  "$KAGGLE" competitions submissions -c pokemon-tcg-ai-battle -v 2>/dev/null | \
    .venv/bin/python -c "
import sys,csv,datetime
d=datetime.datetime.utcnow().strftime('%Y-%m-%d')
r=list(csv.DictReader(sys.stdin))
print(sum(1 for x in r if x['date'][:10]==d))
"
}

submit_copy() {
  local label="$1" msg="$2"
  local before after
  before=$(count_today)
  echo "[rollback] $label: сабмитов за сегодня до отправки = $before"
  ./scripts/submit.sh v088_bc2b "$msg" 2>&1 | tail -2
  sleep 45
  after=$(count_today)
  echo "[rollback] $label: стало $after"
  if [ "$after" -gt "$before" ]; then
    echo "[rollback] $label: ЗАРЕГИСТРИРОВАНА"
    return 0
  fi
  echo "[rollback] $label: НЕ зарегистрирована"
  return 1
}

MSG_A="ROLLBACK to v088 (copy A, 08-16): v089 live-refuted — mirror WR 62%->27% (4W-11L, p=0.006), scores 767/772 vs 865/919. Restoring proven pilot before deadline."
MSG_B="ROLLBACK to v088 (copy B, 08-16): restores tracked-2 = {v088, v088}, evicting last v089 copy. Offline mirror-subclass fidelity +16.3pp did NOT transfer to game outcomes."

if submit_copy "копия A" "$MSG_A"; then
  if submit_copy "копия B" "$MSG_B"; then
    echo "[rollback] ГОТОВО: обе копии v088 залиты"
  else
    echo "[rollback] ВНИМАНИЕ: копия B не прошла — нужен ручной повтор"
  fi
else
  echo "[rollback] ОСТАНОВ: копия A не прошла, копию B не отправляю"
fi

echo "[rollback] итоговое состояние (2 верхние строки = tracked-2):"
"$KAGGLE" competitions submissions -c pokemon-tcg-ai-battle -v 2>/dev/null | \
  .venv/bin/python -c "
import sys,csv
r=list(csv.DictReader(sys.stdin))
for row in r[:4]:
    print(f\"  {row['ref']:>10} {row['fileName']:<18} {row['date'][:16]} {row['status'].split('.')[-1]:<10} score={row.get('publicScore') or '—'}\")
"
