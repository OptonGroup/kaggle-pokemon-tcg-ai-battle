#!/usr/bin/env bash
# featurize_sharded_v089.sh — фичеризация корпуса/холдаута чанками по эпизодам + склейка.
#
# ЗАЧЕМ ЧАНКИ: featurize_corpus_2b.py держит все option-строки в памяти до записи parquet.
# Полный корпус 2B (1.04M решений -> ~5.6M строк x 293 фичи) не влезает в Docker-VM (7.75 ГиБ):
# прогон падал с OOM (exit 137) примерно на 700-м эпизоде из 10635. Чанк ~1800 эпизодов
# (~1M строк) держится с запасом. Эквивалентность чанков сплошному прогону проверена
# (decision_uid не пересекаются, значения фич идентичны).
#
# ЗАЧЕМ АБСОЛЮТНЫЙ ПУТЬ МОНТИРОВАНИЯ: obs_shards в merged/ — симлинки с АБСОЛЮТНЫМИ целями
# (/Users/max/...). Монтирование в /w их ломает; монтируем репо по его же пути.
set -euo pipefail

REPO=/Users/max/kaggle-pokemon-tcg-ai-battle
IMG=poke-featurize:pa
DEC="$1"        # decisions.jsonl
OBS="$2"        # obs_shards
SPLIT="$3"      # split.json
OUT="$4"        # итоговый каталог фич
TOTAL="$5"      # всего эпизодов
STEP="${6:-1800}"

cd "$REPO"
mkdir -p "$OUT"
CHUNKS=()
off=0
while [ "$off" -lt "$TOTAL" ]; do
  cdir="$OUT/_chunk_$off"
  if [ -f "$cdir/feature_spec.json" ]; then
    echo "[shard] offset=$off уже готов, пропуск"
  else
    echo "[shard] offset=$off limit=$STEP -> $cdir"
    docker run --rm --platform linux/amd64 -v "$REPO":"$REPO" -w "$REPO" "$IMG" \
      python scripts/featurize_corpus_2b.py \
      --decisions "$DEC" --obs-dir "$OBS" --split "$SPLIT" \
      --out-dir "$cdir" --offset "$off" --limit "$STEP" \
      > "$OUT/_chunk_$off.log" 2>&1
  fi
  CHUNKS+=("$cdir")
  off=$((off + STEP))
done

echo "[shard] склейка ${#CHUNKS[@]} чанков -> $OUT"
"$REPO/.venv/bin/python" - "$OUT" "${CHUNKS[@]}" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd

out = Path(sys.argv[1]); chunks = [Path(p) for p in sys.argv[2:]]
groups = ["main", "c7", "low", "mid", "easy", "rest"]

# Спек берётся ЦЕЛИКОМ из первого чанка (он содержит n_features/deck19/sentinel/
# id_columns/meta_columns — их ждёт export_bc_2b.py; урезанный спек с одним feature_order
# роняет экспорт на KeyError 'n_features'). Все поля спека — константы кода, не данных,
# поэтому идентичность между чанками проверяется целиком, а не только по feature_order.
full_specs = [json.loads((c / "feature_spec.json").read_text()) for c in chunks]
assert all(s == full_specs[0] for s in full_specs), "feature_spec разъехался между чанками"
(out / "feature_spec.json").write_text(json.dumps(full_specs[0], ensure_ascii=False, indent=2))
specs = [s["feature_order"] for s in full_specs]
report = {}
for g in groups:
    parts = [pd.read_parquet(c / f"{g}.parquet") for c in chunks if (c / f"{g}.parquet").exists()]
    if not parts:
        continue
    df = pd.concat(parts, ignore_index=True)
    uids = df["decision_uid"]
    n_dec = uids.nunique()
    # инвариант: чанки режут по эпизодам, поэтому решение целиком лежит в одном чанке
    assert sum(len(p["decision_uid"].unique()) for p in parts) == n_dec, \
        f"{g}: decision_uid пересеклись между чанками — окна рвут решения"
    df.to_parquet(out / f"{g}.parquet", index=False)
    report[g] = {"rows": int(len(df)), "decisions": int(n_dec), "chunks": len(parts)}
    print(f"  {g}: {len(df)} строк / {n_dec} решений из {len(parts)} чанков")
(out / "shard_concat_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
PY
echo "[shard] ГОТОВО: $OUT"
