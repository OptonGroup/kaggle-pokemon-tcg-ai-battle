#!/usr/bin/env python3
"""concat_features_v088.py — склейка parquet-таблиц базового корпуса и целевого расширения
(K2 dragapult), без пере-featurize базы (та не менялась — экономит ~40 мин).

Инварианты, которые скрипт ПРОВЕРЯЕТ (а не предполагает):
  1. feature_spec.json обеих сторон идентичен (feature_order + deck19) — иначе колонки
     означают разное и склейка молча испортит корпус;
  2. набор колонок совпадает (и порядок приводится к базовому);
  3. episode_id базы и расширения не пересекаются;
  4. ВСЕ строки расширения имеют split=='train' — новые эпизоды никогда не входят в
     замороженный test (тот же инвариант, что защищает merge_corpus_2b_daily_extension.py);
  5. decision_uid уникальны между сторонами (иначе contiguous_bounds склеит чужие решения).

Пример:
  .venv/bin/python scripts/concat_features_v088.py \
    --base out/corpus_2b/versions/v086_targeted/features_v086 \
    --ext  out/corpus_2b/versions/v088_dragapult/ext_features \
    --out  out/corpus_2b/versions/v088_dragapult/features_v088
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

GROUPS = ["main", "c7", "low", "mid", "easy", "rest"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", nargs="+", default=GROUPS, choices=GROUPS)
    args = ap.parse_args()

    base, ext, out = Path(args.base), Path(args.ext), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sb = json.loads((base / "feature_spec.json").read_text())
    se = json.loads((ext / "feature_spec.json").read_text())
    if sb["feature_order"] != se["feature_order"]:
        sys.exit("FATAL: feature_order базы и расширения различаются — склейка запрещена")
    if sb.get("deck19") != se.get("deck19"):
        sys.exit("FATAL: deck19 базы и расширения различаются — склейка запрещена")
    shutil.copyfile(base / "feature_spec.json", out / "feature_spec.json")

    report = {}
    for g in args.groups:
        pb, pe = base / f"{g}.parquet", ext / f"{g}.parquet"
        if not pb.exists():
            print(f"[concat] {g}: нет базового {pb} — пропуск", file=sys.stderr)
            continue
        db = pd.read_parquet(pb)
        if not pe.exists():
            print(f"[concat] {g}: расширения нет, копирую базу как есть", file=sys.stderr)
            db.to_parquet(out / f"{g}.parquet", index=False)
            report[g] = {"base_rows": len(db), "ext_rows": 0, "out_rows": len(db)}
            continue
        de = pd.read_parquet(pe)
        if set(db.columns) != set(de.columns):
            only_b = sorted(set(db.columns) - set(de.columns))[:5]
            only_e = sorted(set(de.columns) - set(db.columns))[:5]
            sys.exit(f"FATAL: {g}: набор колонок различается (только в базе {only_b}, "
                     f"только в расширении {only_e})")
        de = de[db.columns]
        bad_split = de.loc[de["split"] != "train", "split"].unique().tolist()
        if bad_split:
            sys.exit(f"FATAL: {g}: в расширении есть строки со split != train: {bad_split}")
        eb = set(db["episode_id"].unique().tolist())
        ee = set(de["episode_id"].unique().tolist())
        if eb & ee:
            sys.exit(f"FATAL: {g}: пересечение эпизодов базы и расширения "
                     f"({len(eb & ee)}, напр. {sorted(eb & ee)[:5]})")
        ub = set(db["decision_uid"].unique().tolist())
        ue = set(de["decision_uid"].unique().tolist())
        if ub & ue:
            sys.exit(f"FATAL: {g}: коллизия decision_uid ({len(ub & ue)})")
        dc = pd.concat([db, de], ignore_index=True)
        dc.to_parquet(out / f"{g}.parquet", index=False)
        report[g] = {"base_rows": len(db), "ext_rows": len(de), "out_rows": len(dc),
                     "base_decisions": len(ub), "ext_decisions": len(ue),
                     "ext_train_share": 1.0}
        print(f"[concat] {g}: {len(db)} + {len(de)} = {len(dc)} строк "
              f"({len(ub)} + {len(ue)} решений)", file=sys.stderr)

    (out / "concat_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
