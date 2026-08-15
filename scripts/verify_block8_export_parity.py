#!/usr/bin/env python3
"""verify_block8_export_parity.py — прицельная проверка порта БЛОКА 8 в LIVE-артефакт.

ЗАЧЕМ ОТДЕЛЬНО ОТ ТЕСТА ЭКСПОРТА. `export_bc_2b.py` сверяет сгенерированный
bc_features_2b.py с parquet на СЛУЧАЙНОЙ выборке решений; при доступной памяти Docker-VM
она ограничена ~500 строками, где БЛОК 8 нетривиален лишь в малой доле. Здесь выборка
СМЕЩЕНА в сторону механики: берутся решения, где на бенче ЕСТЬ более здоровая копия того
же вида (предикат БЛОКА 8 == 1), плюс контрольные, где её нет. Это прямая проверка того,
что ручной порт `bench_replacement_features` в _FEATURES_TEMPLATE не разошёлся с
`featurize_corpus_2b.py`.

Артефакт bc_features_2b.py — чистый Python со вшитыми таблицами (cg НЕ импортирует),
поэтому скрипт работает на macOS-хосте, без Docker и без движка.

Сверяются ВСЕ 293 фичи (не только БЛОК 8): расхождение в любой из них ломает live.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
BLOCK8 = ["own_bench_same_id_n", "own_bench_best_same_hp", "own_bench_best_same_hp_frac",
          "own_bench_best_same_hp_minus_active", "own_bench_best_same_energy_n",
          "own_bench_best_same_attack_ready", "own_bench_best_same_max_damage",
          "own_bench_healthier_same_exists", "own_bench_charged_healthier_same_exists"]


def load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default="out/corpus_2b/versions/v089_block8/export")
    ap.add_argument("--features-dir", default="out/corpus_2b/versions/v089_block8/features")
    ap.add_argument("--obs-dir",
                    default="out/corpus_2b/versions/v088_dragapult/merged/obs_shards")
    ap.add_argument("--n-positive", type=int, default=300, help="решений с предикатом==1")
    ap.add_argument("--n-control", type=int, default=150, help="решений с предикатом==0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Праймер стейта БЛОКА 2 берём из самого экспортёра: живой модуль накапливает буфер
    # истории хода между вызовами, а выборка здесь — вразнобой по эпизодам/ходам. Без
    # прайминга фичи prev1..prev3_* закономерно расходятся, и это НЕ дефект порта
    # (проверено: без прайминга расходился только prev3_card_id, БЛОК 8 = 0.0).
    exporter = load_mod(REPO / "scripts" / "export_bc_2b.py", "export_bc_2b_forparity")
    feat_mod = load_mod(REPO / args.export_dir / "bc_features_2b.py", "bc_features_2b_v089")
    fo = feat_mod.FEATURE_ORDER
    print(f"[parity] артефакт: {len(fo)} фич")

    fdir = REPO / args.features_dir
    # читаем ТОЛЬКО служебные колонки + предикат (predicate pushdown по группе main)
    idx = pd.read_parquet(fdir / "main.parquet",
                          columns=["decision_uid", "option_idx",
                                   "own_bench_healthier_same_exists"])
    per_dec = idx.groupby("decision_uid")["own_bench_healthier_same_exists"].first()
    pos = list(per_dec.index[per_dec == 1.0])
    neg = list(per_dec.index[per_dec == 0.0])
    rng = np.random.default_rng(42)
    take_pos = list(rng.choice(pos, min(args.n_positive, len(pos)), replace=False))
    take_neg = list(rng.choice(neg, min(args.n_control, len(neg)), replace=False))
    chosen = take_pos + take_neg
    print(f"[parity] решений с предикатом==1: {len(pos)} (беру {len(take_pos)}); "
          f"==0: {len(neg)} (беру {len(take_neg)})")

    rows = pd.read_parquet(fdir / "main.parquet",
                           filters=[("decision_uid", "in", chosen)])
    rows = rows.sort_values(["decision_uid", "option_idx"])

    obs_dir = REPO / args.obs_dir
    by_ep: dict[int, list] = {}
    for uid in chosen:
        eid, ai, ref = uid.split(":")
        by_ep.setdefault(int(eid), []).append((uid, int(ref)))
    obs_by_uid = {}
    for eid, items in by_ep.items():
        shard = obs_dir / f"{eid}.jsonl"
        if not shard.exists():
            continue
        needed = {r for _u, r in items}
        got = {}
        with shard.open() as fh:
            for i, line in enumerate(fh):
                if i in needed:
                    got[i] = json.loads(line)
                    if len(got) == len(needed):
                        break
        for uid, ref in items:
            if ref in got:
                obs_by_uid[uid] = got[ref]

    max_diff = 0.0
    worst = None
    n_rows = n_dec = 0
    b8_active = 0
    per_feat_max = {f: 0.0 for f in fo}
    for uid, grp in rows.groupby("decision_uid", sort=False):
        obs = obs_by_uid.get(uid)
        if obs is None:
            continue
        n_dec += 1
        options = (obs.get("select") or {}).get("option") or []
        if grp["own_bench_healthier_same_exists"].iloc[0] == 1.0:
            b8_active += 1
        for _, r in grp.iterrows():
            oi = int(r["option_idx"])
            if oi >= len(options):
                continue
            exporter._prime_turn_history(feat_mod, obs, r)
            vec = feat_mod.build_features(obs, options[oi], oi, len(options))
            ref_vec = [float(r[f]) for f in fo]
            for j, f in enumerate(fo):
                d = abs(float(vec[j]) - ref_vec[j])
                if d > per_feat_max[f]:
                    per_feat_max[f] = d
                if d > max_diff:
                    max_diff, worst = d, (uid, oi, f, float(vec[j]), ref_vec[j])
            n_rows += 1

    print(f"[parity] сверено решений: {n_dec} (из них с активным БЛОКОМ 8: {b8_active}), "
          f"строк: {n_rows}")
    print(f"[parity] max_abs_diff по ВСЕМ {len(fo)} фичам: {max_diff}")
    b8_max = max(per_feat_max[f] for f in BLOCK8)
    print(f"[parity] max_abs_diff по фичам БЛОКА 8: {b8_max}")
    if worst:
        print(f"[parity] худшее расхождение: uid={worst[0]} opt={worst[1]} фича={worst[2]} "
              f"артефакт={worst[3]} parquet={worst[4]}")
    ok = max_diff == 0.0
    print("\nИТОГ:", "OK — порт БЛОКА 8 в live-артефакт побитово совпал с фичеризатором"
          if ok else "FAIL — артефакт расходится с фичеризатором, live сломается")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"n_decisions": n_dec, "n_rows": n_rows, "n_block8_active": b8_active,
             "max_abs_diff_all": max_diff, "max_abs_diff_block8": b8_max,
             "verdict": "OK" if ok else "FAIL"}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
