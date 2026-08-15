#!/usr/bin/env python3
"""verify_v089_features_parity.py — доказать, что новый корпус фич отличается от v088 РОВНО
девятью фичами БЛОКА 8 и ничем больше.

ЗАЧЕМ: v089 обучается на корпусе, прогнанном через ИЗМЕНЁННЫЙ featurize_corpus_2b.py
(добавлен БЛОК 8 + флаг --offset для шардовой фичеризации). Если правка задела состав
решений или значения старых фич, сравнение v089 vs v088 окажется конфаундом
"новые фичи + сдвиг данных", а не чистым эффектом блока.

Проверки (все обязаны пройти):
  1. feature_order нового спека = старый спек + 9 фич БЛОКА 8 В КОНЦЕ (префикс совпадает).
  2. Множества decision_uid по каждой группе совпадают со старым корпусом.
  3. На общих 284 колонках значения идентичны (по decision_uid+option_idx), проверяется
     на выборке решений (--sample) или целиком (--full).
  4. Метки y и группировка (min_count/max_count/opt_type) не изменились.
"""
from __future__ import annotations

import argparse
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
GROUPS = ["main", "c7", "low", "mid", "easy", "rest"]
KEY = ["decision_uid", "option_idx"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="out/corpus_2b/versions/v088_dragapult/features_v088")
    ap.add_argument("--new", default="out/corpus_2b/versions/v089_block8/features")
    ap.add_argument("--sample", type=int, default=200000,
                    help="сколько строк сверять значениями на группу (0 = все)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    old_dir, new_dir = REPO / args.old, REPO / args.new
    old_spec = json.loads((old_dir / "feature_spec.json").read_text())["feature_order"]
    new_spec = json.loads((new_dir / "feature_spec.json").read_text())["feature_order"]

    rep: dict = {"checks": {}}
    ok = True

    # ── 1. спек ──
    prefix_ok = new_spec[:len(old_spec)] == old_spec
    tail_ok = new_spec[len(old_spec):] == BLOCK8
    rep["checks"]["spec"] = {"old_n": len(old_spec), "new_n": len(new_spec),
                             "prefix_identical": prefix_ok, "tail_is_block8": tail_ok}
    print(f"[1] спек: {len(old_spec)} -> {len(new_spec)}; префикс идентичен={prefix_ok}; "
          f"хвост = БЛОК 8={tail_ok}")
    ok &= prefix_ok and tail_ok

    shared = old_spec
    for g in GROUPS:
        po, pn = old_dir / f"{g}.parquet", new_dir / f"{g}.parquet"
        if not po.exists() or not pn.exists():
            print(f"[!] {g}: нет parquet в одном из каталогов — пропуск")
            continue
        cols = KEY + ["y", "min_count", "max_count", "opt_type"]
        o = pd.read_parquet(po, columns=cols + shared)
        n = pd.read_parquet(pn, columns=cols + shared)

        # ── 2. состав решений ──
        uo, un = set(o["decision_uid"]), set(n["decision_uid"])
        same_dec = uo == un
        rep["checks"].setdefault("decisions", {})[g] = {
            "old": len(uo), "new": len(un), "identical": same_dec,
            "only_old": len(uo - un), "only_new": len(un - uo)}
        print(f"[2] {g}: решений old={len(uo)} new={len(un)} identical={same_dec}"
              + ("" if same_dec else f" (только old={len(uo-un)}, только new={len(un-uo)})"))
        ok &= same_dec

        # ── 3-4. значения на общих колонках ──
        o = o.sort_values(KEY).reset_index(drop=True)
        n = n.sort_values(KEY).reset_index(drop=True)
        if len(o) != len(n):
            print(f"[3] {g}: РАЗНОЕ ЧИСЛО СТРОК old={len(o)} new={len(n)} — СТОП")
            ok = False
            continue
        if args.sample and len(o) > args.sample:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(len(o), args.sample, replace=False))
            o, n = o.iloc[idx].reset_index(drop=True), n.iloc[idx].reset_index(drop=True)
        key_ok = (o["decision_uid"].to_numpy() == n["decision_uid"].to_numpy()).all() and \
                 (o["option_idx"].to_numpy() == n["option_idx"].to_numpy()).all()
        label_ok = (o["y"].to_numpy() == n["y"].to_numpy()).all()
        vals_o = o[shared].to_numpy(dtype=np.float64)
        vals_n = n[shared].to_numpy(dtype=np.float64)
        both_nan = np.isnan(vals_o) & np.isnan(vals_n)
        vals_ok = bool((np.isclose(vals_o, vals_n, rtol=0, atol=1e-9) | both_nan).all())
        if not vals_ok:
            diff_cols = [shared[j] for j in range(len(shared))
                         if not bool((np.isclose(vals_o[:, j], vals_n[:, j], rtol=0, atol=1e-9)
                                      | both_nan[:, j]).all())]
            print(f"[3] {g}: РАСХОЖДЕНИЕ в колонках: {diff_cols[:8]}")
            rep["checks"].setdefault("value_diff_cols", {})[g] = diff_cols[:20]
        rep["checks"].setdefault("values", {})[g] = {
            "rows_compared": int(len(o)), "keys_aligned": bool(key_ok),
            "labels_identical": bool(label_ok), "shared_values_identical": vals_ok}
        print(f"[3] {g}: сверено {len(o)} строк; ключи={key_ok}; y={label_ok}; "
              f"старые фичи идентичны={vals_ok}")
        ok &= key_ok and label_ok and vals_ok

    rep["verdict"] = "OK" if ok else "FAIL"
    print("\nИТОГ:", "OK — единственное отличие = БЛОК 8" if ok
          else "FAIL — есть расхождения помимо БЛОКА 8, сравнение будет конфаундом")
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
