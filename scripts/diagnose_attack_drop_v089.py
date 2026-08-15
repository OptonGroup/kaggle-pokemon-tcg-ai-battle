#!/usr/bin/env python3
"""diagnose_attack_drop_v089.py — определить ПРИРОДУ просадки attack у v089.

ВОПРОС. v089 (БЛОК 8 + вес 1.6) даёт подкласс +8.85пп, но attack −1.08пп. Две
несовместимые интерпретации:
  (A) РАЗМЕН (нулевая сумма, v088_RESULT §4): просадка сидит ТОЛЬКО в состояниях, где
      ретрит был реальной альтернативой атаке. top-1 весит обе одинаково, хотя пропущенный
      защитный ретрит стоит −2 приза => метрика штрафует правку, полезную по игре.
  (B) УЩЕРБ: просадка размазана и по решениям, где ретрит вообще не рассматривался
      (нет здоровой копии на бенче) => БЛОК 8 портит несвязанные решения, правку брать нельзя.

МЕТОД. Attack-решения холдаута режутся по НАБЛЮДАЕМОМУ признаку "ретрит был альтернативой":
own_bench_healthier_same_exists==1 (на бенче есть более здоровая копия того же вида).
Признак берётся из НОВЫХ фич, но он описывает СОСТОЯНИЕ БОРДА, не решение модели, и
проверен на 100% совпадение с сырыми obs (ВАЛИДАЦИЯ 2 в eval_holdout_v089.py).

ЧТЕНИЕ РЕЗУЛЬТАТА. Если в страте "альтернативы не было" дельта ≈ 0 (|z|<2), просадка —
это (A) размен. Если и там значимый минус — это (B), правку чинить/отклонять.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def load_trainer():
    path = REPO / "scripts" / "train_bc_2b_v088.py"
    spec = importlib.util.spec_from_file_location("trainer_v088", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trainer_v088"] = mod
    spec.loader.exec_module(mod)
    return mod


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = a.astype(bool), b.astype(bool)
    b10, b01 = int((a & ~b).sum()), int((~a & b).sum())
    n = b10 + b01
    if n == 0:
        return {"delta_pp": 0.0, "n_discordant": 0, "z": None}
    z = (b10 - b01) / math.sqrt(n)
    return {"delta_pp": round(100.0 * (b10 - b01) / len(a), 3), "n_discordant": n,
            "z": round(z, 3), "p_two_sided": round(math.erfc(abs(z) / math.sqrt(2)), 6)}


def score(T, features_dir: Path, model_dir: Path, group="main"):
    spec = json.loads((features_dir / "feature_spec.json").read_text())
    fo = spec["feature_order"]
    df = T.load_group_df(features_dir, group, fo)
    X = df[fo].to_numpy(dtype=np.float32)
    meta = df[["decision_uid", "option_idx", "y", "min_count", "max_count",
               "opt_type", "opt_card_id"]]
    b = lgb.Booster(model_file=str(model_dir / f"{group}.txt"))
    dec = T.decision_level_eval(meta, b.predict(X, num_iteration=b.best_iteration or None),
                                with_category=True, with_multiset=False)
    return dec, df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-features",
                    default="out/corpus_2b/versions/v088_dragapult/holdout_features")
    ap.add_argument("--new-features",
                    default="out/corpus_2b/versions/v089_block8/holdout_features")
    ap.add_argument("--model-v088",
                    default="out/corpus_2b/versions/v088_dragapult/final/B_k1k2_16")
    ap.add_argument("--model-v089", required=True, action="append", metavar="LABEL=DIR")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    T = load_trainer()
    old_dir, new_dir = REPO / args.old_features, REPO / args.new_features

    dec88, df_old = score(T, old_dir, REPO / args.model_v088)
    base = dec88[["decision_uid", "category"]].copy()
    base["m_v088"] = dec88["model_match"].to_numpy()

    # признак "ретрит был альтернативой" — состояние борда из новых фич
    feat = "own_bench_healthier_same_exists"
    _, df_new = score(T, new_dir, REPO / args.model_v089[0].split("=", 1)[1])
    fmeta = df_new[["decision_uid", feat]].groupby("decision_uid", sort=False).first()
    base = base.join(fmeta, on="decision_uid")

    cands = {}
    for spec_s in args.model_v089:
        label, d = spec_s.split("=", 1)
        dec, _ = score(T, new_dir, REPO / d)
        cands[label] = dict(zip(dec["decision_uid"].to_numpy(), dec["model_match"].to_numpy()))
        base[f"m_{label}"] = base["decision_uid"].map(cands[label]).to_numpy()

    atk = base[base["category"] == "attack"]
    alt_yes = atk[atk[feat] == 1.0]
    alt_no = atk[atk[feat] == 0.0]

    print(f"\nattack-решений всего: {len(atk)}; "
          f"ретрит был альтернативой: {len(alt_yes)}; не было: {len(alt_no)}\n")
    print(f"{'кандидат':<14} {'страта':<34} {'n':>7} {'v088':>8} {'cand':>8} "
          f"{'дельта':>10} {'z':>8}")
    print("-" * 92)
    rep = {"n_attack": int(len(atk)), "n_alt_yes": int(len(alt_yes)),
           "n_alt_no": int(len(alt_no)), "candidates": {}}
    for label in cands:
        rows = []
        for name, sub in (("ретрит БЫЛ альтернативой", alt_yes),
                          ("ретрита НЕ было (контроль)", alt_no),
                          ("attack целиком", atk)):
            m88 = sub["m_v088"].to_numpy()
            mc = sub[f"m_{label}"].to_numpy()
            d = mcnemar(mc, m88)
            print(f"{label:<14} {name:<34} {len(sub):>7} {m88.mean():>8.4f} "
                  f"{mc.mean():>8.4f} {d['delta_pp']:>+9.2f}пп {str(d['z']):>8}")
            rows.append({"slice": name, "n": int(len(sub)),
                         "v088": round(float(m88.mean()), 4),
                         "cand": round(float(mc.mean()), 4), **d})
        rep["candidates"][label] = rows
        print("-" * 92)

    print("\nЧТЕНИЕ: если в страте «ретрита НЕ было» дельта ≈ 0 (|z|<2) — просадка attack "
          "есть РАЗМЕН\nс ретритом (нулевая сумма), а не ущерб несвязанным решениям.")
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
