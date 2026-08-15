#!/usr/bin/env python3
"""Состязательная проверка правки v088 (retreat_state_weight_for) ДО долгого прогона.

Проверяется на РЕАЛЬНОМ срезе корпуса (не на синтетике), потому что цена ошибки —
испорченный многочасовой ретрейн+гейт:
  T1 no-op:      w_global == w_crit == 1.0  -> None (веса в lgb.Dataset не подаются вовсе,
                 прогон обязан быть байт-в-байт равен v087).
  T2 вырождение: w_crit == w_global == 2.0  -> ПОБИТОВО равно retreat_weight_for(2.0)
                 (контракт обратной совместимости из докстринга).
  T3 семантика:  w_global=1.5, w_crit=3.0 -> ровно три уровня весов; вес 3.0 стоит ТОЛЬКО
                 на строках retreat-решений в критическом состоянии; вес 1.5 — на остальных
                 retreat-решениях; 1.0 — на всех прочих; внутри одного decision_uid вес
                 КОНСТАНТЕН (иначе lambdarank получил бы рваный список).
  T4 маска:      критерий критичности совпадает с независимым пересчётом по y==1-строкам.

Запуск: .venv/bin/python tests/test_retreat_state_weight.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
COLS = ["decision_uid", "y", "opt_type", "active_hp_frac", "bench_energy_sum", "turn_idx",
        "split", "episode_id"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    v086 = load_module("t_v086", REPO / "scripts" / "train_bc_2b_v086.py")
    v088 = load_module("t_v088", REPO / "scripts" / "train_bc_2b_v088.py")

    pq = REPO / "out/corpus_2b/versions/v086_targeted/features_v086/main.parquet"
    df = pd.read_parquet(pq, columns=COLS)
    # берём срез, гарантированно содержащий retreat-решения и оба состояния доски
    eps = df["episode_id"].drop_duplicates().to_numpy()[:1500]
    df = df[df["episode_id"].isin(eps)].reset_index(drop=True)
    mask = (df["split"] == "train").to_numpy()
    n_ret = int((df.loc[mask & (df.y == 1).to_numpy(), "opt_type"] == v088.OPT_RETREAT).sum())
    assert n_ret >= 50, f"срез слишком беден на retreat ({n_ret}) — тест не показателен"
    print(f"[test] срез: {len(df)} строк, {df.decision_uid.nunique()} решений, "
          f"{n_ret} retreat-решений")

    HP, BE, TU = 0.6, 2.0, 10.0
    fails = []

    # T1 — no-op
    if v088.retreat_state_weight_for(df, mask, 1.0, 1.0, HP, BE, TU) is not None:
        fails.append("T1: w_global=w_crit=1.0 обязан давать None (no-op), получено не-None")
    else:
        print("[test] T1 no-op: OK")

    # T2 — вырождение в retreat_weight_for
    ref = v086.retreat_weight_for(df, mask, 2.0)
    got = v088.retreat_state_weight_for(df, mask, 2.0, 2.0, HP, BE, TU)
    if ref is None or got is None:
        fails.append("T2: одна из функций вернула None на weight=2.0")
    elif not np.array_equal(ref, got):
        fails.append(f"T2: расхождение с retreat_weight_for на {int((ref != got).sum())} строках")
    else:
        print(f"[test] T2 вырождение: OK (побитово равно на {len(ref)} строках, "
              f"взвешено {int((ref != 1.0).sum())})")

    # T3 — три уровня, вес константен внутри решения
    w = v088.retreat_state_weight_for(df, mask, 1.5, 3.0, HP, BE, TU)
    levels = sorted(set(np.unique(w).tolist()))
    if levels != [1.0, 1.5, 3.0]:
        fails.append(f"T3: ожидались уровни [1.0, 1.5, 3.0], получены {levels}")
    sub = df.loc[mask, ["decision_uid"]].copy()
    sub["w"] = w
    per_dec = sub.groupby("decision_uid")["w"].nunique()
    if int((per_dec > 1).sum()) != 0:
        fails.append(f"T3: вес НЕ константен внутри {int((per_dec > 1).sum())} решений")
    if not fails:
        n3 = int((w == 3.0).sum())
        n15 = int((w == 1.5).sum())
        print(f"[test] T3 уровни: OK (crit-строк {n3}, обычных retreat-строк {n15}, "
              f"прочих {int((w == 1.0).sum())})")

    # T4 — независимый пересчёт критической маски по решениям
    s = df.loc[mask].copy()
    s["w"] = w
    true_rows = s[s.y == 1]
    ret_uids = set(true_rows.loc[true_rows.opt_type == v088.OPT_RETREAT, "decision_uid"])
    dec = s.groupby("decision_uid").first()
    exp_crit = {u for u in ret_uids
                if (dec.at[u, "active_hp_frac"] <= HP and dec.at[u, "bench_energy_sum"] >= BE
                    and dec.at[u, "turn_idx"] >= TU)}
    got_crit = set(s.loc[s.w == 3.0, "decision_uid"])
    if exp_crit != got_crit:
        fails.append(f"T4: критическая маска расходится: ожидалось {len(exp_crit)}, "
                     f"получено {len(got_crit)}, симм.разница {len(exp_crit ^ got_crit)}")
    else:
        print(f"[test] T4 маска: OK ({len(exp_crit)} критических retreat-решений из "
              f"{len(ret_uids)} retreat, доля {len(exp_crit)/max(1,len(ret_uids)):.3f})")

    if fails:
        print("\nПРОВАЛ:", file=sys.stderr)
        for f in fails:
            print("  - " + f, file=sys.stderr)
        return 1
    print("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
