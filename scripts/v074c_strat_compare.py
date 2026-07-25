"""v074c_strat_compare.py — стратифицированное сравнение двух арм гейта по стратам матчапов.

Обобщение scripts/v074_strat_meta.py (тот захардкожен под P0/Full Metal Lab). Читает
results/gates/{label}_core_gate.json обеих арм, складывает повторы внутри армы по каждой паре,
затем объединяет пары как НЕЗАВИСИМЫЕ СТРАТЫ с обратно-дисперсионными весами.

Почему не пулинг сырых счётчиков: пары — страты с WR от 0.26 (lucario) до 0.98 (v017), и при
разном числе прогонов на пару пулинг даёт парадокс Симпсона (на прогоне v074 он выдал
фиктивные −1.14 пп при p=0.0005 там, где код заведомо идентичен). Память проекта:
project_gate_stratified_measurement.

Каждая правка меряется на СВОЕЙ страте плюс на КОНТРОЛЬНОЙ — парах, где изменённая ветка
физически недостижима. Контроль обязан давать ноль; если он даёт значимую дельту, весь прогон
— шум, и вывод по целевой страте не читается.

Запуск (PTCG_ROOT — чекаут, где лежат results/gates):
    PTCG_ROOT=/path/to/repo .venv/bin/python scripts/v074c_strat_compare.py \
        --cand v074c_r1 v074c_r2 --base v074_r5 v074_r6 \
        --cand-name v074c --base-name v074_fullmetal --stratum crustle
"""
from __future__ import annotations

import argparse
import json
import math
import os

ROOT = os.environ.get("PTCG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATES = os.path.join(ROOT, "results", "gates")

# Страты = пары, на которых изменённая ветка МОЖЕТ сработать. Остальные пары автоматически
# становятся контролем для этой страты.
STRATA = {
    # P1 (Rock-Inn demote): ветка требует Crustle 345 в активе соперника.
    "crustle": ["v009a", "ext_tusk_crustle"],
    # P0 (Premium Power Pro): требует {F}-атакующего и копию 1141 в сбросе.
    "lucario": ["ext_maulikgajera_lucario", "ext_nursrijan_lucario"],
    # P3 (Full Metal Lab): деки с картой 1244.
    "archaludon": ["v044", "v036_h2h", "ext_masamikobayashi_archaludon"],
    "alakazam": ["v017", "v016", "ext_prvsiyan_alak_v8", "ext_prvsiyan_alak_v9"],
    "mirror": ["v063_h2h", "v070_h2h", "v072_h2h", "v073_h2h", "v065_h2h"],
}


def two_prop_ztest(x1: float, n1: float, x2: float, n2: float) -> tuple[float, float]:
    if n1 <= 0 or n2 <= 0:
        return (0.0, 1.0)
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(max(p_pool * (1 - p_pool), 1e-12) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    return (z, math.erfc(abs(z) / math.sqrt(2)))


def load_arm(labels: list[str]) -> tuple[dict, list[str]]:
    per_key: dict[str, list] = {}
    missing = []
    for lab in labels:
        path = os.path.join(GATES, f"{lab}_core_gate.json")
        if not os.path.exists(path):
            missing.append(lab)
            continue
        with open(path) as f:
            data = json.load(f)
        for key, v in (data.get("opponents") or {}).items():
            n = v.get("n_games") or 0
            if not n:
                continue
            wins = v.get("wins")
            draws = v.get("draws") or 0
            if wins is None:
                wins = (v.get("win_rate") or 0.0) * n
                draws = 0
            per_key.setdefault(key, []).append(
                (lab, float(wins) + 0.5 * float(draws), float(n)))
    return per_key, missing


def pooled(per_key: dict, key: str):
    runs = per_key.get(key) or []
    if not runs:
        return None
    w = sum(x for _, x, _ in runs)
    n = sum(x for _, _, x in runs)
    return w, n, [x / y for _, x, y in runs]


def strat_meta(rows: list[tuple[str, float, float, float, float]], label: str) -> dict:
    """rows = [(key, cand_wr, cand_n, base_wr, base_n)]. Фиксированный эффект,
    обратно-дисперсионные веса."""
    num = den = 0.0
    used = []
    for key, p1, n1, p2, n2 in rows:
        var = p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
        if var <= 0:
            continue
        w = 1.0 / var
        num += w * (p1 - p2)
        den += w
        used.append((key, 100.0 * (p1 - p2), n1, n2))
    if den == 0:
        print(f"  {label}: нет данных")
        return {}
    d = num / den
    se = math.sqrt(1.0 / den)
    z = d / se
    pv = math.erfc(abs(z) / math.sqrt(2))
    print(f"  {label}")
    for key, dd, n1, n2 in used:
        print(f"      {key:34s} Δ={dd:+6.2f} пп   (n={n1:.0f} vs {n2:.0f})")
    print(f"      {'ИТОГО стратифицированно':34s} Δ={100*d:+6.2f} пп   "
          f"95%CI [{100*(d-1.96*se):+.2f}, {100*(d+1.96*se):+.2f}]   p={pv:.4f}"
          f"{'   ЗНАЧИМО' if pv < 0.05 else ''}")
    return dict(delta_pp=100 * d, ci_lo=100 * (d - 1.96 * se), ci_hi=100 * (d + 1.96 * se),
                p=pv, pairs=[dict(key=k, delta_pp=dd, cand_n=n1, base_n=n2)
                             for k, dd, n1, n2 in used])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--cand-name", default="cand")
    ap.add_argument("--base-name", default="base")
    ap.add_argument("--stratum", default=None,
                    help="имя страты из STRATA; остальные пары = контроль")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cand, miss_c = load_arm(args.cand)
    base, miss_b = load_arm(args.base)
    for lab in miss_c + miss_b:
        print(f"[warn] нет прогона: {lab}_core_gate.json")

    keys = sorted(k for k in cand if k in base)
    print(f"\n===== {args.cand_name} ({len(args.cand)} прогонов) против "
          f"{args.base_name} ({len(args.base)} прогонов) =====")
    print(f"{'пара':34s} {args.cand_name:>10s} {args.base_name:>10s} {'Δпп':>8s} {'p':>8s}   "
          f"прогоны кандидата / базы")
    print("-" * 118)
    rows = []
    table = []
    for k in keys:
        cw, cn, cruns = pooled(cand, k)
        bw, bn, bruns = pooled(base, k)
        z, p = two_prop_ztest(cw, cn, bw, bn)
        c_wr, b_wr = cw / cn, bw / bn
        d = 100.0 * (c_wr - b_wr)
        flag = "  ЗНАЧИМО" if p < 0.05 else ""
        print(f"{k:34s} {c_wr:10.4f} {b_wr:10.4f} {d:+8.2f} {p:8.4f}   "
              f"{' '.join(f'{x:.3f}' for x in cruns)} / "
              f"{' '.join(f'{x:.3f}' for x in bruns)}{flag}")
        rows.append((k, c_wr, cn, b_wr, bn))
        table.append(dict(key=k, cand_wr=c_wr, cand_n=cn, base_wr=b_wr, base_n=bn,
                          delta_pp=d, p=p, cand_runs=cruns, base_runs=bruns))
    print("-" * 118)

    out = dict(cand=args.cand, base=args.base, pairs=table, strata={})
    if args.stratum:
        target = [k for k in STRATA.get(args.stratum, []) if k in dict(
            (r[0], 1) for r in rows)]
        target_set = set(target)
        print(f"\nСТРАТИФИЦИРОВАННО (обратно-дисперсионные веса, страта '{args.stratum}'):")
        out["strata"]["target"] = strat_meta(
            [r for r in rows if r[0] in target_set],
            f"целевая страта {args.stratum}: {', '.join(target) or '(пусто)'}")
        out["strata"]["control"] = strat_meta(
            [r for r in rows if r[0] not in target_set],
            "КОНТРОЛЬ (пары, где изменённая ветка недостижима — обязан быть нулём)")
    else:
        print("\nСТРАТИФИЦИРОВАННО по всем парам:")
        out["strata"]["all"] = strat_meta(rows, "все пары")

    if args.out:
        path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
