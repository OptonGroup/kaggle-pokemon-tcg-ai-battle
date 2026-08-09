"""v074_gate_aggregate.py — пулинг нескольких прогонов гейта по паре арм и z-тест.

Зачем: стенд НЕВОСПРОИЗВОДИМ (движок не принимает сид), шумовой пол ±1-2 пп на пару при
N=1000 (ловушка 2 из docs/gold/PLAN_v074.md). Поэтому вердикт по правке даётся не по
одному прогону, а по пулу независимых повторов ОБЕИХ арм, прогнанных в одних условиях.

Складывает wins (+0.5*draw) и n_games по каждому ключу пары внутри армы, печатает
по-прогонные WR, пул и двухпропорционный z-тест кандидат-vs-базa.

Запуск:
    .venv/bin/python scripts/v074_gate_aggregate.py \
        --cand v074 v074_r2 v074_r3 v074_r4 --base v073 v073_paired v073_rB v073_rC v073_rD
"""
from __future__ import annotations

import argparse
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATES = os.path.join(ROOT, "results", "gates")


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


def load_arm(labels: list[str]) -> tuple[dict, dict]:
    """-> ({key: [(label, wins, n), ...]}, {label: missing_path|None})"""
    per_key: dict[str, list] = {}
    missing: dict[str, str] = {}
    for lab in labels:
        path = os.path.join(GATES, f"{lab}_core_gate.json")
        if not os.path.exists(path):
            missing[lab] = path
            continue
        with open(path) as f:
            data = json.load(f)
        for key, v in (data.get("opponents") or {}).items():
            n = v.get("n_games") or 0
            if not n:
                continue
            wins = v.get("wins")
            draws = v.get("draws") or 0
            if wins is None:                       # старые файлы: только win_rate
                wins = (v.get("win_rate") or 0.0) * n
                draws = 0
            per_key.setdefault(key, []).append((lab, float(wins) + 0.5 * float(draws), float(n)))
    return per_key, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", nargs="+", required=True, help="метки прогонов кандидата")
    ap.add_argument("--base", nargs="+", required=True, help="метки прогонов базы")
    ap.add_argument("--cand-name", default="v074")
    ap.add_argument("--base-name", default="v073")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cand, miss_c = load_arm(args.cand)
    base, miss_b = load_arm(args.base)
    for lab, path in list(miss_c.items()) + list(miss_b.items()):
        print(f"[warn] нет файла прогона: {path}")

    keys = [k for k in cand if k in base]
    rows = []
    print(f"\n{'пара':26s} {args.cand_name+' прогоны':32s} {'пул':>7s}   "
          f"{args.base_name+' прогоны':32s} {'пул':>7s}   {'Δпп':>6s} {'p':>7s}")
    print("-" * 130)
    for k in keys:
        cw = sum(w for _, w, _ in cand[k]); cn = sum(n for _, _, n in cand[k])
        bw = sum(w for _, w, _ in base[k]); bn = sum(n for _, _, n in base[k])
        z, p = two_prop_ztest(cw, cn, bw, bn)
        c_runs = " ".join(f"{w/n:.3f}" for _, w, n in cand[k])
        b_runs = " ".join(f"{w/n:.3f}" for _, w, n in base[k])
        d = 100.0 * (cw / cn - bw / bn)
        flag = ""
        if p < 0.05:
            flag = "  ЗНАЧИМО " + ("вверх" if d > 0 else "ВНИЗ")
        print(f"{k:26s} {c_runs:32s} {cw/cn:7.4f}   {b_runs:32s} {bw/bn:7.4f}   "
              f"{d:+6.2f} {p:7.4f}{flag}")
        rows.append(dict(key=k, cand_pooled=cw / cn, cand_n=cn, base_pooled=bw / bn,
                         base_n=bn, delta_pp=d, z=z, p=p,
                         cand_runs=[dict(label=l, wr=w / n, n=n) for l, w, n in cand[k]],
                         base_runs=[dict(label=l, wr=w / n, n=n) for l, w, n in base[k]]))
    print("-" * 130)
    print(f"прогонов: {args.cand_name}={len(args.cand)} ({', '.join(args.cand)}), "
          f"{args.base_name}={len(args.base)} ({', '.join(args.base)})")

    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(dict(cand_labels=args.cand, base_labels=args.base, pairs=rows), f, indent=1)
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
