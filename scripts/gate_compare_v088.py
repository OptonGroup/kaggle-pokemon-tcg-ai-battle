#!/usr/bin/env python3
"""gate_compare_v088.py — стратифицированное сравнение кандидата с КЭШЕМ базы на одних и тех
же сидах гейт-пула (MEASUREMENT_POLICY: стратификация + контрольная страта + >=2 рерана +
обратно-дисперсионные веса + Stouffer, БЕЗ пулинга разнородных страт).

Читает band_summary.json из пар каталогов out/gate_pool2/{cand,base}_{seed} и
{cand,base}_cw_{seed} (crustle_wall — контрольная страта: механизм K1/K2 её не касается,
значимый минус там = сигнал общей регрессии, а не целевого эффекта).

По каждой страте (архетипу) на каждом сиде: дельта WR и её дисперсия (сумма дисперсий
двух независимых прогонов, игры разные). Между сидами — обратно-дисперсионное среднее;
z = delta / se. Отдельно то же для бэнд-взвешенных WR (band_700_799 / band_800_plus),
это главные числа приёмки, сопоставимые с «v087 vs v085: band +3.67пп z=2.82».

Пример:
  .venv/bin/python scripts/gate_compare_v088.py --cand v088 --base v087 \
    --seeds 20260809 20260810 20260811 --cw-seeds 20260809 20260810
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "out" / "gate_pool2"


def load(tag: str, seed: str, cw: bool) -> dict | None:
    p = GATE / (f"{tag}_cw_{seed}" if cw else f"{tag}_{seed}") / "band_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def iv_combine(deltas: list[float], variances: list[float]) -> tuple[float, float, float]:
    """Обратно-дисперсионное среднее: w_i = 1/var_i; se = sqrt(1/Σw_i)."""
    pairs = [(d, v) for d, v in zip(deltas, variances) if v and v > 0]
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    w = [1.0 / v for _, v in pairs]
    delta = sum(wi * d for wi, (d, _) in zip(w, pairs)) / sum(w)
    se = math.sqrt(1.0 / sum(w))
    return delta, se, (delta / se if se else float("nan"))


def stouffer(zs: list[float]) -> float:
    zs = [z for z in zs if z == z]
    return sum(zs) / math.sqrt(len(zs)) if zs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--cw-seeds", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report: dict = {"cand": args.cand, "base": args.base, "per_archetype": {},
                    "per_band": {}, "control_crustle_wall": {}, "safety": {}}

    # ── страты-архетипы + бэнды (основной пул) ───────────────────────────────
    arch_d: dict[str, list] = {}
    arch_v: dict[str, list] = {}
    arch_z: dict[str, list] = {}
    band_d: dict[str, list] = {}
    band_v: dict[str, list] = {}
    per_seed_rows = []
    for seed in args.seeds:
        c, b = load(args.cand, seed, False), load(args.base, seed, False)
        if c is None or b is None:
            print(f"[warn] пропуск сида {seed}: cand={c is not None} base={b is not None}")
            continue
        for a, ca in c["archetype_table"].items():
            ba = b["archetype_table"].get(a)
            if not ca or not ba:
                continue
            d = ca["win_rate"] - ba["win_rate"]
            v = ca["variance"] + ba["variance"]
            arch_d.setdefault(a, []).append(d)
            arch_v.setdefault(a, []).append(v)
            arch_z.setdefault(a, []).append(d / math.sqrt(v) if v > 0 else float("nan"))
            per_seed_rows.append({"seed": seed, "archetype": a, "cand_wr": ca["win_rate"],
                                  "base_wr": ba["win_rate"], "n": ca["n_games"],
                                  "delta_pp": round(100 * d, 2)})
        for bk, cb in c["bands"].items():
            bb = b["bands"].get(bk)
            if not bb:
                continue
            d = cb["weighted_wr_renormalized"] - bb["weighted_wr_renormalized"]
            v = cb["weighted_var_renormalized"] + bb["weighted_var_renormalized"]
            band_d.setdefault(bk, []).append(d)
            band_v.setdefault(bk, []).append(v)

    for a in sorted(arch_d):
        d, se, z = iv_combine(arch_d[a], arch_v[a])
        report["per_archetype"][a] = {
            "delta_pp": round(100 * d, 3), "se_pp": round(100 * se, 3), "z": round(z, 3),
            "stouffer_z": round(stouffer(arch_z[a]), 3), "n_seeds": len(arch_d[a]),
            "per_seed_delta_pp": [round(100 * x, 2) for x in arch_d[a]]}
    for bk in sorted(band_d):
        d, se, z = iv_combine(band_d[bk], band_v[bk])
        report["per_band"][bk] = {
            "delta_pp": round(100 * d, 3), "se_pp": round(100 * se, 3), "z": round(z, 3),
            "n_seeds": len(band_d[bk]),
            "per_seed_delta_pp": [round(100 * x, 2) for x in band_d[bk]]}

    # ── контрольная страта crustle_wall ──────────────────────────────────────
    cw_d, cw_v = [], []
    for seed in args.cw_seeds:
        c, b = load(args.cand, seed, True), load(args.base, seed, True)
        if c is None or b is None:
            print(f"[warn] пропуск cw-сида {seed}")
            continue
        for a, ca in c["archetype_table"].items():
            ba = b["archetype_table"].get(a)
            if not ca or not ba:
                continue
            cw_d.append(ca["win_rate"] - ba["win_rate"])
            cw_v.append(ca["variance"] + ba["variance"])
    if cw_d:
        d, se, z = iv_combine(cw_d, cw_v)
        report["control_crustle_wall"] = {"delta_pp": round(100 * d, 3),
                                          "se_pp": round(100 * se, 3), "z": round(z, 3),
                                          "n_seeds": len(cw_d),
                                          "per_seed_delta_pp": [round(100 * x, 2) for x in cw_d]}

    # ── безопасность: исключения/нелегальные ходы/таймауты кандидата ─────────
    exc = ill = to = cr = ngames = 0
    for seed in args.seeds:
        p = GATE / f"{args.cand}_{seed}" / "candidate_core.json"
        if not p.exists():
            continue
        core = json.loads(p.read_text())
        for o in core["opponents"].values():
            exc += o.get("exc_agent", 0); ill += o.get("illegal_agent", 0)
            to += o.get("timeouts", 0); cr += o.get("crashes", 0)
            ngames += o.get("n_games", 0)
    report["safety"] = {"n_games": ngames, "exc_agent": exc, "illegal_agent": ill,
                        "timeouts": to, "crashes": cr,
                        "clean": (exc == 0 and ill == 0 and to == 0 and cr == 0)}

    print(f"=== {args.cand} vs {args.base} (сиды: {', '.join(args.seeds)}) ===\n")
    print(f"{'страта':16s} {'Δпп':>8s} {'se':>7s} {'z':>7s} {'Stouffer':>9s}  по сидам")
    for a, r in report["per_archetype"].items():
        print(f"{a:16s} {r['delta_pp']:+8.2f} {r['se_pp']:>7.2f} {r['z']:+7.2f} "
              f"{r['stouffer_z']:+9.2f}  {r['per_seed_delta_pp']}")
    print()
    for bk, r in report["per_band"].items():
        print(f"{bk:16s} {r['delta_pp']:+8.2f} {r['se_pp']:>7.2f} {r['z']:+7.2f} "
              f"{'':>9s}  {r['per_seed_delta_pp']}")
    if report["control_crustle_wall"]:
        r = report["control_crustle_wall"]
        print(f"\n{'КОНТРОЛЬ cw':16s} {r['delta_pp']:+8.2f} {r['se_pp']:>7.2f} {r['z']:+7.2f}"
              f"  {r['per_seed_delta_pp']}")
    s = report["safety"]
    print(f"\nбезопасность: {s['n_games']} игр, exc={s['exc_agent']} illegal={s['illegal_agent']} "
          f"timeouts={s['timeouts']} crashes={s['crashes']} -> {'ЧИСТО' if s['clean'] else 'ГРЯЗНО'}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
