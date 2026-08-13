#!/usr/bin/env python3
"""Стратифицированное сравнение v084s против кэша v083 (пул 2.0 + cw), бар §6.2.

Пулит wins/n по сидам ВНУТРИ каждого оппонента (никакого пулинга между стратами),
считает двухпропорционный z и дельты бэндов по BAND_WEIGHTS из scout_gate_pool2.
Использование: .venv/bin/python scripts/compare_v084_gate.py [префикс_кандидата]
(дефолт v084s_; для v2: v084v2_)
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GP = os.path.join(ROOT, "out", "gate_pool2")

MAIN_SEEDS = ["20260809", "20260810", "20260811"]
CW_SEEDS = ["20260809", "20260810"]

BAND_WEIGHTS = {
    "band_700_799": {"lucario": 0.26, "alakazam": 0.19, "archaludon": 0.17, "mirror": 0.09},
    "band_800_plus": {"mirror": 0.35, "alakazam": 0.21, "archaludon": 0.11},
}
ARCH_OF_OPP = {
    "ext2_tetsutani_grimmsnarl": "mirror",
    "ext2_jazivxt_codex": "alakazam",
    "ext2_prvsiyan_router": "crustle_mill",
    "ext_maulikgajera_lucario": "lucario",
    "v044_archaludon": "archaludon",
    "ext_masamikobayashi_archaludon": "archaludon",
    "v009a_crustle": "crustle_wall",
    "ext_tusk_crustle": "crustle_wall",
}


def load_core(dirpath):
    for fn in os.listdir(dirpath):
        if fn.endswith("_core.json"):
            return json.load(open(os.path.join(dirpath, fn)))
    raise FileNotFoundError(dirpath)


def collect(prefix, seeds, cw_prefix, cw_seeds):
    """-> {opp: (wins, n)} пулинг по сидам внутри оппонента."""
    acc = {}
    health = {"exc": 0, "illegal": 0, "timeouts": 0, "crashes": 0}
    for s in seeds:
        core = load_core(os.path.join(GP, f"{prefix}{s}"))
        for opp, r in core["opponents"].items():
            w, n = acc.get(opp, (0, 0))
            acc[opp] = (w + r["wins"], n + r["n_games"])
            health["exc"] += r["exc_agent"]
            health["illegal"] += r["illegal_agent"]
            health["timeouts"] += r["timeouts"]
            health["crashes"] += r["crashes"]
    for s in cw_seeds:
        core = load_core(os.path.join(GP, f"{cw_prefix}{s}"))
        for opp, r in core["opponents"].items():
            w, n = acc.get(opp, (0, 0))
            acc[opp] = (w + r["wins"], n + r["n_games"])
            health["exc"] += r["exc_agent"]
            health["illegal"] += r["illegal_agent"]
            health["timeouts"] += r["timeouts"]
            health["crashes"] += r["crashes"]
    return acc, health


def two_prop_z(w1, n1, w0, n0):
    p1, p0 = w1 / n1, w0 / n0
    p = (w1 + w0) / (n1 + n0)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))
    return (p1 - p0) / se if se > 0 else 0.0


def arch_wr(acc, arch):
    """WR архетипа: равновзвешенное среднее per-opponent WR (как в band_summary)."""
    rates, n_tot = [], 0
    for opp, (w, n) in acc.items():
        if ARCH_OF_OPP.get(opp) == arch and n > 0:
            rates.append(w / n)
            n_tot += n
    return (sum(rates) / len(rates), n_tot) if rates else (None, 0)


def main():
    pfx = sys.argv[1] if len(sys.argv) > 1 else "v084s_"
    cand, ch = collect(pfx, MAIN_SEEDS, pfx + "cw_", CW_SEEDS)
    base, bh = collect("v083_", MAIN_SEEDS, "v083_cw_", CW_SEEDS)

    print(f"health cand: {ch}")
    print(f"health base: {bh}")
    print()
    print(f"{'оппонент':38s} {'v084s':>8s} {'v083':>8s} {'Δпп':>7s} {'z':>6s}  n")
    for opp in sorted(cand, key=lambda o: ARCH_OF_OPP.get(o, "zz")):
        w1, n1 = cand[opp]
        if opp not in base:
            print(f"{opp:38s} {w1/n1:8.3f} {'—':>8s}")
            continue
        w0, n0 = base[opp]
        z = two_prop_z(w1, n1, w0, n0)
        print(
            f"{opp:38s} {w1/n1:8.3f} {w0/n0:8.3f} {100*(w1/n1-w0/n0):+7.1f} {z:+6.2f}  {n1}/{n0}"
            f"  [{ARCH_OF_OPP.get(opp,'?')}]"
        )

    print()
    for band, weights in BAND_WEIGHTS.items():
        d_band, wsum = 0.0, 0.0
        var_band = 0.0
        for arch, share in weights.items():
            r1, n1 = arch_wr(cand, arch)
            r0, n0 = arch_wr(base, arch)
            if r1 is None or r0 is None:
                continue
            d_band += share * (r1 - r0)
            wsum += share
            var_band += (share ** 2) * (
                r1 * (1 - r1) / max(n1, 1) + r0 * (1 - r0) / max(n0, 1)
            )
        d = d_band / wsum
        se = math.sqrt(var_band) / wsum
        print(
            f"{band}: Δ={100*d:+.2f}пп (renorm по покрытой массе {wsum:.2f}), "
            f"SE={100*se:.2f}пп, z={d/se:+.2f}"
        )


if __name__ == "__main__":
    main()
