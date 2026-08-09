#!/usr/bin/env python3
"""v076_gate_paired.py — paired v076-vs-v075 read of the Phase-3 gate.

The gate's built-in comparison is against a HISTORICAL baseline file (v073's numbers), so
part of any delta it reports belongs to v075, which already shipped. This script instead
pairs the two candidates run on the SAME seed and the SAME opponent set, then combines the
two seeds with inverse-variance weights -- the protocol from
`project_gate_stratified_measurement` (pooling raw counters across pairs invites Simpson's
paradox, so every pair is differenced on its own and only the differences are pooled).

Reports per pair: WR of each side per seed, the per-seed delta, the pooled delta with its
z and p, and whether the two seeds agree in sign (replication, not just significance).

Usage:
    .venv/bin/python scripts/v076_gate_paired.py \
        --cand out/v076_gate/v076_seedA_core.json out/v076_gate/v076_seedB_core.json \
        --base out/v076_gate/v075_seedA_core.json out/v076_gate/v075_seedB_core.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Opponent archetype of each gate pair, for the stratified roll-up. The gate plays our
# agent against these submissions; the label is the deck the OPPONENT brings.
STRATA = {
    "v009a": "crustle", "ext_tusk_crustle": "crustle",
    "v044": "archaludon", "v036_h2h": "archaludon",
    "ext_masamikobayashi_archaludon": "archaludon",
    "ext_maulikgajera_lucario": "lucario", "ext_nursrijan_lucario": "lucario",
    "v017": "alakazam", "v016": "alakazam",
    "ext_prvsiyan_alak_v8": "alakazam", "ext_prvsiyan_alak_v9": "alakazam",
    "v063_h2h": "mirror", "v065_h2h": "mirror", "v072_h2h": "mirror",
    "v073_h2h": "mirror", "v049": "other",
}
# Pairs the edit has no mechanism to touch -- used as the control stratum: a real effect
# shows up in the targeted strata and NOT here. (The v076 edits are resource-card
# priorities, which apply in every matchup, so no pair is a true control; the mirror
# pairs are the closest thing -- both sides get the same change -- and are reported
# separately rather than claimed as a clean control.)
CONTROL = {"v063_h2h", "v065_h2h", "v072_h2h", "v073_h2h"}


def load(paths) -> dict:
    out = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        for k, v in (d.get("opponents") or {}).items():
            out.setdefault(k, []).append(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--out", default="out/v076_gate/paired_summary.json")
    args = ap.parse_args()

    c, b = load(args.cand), load(args.base)
    keys = [k for k in c if k in b and len(c[k]) == len(b[k])]
    rows = []
    for k in sorted(keys):
        seeds = []
        for cs, bs in zip(c[k], b[k]):
            nc, nb = cs["n_games"], bs["n_games"]
            pc, pb = cs["win_rate"], bs["win_rate"]
            var = pc * (1 - pc) / nc + pb * (1 - pb) / nb
            seeds.append({"cand": pc, "base": pb, "delta": pc - pb, "var": var,
                          "n_cand": nc, "n_base": nb})
        wsum = sum(1 / s["var"] for s in seeds if s["var"] > 0)
        if wsum <= 0:
            continue
        pooled = sum(s["delta"] / s["var"] for s in seeds if s["var"] > 0) / wsum
        se = math.sqrt(1 / wsum)
        z = pooled / se if se > 0 else 0.0
        p = math.erfc(abs(z) / math.sqrt(2))
        signs = {(1 if s["delta"] > 0 else (-1 if s["delta"] < 0 else 0)) for s in seeds}
        rows.append({"pair": k, "stratum": STRATA.get(k, "other"), "seeds": seeds,
                     "pooled_delta": pooled, "se": se, "z": z, "p": p,
                     "replicated": len(signs) == 1 and 0 not in signs,
                     "control": k in CONTROL})

    rows.sort(key=lambda r: -abs(r["pooled_delta"]))
    print(f"\n{'пара':<32} {'страта':<11} {'A: v076/v075':>16} {'B: v076/v075':>16} "
          f"{'Δ пул':>8} {'z':>6} {'p':>8} {'реплик':>7}")
    for r in rows:
        s = r["seeds"]
        a = f"{s[0]['cand']:.3f}/{s[0]['base']:.3f}"
        bb = f"{s[1]['cand']:.3f}/{s[1]['base']:.3f}" if len(s) > 1 else "—"
        mark = " [контроль]" if r["control"] else ""
        print(f"{r['pair']:<32} {r['stratum']:<11} {a:>16} {bb:>16} "
              f"{r['pooled_delta']*100:>+7.2f} {r['z']:>6.2f} {r['p']:>8.4f} "
              f"{'да' if r['replicated'] else 'нет':>7}{mark}")

    print(f"\n{'страта':<14} {'пар':>4} {'Δ взвеш.':>9} {'z':>6} {'p':>8}")
    strata = {}
    for r in rows:
        strata.setdefault(r["stratum"], []).append(r)
    for st, rs in sorted(strata.items()):
        w = sum(1 / (r["se"] ** 2) for r in rs if r["se"] > 0)
        d = sum(r["pooled_delta"] / (r["se"] ** 2) for r in rs if r["se"] > 0) / w
        se = math.sqrt(1 / w)
        z = d / se
        print(f"{st:<14} {len(rs):>4} {d*100:>+8.2f} {z:>6.2f} "
              f"{math.erfc(abs(z)/math.sqrt(2)):>8.4f}")

    noncontrol = [r for r in rows if not r["control"]]
    w = sum(1 / (r["se"] ** 2) for r in noncontrol)
    d = sum(r["pooled_delta"] / (r["se"] ** 2) for r in noncontrol) / w
    se = math.sqrt(1 / w)
    z = d / se
    print(f"\nПО ПОЛЮ (без зеркал): Δ = {d*100:+.2f} пп, z={z:.2f}, "
          f"p={math.erfc(abs(z)/math.sqrt(2)):.5f}")
    regr = [r for r in rows if r["pooled_delta"] < 0 and r["p"] < 0.05]
    print(f"значимых регрессий (p<0.05, Δ<0): {len(regr)}"
          + (": " + ", ".join(r["pair"] for r in regr) if regr else ""))
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\n[paired] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
