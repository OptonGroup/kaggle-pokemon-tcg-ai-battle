#!/usr/bin/env python3
"""v076_gap_rank.py — Phase 1.4 of docs/gold/PLAN_v076.md: rank
(decision-category x opponent-archetype) cells by how much of the live WR gap they
can plausibly explain, so Phase 2 spends its budget on the few cells that matter.

Contribution score (the plan's formula):

    contrib = wr_gap(archetype) * decision_share(cell) * (1 - agreement(cell))

  * wr_gap        — expert WR against that archetype (measured on the reference
                    corpus by fidelity_expert_topk.py) minus OUR live WR against the
                    same archetype (out/cross_version_ledger.csv, 697 games
                    v063->v075). This is the "how much is actually lost here" term.
  * decision_share— the cell's share of all scored decisions: a category we get wrong
                    twice a game cannot explain a 8pp WR gap.
  * 1 - agreement — how differently we play that cell from the expert. Semantic
                    (index-free) agreement is used whenever at least half the cell's
                    decisions resolve to concrete cards; otherwise exact-index
                    agreement, and the choice is reported per row.

Nothing here is causal — contrib ranks WHERE TO LOOK, it does not prove that closing
a cell closes that share of the gap. Cells below --min-n are dropped, and the dropped
mass is printed so a silent truncation cannot masquerade as full coverage.

Usage:
    .venv/bin/python scripts/v076_gap_rank.py \
        --rows out/v076_fidelity/rows_top3.jsonl \
        --ledger out/cross_version_ledger.csv \
        --out out/v076_fidelity/gap_rank.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_our_live_wr(ledger: Path) -> dict:
    n, w = Counter(), Counter()
    with open(ledger, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a = r["arche"]
            n[a] += 1
            w[a] += int(r["win"])
    return {a: {"n": n[a], "wr": w[a] / n[a]} for a in n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="out/v076_fidelity/rows_top3.jsonl")
    ap.add_argument("--ledger", default="out/cross_version_ledger.csv")
    ap.add_argument("--out", default="out/v076_fidelity/gap_rank.csv")
    ap.add_argument("--min-n", type=int, default=100, help="min decisions per cell")
    ap.add_argument("--min-live-n", type=int, default=15,
                    help="min OUR live games vs an archetype for its wr_gap to be trusted")
    ap.add_argument("--key", choices=["category", "headline"], default="category")
    args = ap.parse_args()

    live = load_our_live_wr(Path(args.ledger))

    cells = defaultdict(lambda: {"n": 0, "match": 0, "res": 0, "sem": 0})
    arch_games = defaultdict(set)
    arch_wins = {}
    total = 0
    with open(args.rows, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            arch = r.get("opp_archetype", "UNKNOWN")
            key = (r[args.key], arch)
            c = cells[key]
            c["n"] += 1
            c["match"] += int(r["matched"])
            if r.get("resolvable"):
                c["res"] += 1
                c["sem"] += int(r["sem_matched"])
            gid = (r.get("episodeId"), r.get("name"))
            arch_games[arch].add(gid)
            arch_wins[(arch, gid)] = bool(r.get("won"))
            total += 1

    expert_wr = {a: sum(arch_wins[(a, g)] for g in gs) / len(gs)
                 for a, gs in arch_games.items() if gs}

    rows = []
    for (cat, arch), c in cells.items():
        n = c["n"]
        exact = c["match"] / n
        sem = (c["sem"] / c["res"]) if c["res"] else None
        use_sem = sem is not None and c["res"] >= 0.5 * n
        agr = sem if use_sem else exact
        ours = live.get(arch)
        our_wr = ours["wr"] if ours else None
        our_n = ours["n"] if ours else 0
        gap = (expert_wr[arch] - our_wr) if our_wr is not None else None
        share = n / total
        trusted = our_n >= args.min_live_n and gap is not None
        contrib = (gap * share * (1 - agr)) if trusted else None
        rows.append({
            "category": cat, "opp_archetype": arch, "n_decisions": n,
            "decision_share": round(share, 5),
            "agreement": round(agr, 4), "agreement_kind": "sem" if use_sem else "exact",
            "exact": round(exact, 4), "sem": (round(sem, 4) if sem is not None else None),
            "expert_wr": round(expert_wr[arch], 4),
            "our_live_wr": (round(our_wr, 4) if our_wr is not None else None),
            "our_live_n": our_n,
            "wr_gap": (round(gap, 4) if gap is not None else None),
            "contrib": (round(contrib, 6) if contrib is not None else None),
            "expert_games": len(arch_games[arch]),
        })

    kept = [r for r in rows if r["n_decisions"] >= args.min_n]
    dropped = [r for r in rows if r["n_decisions"] < args.min_n]
    drop_mass = sum(r["n_decisions"] for r in dropped) / total if total else 0.0
    untrusted = [r for r in kept if r["contrib"] is None]

    kept.sort(key=lambda r: (r["contrib"] is None, -(r["contrib"] or 0)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(kept[0].keys()) if kept else [])
        wcsv.writeheader()
        wcsv.writerows(kept)

    scored = [r for r in kept if r["contrib"] is not None]
    tot_contrib = sum(r["contrib"] for r in scored) or 1e-12
    print(f"\n=== {len(kept)} ячеек >= {args.min_n} решений "
          f"({len(dropped)} отброшено = {drop_mass*100:.1f}% решений; "
          f"{len(untrusted)} без надёжного live-WR) ===\n")
    print(f"{'#':>3} {'category':<26} {'opp':<18} {'n':>6} {'share':>6} "
          f"{'agr':>6} {'k':>5} {'expWR':>6} {'ourWR':>6} {'gap':>6} {'contrib':>8} {'cum%':>6}")
    cum = 0.0
    for i, r in enumerate(scored[:40], 1):
        cum += r["contrib"]
        print(f"{i:>3} {r['category']:<26} {r['opp_archetype']:<18} {r['n_decisions']:>6} "
              f"{r['decision_share']*100:>5.1f}% {r['agreement']:>6.3f} "
              f"{r['agreement_kind']:>5} {r['expert_wr']:>6.3f} {r['our_live_wr']:>6.3f} "
              f"{r['wr_gap']:>+6.3f} {r['contrib']:>8.5f} {cum/tot_contrib*100:>5.1f}%")
    if untrusted:
        print(f"\nбез live-WR (our_live_n < {args.min_live_n}) — не ранжированы:")
        for r in sorted(untrusted, key=lambda r: -r["n_decisions"])[:10]:
            print(f"    {r['category']:<26} {r['opp_archetype']:<18} n={r['n_decisions']:>5} "
                  f"agr={r['agreement']:.3f} expWR={r['expert_wr']:.3f} our_n={r['our_live_n']}")
    print(f"\n[gap_rank] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
