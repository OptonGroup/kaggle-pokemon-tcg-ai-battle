#!/usr/bin/env python3
"""v076_offline_verify.py — the v075 surgical-isolation protocol for one v076 edit.

Two --rows-out files produced by fidelity_expert_topk.py on the SAME corpus, one from the
incumbent pilot and one from the candidate. The two runs are deterministic and share the
decision order, so rows pair up positionally; the pairing is asserted on
(episodeId, turn, ctx, expert) before anything is compared.

Reports, per decision category:
  changed      — how many decisions the edit moved at all
  toward       — moved from disagreeing with the expert to agreeing (semantic)
  away         — moved from agreeing to disagreeing  <- must stay ~0 outside the target
  net          — toward - away
  sem_before/after — the category's agreement, so a "win" that is really noise is visible

An edit passes isolation when every NON-target category shows changed == 0. That is
stricter than "net >= 0": a category that churns without a net change is still an
unintended behaviour change, and v075 shipped only because it moved nothing else.

Usage:
    .venv/bin/python scripts/v076_offline_verify.py \
        --before out/v076_fidelity/iter_v075.jsonl \
        --after  out/v076_fidelity/iter_v076a.jsonl \
        --target play-support
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(p: Path) -> list:
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--target", nargs="*", default=[],
                    help="categories the edit is ALLOWED to move")
    ap.add_argument("--key", default="category", choices=["category", "headline"])
    ap.add_argument("--target-ref", nargs="*", default=[],
                    help="action signatures the edit is allowed to add/remove, e.g. "
                         "\"('play', 1086)\". Categories are labelled by the EXPERT's "
                         "choice, so an edit to OUR ranking of one card necessarily "
                         "shows up under several of them; this is the surgical check.")
    args = ap.parse_args()

    b, a = load(Path(args.before)), load(Path(args.after))
    if len(b) != len(a):
        print(f"ОШИБКА: разное число решений {len(b)} vs {len(a)} — корпуса не совпадают")
        return 1

    stats = defaultdict(lambda: {"n": 0, "changed": 0, "toward": 0, "away": 0,
                                 "sb": 0, "sa": 0})
    refs = set(args.target_ref)
    off_ref = []
    mismatched = 0
    for rb, ra in zip(b, a):
        if (rb.get("episodeId"), rb.get("turn"), rb.get("ctx"), rb.get("expert")) != \
           (ra.get("episodeId"), ra.get("turn"), ra.get("ctx"), ra.get("expert")):
            mismatched += 1
            continue
        s = stats[rb[args.key]]
        s["n"] += 1
        s["sb"] += int(rb["sem_matched"])
        s["sa"] += int(ra["sem_matched"])
        if rb["our_ref"] != ra["our_ref"]:
            s["changed"] += 1
            if ra["sem_matched"] and not rb["sem_matched"]:
                s["toward"] += 1
            elif rb["sem_matched"] and not ra["sem_matched"]:
                s["away"] += 1
            if refs:
                moved = set(rb["our_ref"]) ^ set(ra["our_ref"])
                if not (moved & refs):
                    off_ref.append((rb[args.key], rb["our_ref"], ra["our_ref"]))
    if mismatched:
        print(f"ОШИБКА: {mismatched} строк не спарились по (episodeId, turn, ctx, expert)")
        return 1

    targets = set(args.target)
    print(f"\n{'категория':<30} {'n':>7} {'изменено':>9} {'→эксперт':>9} {'←от него':>9} "
          f"{'нетто':>7} {'sem до':>7} {'sem после':>10}")
    tot = {"n": 0, "changed": 0, "toward": 0, "away": 0}
    leaks = []
    for c, s in sorted(stats.items(), key=lambda kv: -kv[1]["changed"]):
        if s["changed"] == 0 and c not in targets:
            continue
        mark = " ←цель" if c in targets else (" ←УТЕЧКА" if s["changed"] else "")
        print(f"{c:<30} {s['n']:>7} {s['changed']:>9} {s['toward']:>9} {s['away']:>9} "
              f"{s['toward']-s['away']:>+7} {s['sb']/s['n']:>7.3f} {s['sa']/s['n']:>10.3f}{mark}")
        if c not in targets and s["changed"]:
            leaks.append((c, s["changed"]))
    for c, s in stats.items():
        for k in tot:
            tot[k] += s[k]
    sb = sum(s["sb"] for s in stats.values())
    sa = sum(s["sa"] for s in stats.values())
    print(f"\nВСЕГО: решений {tot['n']}, изменено {tot['changed']} "
          f"({tot['changed']/tot['n']*100:.2f}%), к эксперту {tot['toward']}, "
          f"от эксперта {tot['away']}, нетто {tot['toward']-tot['away']:+d}")
    print(f"sem-согласие в целом: {sb/tot['n']:.4f} -> {sa/tot['n']:.4f} "
          f"({(sa-sb)/tot['n']*100:+.2f} пп)")
    if refs:
        print(f"\nХИРУРГИЧЕСКАЯ ИЗОЛЯЦИЯ по действию {sorted(refs)}:")
        if off_ref:
            print(f"  НАРУШЕНА — {len(off_ref)} изменений не затрагивают целевое действие:")
            for c, before, after in off_ref[:10]:
                print(f"    [{c}] {before} -> {after}")
            return 2
        print(f"  ЧИСТАЯ — все {tot['changed']} изменённых решений добавили или убрали "
              f"именно это действие, больше ничего не сдвинулось")
        return 0
    if leaks:
        print("\nИЗОЛЯЦИЯ НАРУШЕНА — изменения вне целевых категорий:")
        for c, n in leaks:
            print(f"    {c}: {n}")
        return 2
    print("\nИЗОЛЯЦИЯ ЧИСТАЯ: вне целевых категорий не изменилось ни одного решения")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
