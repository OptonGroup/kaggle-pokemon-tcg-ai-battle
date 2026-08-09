#!/usr/bin/env python3
"""pick_candidate_decklist.py — Phase 1 step 2: pick the single best decklist
per candidate archetype from the winning-deck signatures gathered by
scripts/build_candidate_decks.py, validate it with
deckbelief_common.build_card_caps_and_ace_mask, and write
decks/candidates/{Lopunny,OgerponTeal}.csv (single-column card_id, one row
per copy, 60 rows — same format as decks/meta/*.csv).

Selection rule (per user spec):
  1. Group saved WINNING decks by exact 60-card multiset signature.
  2. N >= 15 wins required to even be considered.
  3. Among N>=15 signatures, pick the one with the highest "best pilot"
     rating (max team rating among winning teams that played it).
  4. If the top two candidates' max-rating differ by < 30 mu, break the tie
     by convergence (# distinct winning team_names on that signature).
  5. Raw in-archetype WR (wins vs losses, ALL games not just wins) is a
     sanity check only (expect >= ~45%), not a selection criterion.

Usage:
    .venv/bin/python scripts/pick_candidate_decklist.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from scan_daily_zip_field import load_leaderboard_by_name  # noqa: E402
from deckbelief_common import build_card_caps_and_ace_mask  # noqa: E402

CAND_DIR = REPO / "out" / "field_recon_20260729_0801" / "candidate_decks"
OUT_DIR = REPO / "decks" / "candidates"
MIN_N = 15
TIE_BREAK_DELTA = 30.0

ARCHETYPE_OUT_NAME = {
    "lopunny": "lopunny.csv",
    "ogerpon_teal": "ogerpon_teal.csv",
}


def sig_key(deck_ids: list[int]) -> str:
    return ",".join(f"{cid}:{cnt}" for cid, cnt in sorted(Counter(deck_ids).items()))


def sig_to_deck_ids(key: str) -> list[int]:
    out = []
    for pair in key.split(","):
        cid_s, cnt_s = pair.split(":")
        out.extend([int(cid_s)] * int(cnt_s))
    return out


def validate_deck(deck_ids: list[int]) -> list[str]:
    """Returns a list of validation error strings (empty == passes)."""
    errors = []
    if len(deck_ids) != 60:
        errors.append(f"deck has {len(deck_ids)} cards, expected 60")

    card_caps, ace_mask = build_card_caps_and_ace_mask(
        csv_path=str(REPO / "data" / "EN_Card_Data.csv")
    )
    counts = Counter(deck_ids)
    ace_total = 0
    for cid, cnt in counts.items():
        if cid < 0 or cid >= card_caps.shape[0]:
            errors.append(f"card id {cid} out of range")
            continue
        if bool(ace_mask[cid]):
            ace_total += cnt
            continue
        cap = float(card_caps[cid])
        if cnt > cap:
            errors.append(f"card id {cid}: {cnt} copies > cap {cap}")
    if ace_total > 1:
        errors.append(f"ACE SPEC total copies = {ace_total} > 1")
    return errors


def main():
    name_to_rating, leaderboard_path = load_leaderboard_by_name()
    print(f"[pick] rating source: {leaderboard_path} ({len(name_to_rating)} teams)\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {}

    for arche, out_name in ARCHETYPE_OUT_NAME.items():
        print(f"{'='*24} {arche} {'='*24}")
        wins = [json.loads(l) for l in open(CAND_DIR / f"{arche}.jsonl", encoding="utf-8")]
        wl = json.load(open(CAND_DIR / f"{arche}_sig_wl.json", encoding="utf-8"))

        by_sig = defaultdict(list)
        for w in wins:
            by_sig[sig_key(w["deck_ids"])].append(w)

        rows = []
        for key, recs in by_sig.items():
            teams = sorted({r["team_name"] for r in recs})
            ratings = [name_to_rating[t] for t in teams if t in name_to_rating]
            max_rating = max(ratings) if ratings else None
            n = len(recs)
            w_l = wl.get(key, {})
            w_, l_ = w_l.get("wins", 0), w_l.get("losses", 0)
            wr = w_ / (w_ + l_) if (w_ + l_) else None
            rows.append({
                "key": key, "n": n, "max_rating": max_rating,
                "conv": len(teams), "teams": teams, "wr": wr, "w": w_, "l": l_,
            })

        rows.sort(key=lambda r: -r["n"])
        print(f"Total winning games: {len(wins)}  Unique signatures: {len(rows)}")
        print("\nAll signatures (sorted by N wins):")
        for r in rows:
            passed = "PASS" if r["n"] >= MIN_N else "fail(N<15)"
            mr = f"{r['max_rating']:.1f}" if r["max_rating"] is not None else "None"
            wr_s = f"{r['wr']:.1%}" if r["wr"] is not None else "None"
            print(f"  [{passed:10s}] N={r['n']:4d}  max_rating={mr:>8s}  conv={r['conv']:3d}  "
                  f"WR={wr_s} ({r['w']}W-{r['l']}L)  teams_sample={r['teams'][:4]}")

        eligible = [r for r in rows if r["n"] >= MIN_N]
        if not eligible:
            print(f"\n*** NO signature reached N>=15 wins for {arche}. "
                  f"NOT producing a final decklist for this archetype. ***\n")
            report[arche] = {"status": "NO_ELIGIBLE_SIGNATURE", "rows": rows}
            continue

        # rank by max_rating (None treated as -inf), then apply the <30mu
        # convergence tie-break against the runner-up only when they're close.
        def rating_or_neg_inf(r):
            return r["max_rating"] if r["max_rating"] is not None else float("-inf")

        eligible_sorted = sorted(eligible, key=lambda r: -rating_or_neg_inf(r))
        best = eligible_sorted[0]
        if len(eligible_sorted) > 1:
            runner_up = eligible_sorted[1]
            if rating_or_neg_inf(best) - rating_or_neg_inf(runner_up) < TIE_BREAK_DELTA:
                # tie-break by convergence
                tied = [r for r in eligible_sorted
                        if rating_or_neg_inf(best) - rating_or_neg_inf(r) < TIE_BREAK_DELTA]
                best = max(tied, key=lambda r: r["conv"])
                print(f"\n[pick] tie-break triggered (max_rating within {TIE_BREAK_DELTA}mu) "
                      f"-> chose by convergence")

        print(f"\n[pick] SELECTED signature for {arche}: N={best['n']} wins, "
              f"max_rating={best['max_rating']}, conv={best['conv']}, "
              f"WR={best['wr']:.1%} ({best['w']}W-{best['l']}L)")
        if best["wr"] is not None and best["wr"] < 0.45:
            print(f"[pick] WARNING: WR {best['wr']:.1%} is below the ~45% sanity floor")

        deck_ids = sig_to_deck_ids(best["key"])
        errors = validate_deck(deck_ids)
        if errors:
            print(f"[pick] VALIDATION FAILED for {arche}:")
            for e in errors:
                print(f"    - {e}")
            report[arche] = {"status": "VALIDATION_FAILED", "errors": errors, "rows": rows, "best": best}
            continue
        print(f"[pick] validation PASSED (60 cards, per-card cap, ACE SPEC <=1)")

        out_csv = OUT_DIR / out_name
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            for cid in deck_ids:
                w.writerow([cid])
        print(f"[pick] wrote {out_csv} (60 rows)\n")

        report[arche] = {
            "status": "OK", "rows": rows, "best": best, "out_csv": str(out_csv),
        }

    print("\n=== SUMMARY ===")
    for arche, r in report.items():
        print(f"{arche}: {r['status']}")


if __name__ == "__main__":
    main()
