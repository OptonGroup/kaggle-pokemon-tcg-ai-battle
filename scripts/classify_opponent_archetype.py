#!/usr/bin/env python3
"""Classify opponent archetype for each episode in an index.csv from ground-truth
deck contents in the replay JSON (initial deck of the non-our player index) —
NOT from our own agent's in-game belief/detect_matchup (that can be wrong).

Usage:
    .venv/bin/python scripts/classify_opponent_archetype.py <out_dir>
    # <out_dir> must contain replays/{episode_id}.json and index.csv
    # (produced by scripts/download_episodes.py)

Writes <out_dir>/archetype_index.csv and prints a W/L-by-archetype summary,
plus a W/L-by-opponent-rating-band summary (100-wide bands 0-100..1200-1300,
bands with 0 games omitted). Rating bands need opp_team_id in index.csv
(added by download_episodes.py) joined against a public-leaderboard CSV
snapshot — auto-downloaded into <out_dir> if none is present there yet.
"""
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Key-card id sets for archetypes seen so far in this skill-band's meta.
# Extend this dict when a new archetype shows up as UNKNOWN with unfamiliar key cards
# (check opp_key_cards in the printed detail / archetype_index.csv).
ARCHETYPE_IDS = {
    "crustle": {344, 345, 756},          # Dwebble/Crustle + Mega Kangaskhan ex
    "archaludon": {169, 190},
    "lucario": {677, 678},
    "tr_mewtwo": {431, 401},
    "abomasnow": {722, 723},
    "alakazam": {741, 742, 743},          # NOTE: id 140 (Fezandipiti ex) deliberately
                                           # excluded — it's a generic tech card shared
                                           # across many decks (e.g. Dragapult), NOT
                                           # exclusive to Alakazam; caused false positives.
    "dragapult": {119, 120, 121},          # Dreepy/Drakloak/Dragapult ex
    "mirror_grimmsnarl": {646, 647, 648},
}

# Name-keyword fallback for archetypes not covered by id sets above.
NAME_KEYWORDS = {
    "starmie": ["Starmie"],
    "hop_trevenant": ["Trevenant", "Zacian"],
    "ogerpon": ["Ogerpon"],
    "slowking": ["Slowking"],
    "dipplin": ["Dipplin", "Hydrapple"],
    "bellibolt": ["Bellibolt", "Iono"],
    "zoroark": ["Zoroark"],
    "garchomp": ["Garchomp", "Cynthia"],
    "gholdengo": ["Gholdengo"],
    "charizard": ["Charizard"],
    "raging_bolt": ["Raging Bolt"],
    "miraidon": ["Miraidon"],
    "gardevoir": ["Gardevoir"],
    "pidgeot": ["Pidgeot"],
}


def find_initial_players(replay):
    steps = replay["steps"]
    for step in steps:
        for agent_state in step:
            viz = agent_state.get("visualize")
            if not viz:
                continue
            for entry in viz:
                cur = entry.get("current")
                if cur and cur.get("players"):
                    players = cur["players"]
                    if all(p.get("deck") for p in players):
                        return players
    return None


def classify(deck_ids, deck_names):
    for arche, ids in ARCHETYPE_IDS.items():
        if deck_ids & ids:
            return arche
    joined = " ".join(deck_names)
    for arche, kws in NAME_KEYWORDS.items():
        for kw in kws:
            if kw in joined:
                return arche
    return None


def load_leaderboard_scores(out_dir):
    """Return {team_id: score} from the freshest *leaderboard*.csv in out_dir,
    downloading one via the Kaggle CLI if none is present yet."""
    files = sorted(out_dir.glob("*leaderboard*.csv"))
    if not files:
        subprocess.run(
            ["kaggle", "competitions", "leaderboard", "pokemon-tcg-ai-battle",
             "-d", "-p", str(out_dir)],
            check=False, capture_output=True,
        )
        files = sorted(out_dir.glob("*leaderboard*.csv"))
    if not files:
        return {}
    path = max(files, key=lambda p: p.stat().st_mtime)
    scores = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                scores[int(row["TeamId"])] = float(row["Score"])
            except (KeyError, ValueError):
                continue
    return scores


def rating_band(score):
    lo = int(score // 100) * 100
    lo = max(0, min(lo, 1200))
    return f"{lo}-{lo + 100}"


def main():
    import json

    out_dir = Path(sys.argv[1])
    replay_dir = out_dir / "replays"
    index_csv = out_dir / "index.csv"
    rows = list(csv.DictReader(open(index_csv)))
    results = []
    for row in rows:
        ep_id = row["episode_id"]
        our_index = int(row["our_index"])
        opp_index = 1 - our_index
        reward = float(row["our_reward"])
        replay_path = replay_dir / f"{ep_id}.json"
        if not replay_path.exists():
            results.append({**row, "archetype": "NO_REPLAY", "opp_key_cards": ""})
            continue
        replay = json.loads(replay_path.read_text())
        players = find_initial_players(replay)
        if players is None:
            results.append({**row, "archetype": "NO_DECK_DATA", "opp_key_cards": ""})
            continue
        opp_deck = players[opp_index].get("deck", [])
        deck_ids = {c["id"] for c in opp_deck}
        deck_names = [c["name"] for c in opp_deck]
        arche = classify(deck_ids, deck_names)
        name_counts = Counter(deck_names)
        top_names = ", ".join(f"{n}x{c}" for n, c in name_counts.most_common(6))
        results.append({
            **row,
            "archetype": arche or "UNKNOWN",
            "opp_key_cards": top_names,
        })

    if not results:
        print("No episodes in index.csv")
        return

    out_csv = out_dir / "archetype_index.csv"
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {out_csv}\n")

    # self-validation episodes (opp_team_name empty/None) are excluded from WR
    ladder = [r for r in results if r.get("opp_team_name")]
    wins = [r for r in ladder if float(r["our_reward"]) > 0]
    losses = [r for r in ladder if float(r["our_reward"]) < 0]

    print(f"Ladder games: {len(ladder)}  W={len(wins)} L={len(losses)}"
          f"  WR={100*len(wins)/len(ladder):.1f}%" if ladder else "No ladder games yet")

    def print_arche_table(rows_w, rows_l):
        by_arche = Counter(r["archetype"] for r in rows_w + rows_l)
        for arche in sorted(by_arche, key=lambda a: -by_arche[a]):
            aw = sum(1 for r in rows_w if r["archetype"] == arche)
            al = sum(1 for r in rows_l if r["archetype"] == arche)
            total = aw + al
            print(f"  {arche}: {aw}W-{al}L  ({100*aw/total:.0f}%)" if total else f"  {arche}: 0")

    print("\n=== COMBINED WR by archetype (W-L, WR%) ===")
    print_arche_table(wins, losses)

    sub_ids = sorted({r["submission_id"] for r in ladder})
    if len(sub_ids) > 1:
        for sid in sub_ids:
            copy_labels = sorted({r["copy"] for r in ladder if r["submission_id"] == sid})
            sub_w = [r for r in wins if r["submission_id"] == sid]
            sub_l = [r for r in losses if r["submission_id"] == sid]
            print(f"\n=== submission_id={sid} (copy {','.join(copy_labels)}): "
                  f"{len(sub_w)}W-{len(sub_l)}L ===")
            print_arche_table(sub_w, sub_l)

    scores = load_leaderboard_scores(out_dir)
    if not scores:
        print("\n=== WR by opponent rating band: SKIPPED (no leaderboard CSV, could not download) ===")
    else:
        for r in ladder:
            tid = r.get("opp_team_id")
            r["opp_rating"] = scores.get(int(tid)) if tid else None
        rated_ids = {r["episode_id"] for r in ladder if r.get("opp_rating") is not None}
        missing = len(ladder) - len(rated_ids)

        def print_band_table(rows_w, rows_l):
            by_band = Counter(rating_band(r["opp_rating"]) for r in rows_w + rows_l)
            for band in sorted(by_band, key=lambda b: int(b.split("-")[0])):
                bw = sum(1 for r in rows_w if rating_band(r["opp_rating"]) == band)
                bl = sum(1 for r in rows_l if rating_band(r["opp_rating"]) == band)
                total = bw + bl
                if total:
                    print(f"  {band}: {bw}W-{bl}L  ({100*bw/total:.0f}%)")

        rated_w = [r for r in wins if r["episode_id"] in rated_ids]
        rated_l = [r for r in losses if r["episode_id"] in rated_ids]
        print(f"\n=== COMBINED WR by opponent rating band (100-wide, W-L, WR%) "
              f"[{missing}/{len(ladder)} opponents not on current leaderboard snapshot, excluded] ===")
        print_band_table(rated_w, rated_l)

        if len(sub_ids) > 1:
            for sid in sub_ids:
                copy_labels = sorted({r["copy"] for r in ladder if r["submission_id"] == sid})
                sub_w = [r for r in rated_w if r["submission_id"] == sid]
                sub_l = [r for r in rated_l if r["submission_id"] == sid]
                print(f"\n=== submission_id={sid} (copy {','.join(copy_labels)}) "
                      f"rating bands: {len(sub_w)}W-{len(sub_l)}L ===")
                print_band_table(sub_w, sub_l)

    print("\n=== Per-episode detail ===")
    for r in results:
        res = "W" if float(r["our_reward"]) > 0 else ("L" if float(r["our_reward"]) < 0 else "?")
        tag = "(self-val)" if not r.get("opp_team_name") else r["opp_team_name"]
        print(f"  ep{r['episode_id']} copy{r['copy']} {res} opp={tag} "
              f"arche={r['archetype']} cards=[{r['opp_key_cards']}]")


if __name__ == "__main__":
    main()
