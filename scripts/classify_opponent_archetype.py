#!/usr/bin/env python3
"""Classify opponent archetype for each episode in an index.csv from ground-truth
deck contents in the replay JSON (initial deck of the non-our player index) —
NOT from our own agent's in-game belief/detect_matchup (that can be wrong).

Usage:
    .venv/bin/python scripts/classify_opponent_archetype.py <out_dir> \
        [--cluster-unknown] [--audit] [--by-cluster]
    # <out_dir> must contain replays/{episode_id}.json and index.csv
    # (produced by scripts/download_episodes.py)
    #
    # --cluster-unknown: decks the key-card table doesn't recognise are grouped by
    #   cosine similarity and named from the card data (auto:<Pokemon>) instead of
    #   all collapsing into a single opaque "UNKNOWN" bucket.
    # --audit: also cluster EVERY opponent deck and print a cluster-vs-label
    #   cross-tab — this is how you check whether the hand-written table below
    #   still matches what the field actually plays (a cluster split across two
    #   labels, or two clusters under one label, means the table has gone stale).
    # --by-cluster: cluster every opponent deck and print the W/L table keyed by
    #   CLUSTER instead of by hand-written label. This is what you want whenever
    #   the audit shows one label covering several clusters: the label's WR is an
    #   average over decks that play differently (e.g. "crustle" has covered both
    #   the real Dwebble/Crustle wall and the Great Tusk library-out deck), and
    #   only the per-cluster split tells you which of them you actually lose to.
    #   Implied by --audit.

classify() is also the single archetype source of truth for the forensic
pipeline — scripts/replay_digest.py imports it, so a matchup is named the same
way in the WR report and in the per-game digests. Add new archetypes HERE only.

Writes <out_dir>/archetype_index.csv and prints a W/L-by-archetype summary,
plus a W/L-by-opponent-rating-band summary (100-wide bands 0-100..1200-1300,
bands with 0 games omitted). Rating bands need opp_team_id in index.csv
(added by download_episodes.py) joined against a public-leaderboard CSV
snapshot — auto-downloaded into <out_dir> if none is present there yet.
"""
import csv
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster_archetypes import cluster_and_name, load_card_meta

# Key-card id sets for archetypes seen so far in this skill-band's meta.
# Extend this dict when a new archetype shows up as UNKNOWN with unfamiliar key cards
# (check opp_key_cards in the printed detail / archetype_index.csv).
ARCHETYPE_IDS = {
    "crustle": {344, 345, 532, 533},      # Dwebble/Crustle (DRI + BLK prints)
    "mega_kangaskhan": {756},              # Mega Kangaskhan ex
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
    "ogerpon_teal": {95, 96, 349},         # Teal Mask Ogerpon (+ex, + DRI reprint)
    "ogerpon_hearthflame": {99, 358},      # Hearthflame Mask Ogerpon (ex + DRI reprint)
    "ogerpon_wellspring": {108, 370},      # Wellspring Mask Ogerpon (ex + DRI reprint)
    "ogerpon_cornerstone": {117, 386},     # Cornerstone Mask Ogerpon (ex + DRI reprint)
    "lopunny": {849},                      # Mega Lopunny ex
    "garchomp": {379, 380, 381},           # Cynthia's Gible/Gabite/Garchomp ex
}

# Name-keyword fallback for archetypes not covered by id sets above.
# ORDER MATTERS — dict insertion order is the match order, so a broad/generic
# keyword must sit below every specific one it would otherwise swallow.
NAME_KEYWORDS = {
    "starmie": ["Starmie"],
    "hop_trevenant": ["Trevenant", "Zacian", "Hop's Cramorant", "Hop's Phantump"],
    "slowking": ["Slowking"],
    "dipplin": ["Dipplin", "Hydrapple", "Applin"],
    "bellibolt": ["Bellibolt", "Iono"],
    "zoroark": ["Zoroark"],
    "gholdengo": ["Gholdengo"],
    "charizard": ["Charizard"],
    "raging_bolt": ["Raging Bolt"],
    "miraidon": ["Miraidon"],
    "gardevoir": ["Gardevoir"],
    "pidgeot": ["Pidgeot"],
    "ceruledge": ["Ceruledge"],
    "palafin": ["Palafin"],
    "dunsparce": ["Dunsparce", "Dudunsparce", "Hop's Snorlax"],
    # Generic tech card, only ever a last resort — see the id-table note above.
    # Must stay LAST so it never outranks a real archetype's own line.
    "fezandipiti_multibox": ["Fezandipiti ex"],
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
    """Archetype label for one 60-card list, or None if nothing matches.

    Card names arrive with typographic apostrophes ("Hop’s Snorlax") while the
    keyword tables are written with straight ones, so both sides are normalised
    before matching — otherwise every possessive-named line silently misses.
    """
    for arche, ids in ARCHETYPE_IDS.items():
        if set(deck_ids) & ids:
            return arche
    joined = " ".join(deck_names).replace("’", "'")
    for arche, kws in NAME_KEYWORDS.items():
        for kw in kws:
            if kw.replace("’", "'") in joined:
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
        for zpath in out_dir.glob("*.zip"):
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(out_dir)
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

    args = sys.argv[1:]
    do_cluster = "--cluster-unknown" in args
    do_audit = "--audit" in args
    do_by_cluster = "--by-cluster" in args or do_audit
    positional = [a for a in args if not a.startswith("--")]
    out_dir = Path(positional[0])
    replay_dir = out_dir / "replays"
    index_csv = out_dir / "index.csv"
    rows = list(csv.DictReader(open(index_csv)))
    results = []
    opp_decks = {}   # index into results -> that opponent's 60 card IDs
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
        opp_decks[len(results)] = [c["id"] for c in opp_deck]
        results.append({
            **row,
            "archetype": arche or "UNKNOWN",
            "opp_key_cards": top_names,
        })

    if not results:
        print("No episodes in index.csv")
        return

    if do_cluster or do_by_cluster:
        meta = load_card_meta()
        target = (sorted(opp_decks) if do_by_cluster
                  else [i for i in sorted(opp_decks) if results[i]["archetype"] == "UNKNOWN"])
        labels = cluster_and_name([opp_decks[i] for i in target], meta)
        for i, lab in zip(target, labels):
            results[i]["cluster"] = lab
            if do_cluster and results[i]["archetype"] == "UNKNOWN":
                results[i]["archetype"] = f"auto:{lab}"
        for r in results:
            r.setdefault("cluster", "")

    if do_audit:
        print("=== AUDIT: cosine clusters vs hand-written labels ===")
        print("(a cluster spread over several labels, or one label covering several "
              "clusters, is where the key-card table has drifted from the field)\n")
        by_cluster = {}
        for i in sorted(opp_decks):
            by_cluster.setdefault(results[i]["cluster"], []).append(results[i]["archetype"])
        for clust, labs in sorted(by_cluster.items(), key=lambda kv: -len(kv[1])):
            spread = ", ".join(f"{lab} x{n}" for lab, n in Counter(labs).most_common())
            print(f"  {len(labs):3d} decks  cluster={clust:<28} -> {spread}")
        print()

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

    def print_arche_table(rows_w, rows_l, key="archetype"):
        by_arche = Counter(r.get(key) or "?" for r in rows_w + rows_l)
        for arche in sorted(by_arche, key=lambda a: -by_arche[a]):
            aw = sum(1 for r in rows_w if (r.get(key) or "?") == arche)
            al = sum(1 for r in rows_l if (r.get(key) or "?") == arche)
            total = aw + al
            print(f"  {arche}: {aw}W-{al}L  ({100*aw/total:.0f}%)" if total else f"  {arche}: 0")

    print("\n=== COMBINED WR by archetype (W-L, WR%) ===")
    print_arche_table(wins, losses)

    if do_by_cluster:
        # The label table above averages over every deck sharing a label; this one
        # splits by what the decks actually ARE. Where the two disagree, trust this.
        print("\n=== COMBINED WR by cosine cluster (W-L, WR%) ===")
        print_arche_table(wins, losses, key="cluster")

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
                bl_rows = [r for r in rows_l if rating_band(r["opp_rating"]) == band]
                bl = len(bl_rows)
                total = bw + bl
                if total:
                    lost_to = Counter(r["archetype"] for r in bl_rows)
                    lost_str = ", ".join(f"{k} x{v}" for k, v in lost_to.most_common())
                    suffix = f"  | lost to: {lost_str}" if lost_str else ""
                    print(f"  {band}: {bw}W-{bl}L  ({100*bw/total:.0f}%){suffix}")

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
        clust = f" cluster=[{r['cluster']}]" if r.get("cluster") else ""
        print(f"  ep{r['episode_id']} copy{r['copy']} {res} opp={tag} "
              f"arche={r['archetype']}{clust} cards=[{r['opp_key_cards']}]")


if __name__ == "__main__":
    main()
