"""
crustle_list_recon.py — Find the BEST (most consistent) Crustle decklist
among top-player + all_teams replays, for IL cloning.

Outputs:
  - All Crustle list clusters with Pokemon count, search cards, winrate, IL volume
  - Comparison vs our frozen v009a list
  - Recommendation

Usage:
    .venv/bin/python src/eval/crustle_list_recon.py
"""

import json
import sys
import csv
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).parent.parent.parent
EN_CARD_CSV = REPO / "data" / "EN_Card_Data.csv"
V009A_DECK_CSV = REPO / "submissions" / "v009a_crustle" / "deck.csv"
REPLAY_DIRS = [
    REPO / "data" / "episodes" / "top_players",
    REPO / "data" / "episodes" / "all_teams",
    REPO / "data" / "episodes",
]

sys.path.insert(0, str(REPO / "src"))
from replays.parse import walk_replays, parse_replay, _CONTEXT_NAMES

DWEBBLE_ID = 344
CRUSTLE_ID = 345

# Known search/draw/consistency trainer IDs and keywords
# We'll identify these by name since we have the CSV
SEARCH_DRAW_KEYWORDS = [
    "ultra ball", "nest ball", "poke ball", "hyper aroma",
    "buddy-buddy poffin", "arven", "iono", "professor's research",
    "n's plan", "enriching energy", "professor", "pokémon communication",
    "ball", "poffin", "lanette", "hisuian heavy ball",
]

POKEMON_STAGES = {"Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"}
ENERGY_STAGES = {"Basic Energy", "Special Energy"}


def load_card_info():
    info = {}
    with open(EN_CARD_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cid = int(row["Card ID"].strip())
            except (ValueError, KeyError):
                continue
            info[cid] = {
                "name": row.get("Card Name", "").strip(),
                "stage": row.get("Stage (Pokémon)/Type (Energy and Trainer)", "").strip(),
                "category": row.get("Category", "").strip(),
            }
    return info


def is_pokemon(cid, card_info):
    return card_info.get(cid, {}).get("stage", "") in POKEMON_STAGES


def is_search_draw(cid, card_info):
    name = card_info.get(cid, {}).get("name", "").lower()
    return any(kw in name for kw in SEARCH_DRAW_KEYWORDS)


def deck_pokemon_count(deck, card_info):
    return sum(1 for cid in deck if is_pokemon(cid, card_info))


def deck_search_count(deck, card_info):
    return sum(1 for cid in deck if is_search_draw(cid, card_info))


def extract_deck_from_replay(path):
    """Returns {player_idx: [card_id x60], ...}"""
    with open(path) as f:
        data = json.load(f)
    steps = data.get("steps", [])
    if len(steps) < 2:
        return {}
    step1 = steps[1]
    decks = {}
    for pi in range(min(2, len(step1))):
        action = step1[pi].get("action", [])
        if len(action) == 60 and all(isinstance(x, int) for x in action):
            decks[pi] = action
    return decks


def deck_cluster_key(deck, threshold=58):
    """
    Returns a frozenset-based key for "near-identical" clustering (>= threshold shared).
    We use sorted tuple as exact key, then cluster near-identical ones post-hoc.
    """
    return tuple(sorted(deck))


def cluster_near_identical(exact_decks, threshold=58):
    """
    Given a list of (exact_key, records) pairs, merge clusters where
    two exact decks share >= threshold cards out of 60.
    Returns list of merged clusters: each cluster is a list of exact_keys.
    """
    keys = list(exact_decks.keys())
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Compare each pair
    for i, k1 in enumerate(keys):
        c1 = Counter(k1)
        for j, k2 in enumerate(keys):
            if j <= i:
                continue
            c2 = Counter(k2)
            # shared = sum of min counts
            shared = sum(min(c1[x], c2[x]) for x in c1)
            if shared >= threshold:
                union(k1, k2)

    # Group by root
    clusters = defaultdict(list)
    for k in keys:
        clusters[find(k)].append(k)
    return list(clusters.values())


def load_v009a_deck():
    if not V009A_DECK_CSV.exists():
        return []
    with open(V009A_DECK_CSV) as f:
        ids = [int(line.strip()) for line in f if line.strip()]
    return ids


def format_decklist(deck, card_info):
    counts = Counter(deck)
    lines = []
    for cid, cnt in sorted(counts.items(), key=lambda x: (
        0 if is_pokemon(x[0], card_info) else 1,
        -x[1], x[0]
    )):
        name = card_info.get(cid, {}).get("name", f"id={cid}")
        stage = card_info.get(cid, {}).get("stage", "?")
        lines.append(f"  {cnt}x {name} (id={cid}) [{stage}]")
    return "\n".join(lines)


def main():
    print("=" * 72)
    print("  CRUSTLE LIST RECON: Find Most Consistent Crustle Decklist")
    print("=" * 72)

    card_info = load_card_info()
    print(f"  Loaded {len(card_info)} cards from EN_Card_Data.csv")

    # Walk replays (deduplicated)
    all_dirs = REPLAY_DIRS
    all_paths = walk_replays(all_dirs)
    seen = set()
    paths = []
    for p in all_paths:
        if p.name not in seen:
            seen.add(p.name)
            paths.append(p)
    print(f"  Found {len(paths)} unique replay files")

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Extract all decks + game outcomes
    # ─────────────────────────────────────────────────────────────────────
    print("\n[1] Extracting decks from replays...")
    game_records = []
    errors = 0
    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
            ep_id = data.get("id") or data.get("info", {}).get("EpisodeId", 0)
            agents = data.get("info", {}).get("Agents", [])
            rewards = data.get("rewards", [])
            decks = extract_deck_from_replay(path)
            for pi, deck in decks.items():
                player_name = agents[pi]["Name"] if pi < len(agents) else f"player{pi}"
                won = rewards[pi] > 0 if pi < len(rewards) else False
                game_records.append({
                    "ep_id": ep_id, "pi": pi, "player": player_name,
                    "deck": deck, "won": won, "path": path,
                })
        except Exception as e:
            errors += 1
    print(f"  Extracted {len(game_records)} player-game records ({errors} errors)")

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Filter to Crustle decks
    # ─────────────────────────────────────────────────────────────────────
    print("\n[2] Filtering to Crustle-bearing decks (contain id 344 or 345)...")
    crustle_records = [
        r for r in game_records
        if DWEBBLE_ID in r["deck"] or CRUSTLE_ID in r["deck"]
    ]
    print(f"  Crustle decks: {len(crustle_records)} / {len(game_records)} player-games")

    if not crustle_records:
        print("ERROR: No Crustle decks found!")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Exact deck clustering + then merge near-identical
    # ─────────────────────────────────────────────────────────────────────
    print("\n[3] Clustering Crustle decks (exact then >=58/60 merge)...")

    exact_decks = defaultdict(list)
    for r in crustle_records:
        key = deck_cluster_key(r["deck"])
        exact_decks[key].append(r)

    print(f"  Distinct exact decklists: {len(exact_decks)}")

    # Merge near-identical (>=58/60)
    clusters = cluster_near_identical(exact_decks, threshold=58)
    print(f"  After >=58/60 merge: {len(clusters)} clusters")

    # Build cluster info
    cluster_data = []
    for cluster_keys in clusters:
        recs = []
        for k in cluster_keys:
            recs.extend(exact_decks[k])

        # Representative deck: most common exact deck in cluster
        freq = Counter(k for k in cluster_keys for _ in exact_decks[k])
        rep_key = freq.most_common(1)[0][0]
        rep_deck = list(rep_key)

        games = len(recs)
        wins = sum(1 for r in recs if r["won"])
        winrate = wins / games if games else 0
        players = sorted(set(r["player"] for r in recs))
        n_pokemon = deck_pokemon_count(rep_deck, card_info)
        n_search = deck_search_count(rep_deck, card_info)

        cluster_data.append({
            "keys": cluster_keys,
            "records": recs,
            "rep_deck": rep_deck,
            "games": games,
            "wins": wins,
            "winrate": winrate,
            "players": players,
            "n_pokemon": n_pokemon,
            "n_search": n_search,
            "n_exact_variants": len(cluster_keys),
        })

    # Sort by games desc
    cluster_data.sort(key=lambda x: -x["games"])

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Count winner-MAIN decisions per cluster
    # ─────────────────────────────────────────────────────────────────────
    print("\n[4] Counting winner-MAIN (ctx=0) decisions per cluster...")

    # Map (ep_id, pi) -> cluster index for quick lookup
    ep_pi_to_cluster = {}
    for ci, cd in enumerate(cluster_data):
        for r in cd["records"]:
            ep_pi_to_cluster[(r["ep_id"], r["pi"])] = ci

    # Collect all unique paths across all crustle records
    all_crustle_paths = list(set(r["path"] for r in crustle_records))

    # Initialize decision counters per cluster
    for cd in cluster_data:
        cd["winner_main_decisions"] = 0
        cd["total_decisions"] = 0

    parse_errors2 = 0
    for path in all_crustle_paths:
        try:
            with open(path) as f:
                data = json.load(f)
            ep_id = data.get("id") or data.get("info", {}).get("EpisodeId", 0)
            pairs = parse_replay(path)
            for pair in pairs:
                pi = pair["player_index"]
                ci = ep_pi_to_cluster.get((ep_id, pi))
                if ci is None:
                    continue
                cluster_data[ci]["total_decisions"] += 1
                if pair["won"] and pair["select_context"] == 0:
                    cluster_data[ci]["winner_main_decisions"] += 1
        except Exception as e:
            parse_errors2 += 1
    print(f"  Done. Parse errors: {parse_errors2}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Load v009a frozen deck for comparison
    # ─────────────────────────────────────────────────────────────────────
    v009a_deck = load_v009a_deck()
    if v009a_deck:
        v009a_n_pokemon = deck_pokemon_count(v009a_deck, card_info)
        v009a_n_search = deck_search_count(v009a_deck, card_info)
        # Find its cluster
        v009a_key = deck_cluster_key(v009a_deck)
        v009a_cluster_idx = None
        for ci, cd in enumerate(cluster_data):
            if v009a_key in cd["keys"]:
                v009a_cluster_idx = ci
                break
        # If not exact match, find nearest by overlap
        if v009a_cluster_idx is None:
            v009a_c = Counter(v009a_key)
            best_overlap = 0
            for ci, cd in enumerate(cluster_data):
                rep_c = Counter(cd["rep_deck"])
                overlap = sum(min(v009a_c[x], rep_c[x]) for x in v009a_c)
                if overlap > best_overlap:
                    best_overlap = overlap
                    v009a_cluster_idx = ci
            if best_overlap < 55:
                v009a_cluster_idx = None
    else:
        v009a_n_pokemon = v009a_n_search = None
        v009a_cluster_idx = None

    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Print results
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  ALL CRUSTLE LIST CLUSTERS (sorted by games)")
    print("=" * 72)
    print(f"  {'#':>2}  {'Games':>5}  {'WR':>6}  {'Pkmn':>5}  {'Srch':>5}  "
          f"{'WinMain':>8}  {'Variants':>8}  Players (top 3)")
    print("  " + "-" * 72)

    for ci, cd in enumerate(cluster_data):
        top_players = cd["players"][:3]
        is_v009a = (ci == v009a_cluster_idx)
        marker = " ◄ v009a" if is_v009a else ""
        print(f"  {ci:>2}  {cd['games']:>5}  {cd['winrate']:>5.1%}  "
              f"{cd['n_pokemon']:>5}  {cd['n_search']:>5}  "
              f"{cd['winner_main_decisions']:>8}  {cd['n_exact_variants']:>8}  "
              f"{', '.join(top_players[:3])}{marker}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 7: Detailed view of top clusters + comparison
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  DETAILED DECKLISTS — TOP CLUSTERS + v009a COMPARISON")
    print("=" * 72)

    # v009a comparison
    if v009a_deck:
        print(f"\n--- CURRENT FROZEN v009a Crustle list ---")
        print(f"  Pokemon count: {v009a_n_pokemon}  |  Search/draw count: {v009a_n_search}")
        if v009a_cluster_idx is not None:
            cd = cluster_data[v009a_cluster_idx]
            print(f"  In data: cluster #{v009a_cluster_idx}, {cd['games']} games, "
                  f"winrate={cd['winrate']:.1%}, winner-MAIN={cd['winner_main_decisions']}")
        else:
            print(f"  Not found in replay data (not played by tracked players)")
        print(f"\n  Decklist:")
        print(format_decklist(v009a_deck, card_info))

    # Identify candidates: Crustle clusters with more Pokemon and/or more search than v009a
    print("\n" + "=" * 72)
    print("  SHORTLIST: Clusters with >=games, >=winrate, MORE consistent than v009a")
    print("=" * 72)

    ref_pokemon = v009a_n_pokemon if v009a_n_pokemon else 8
    ref_search = v009a_n_search if v009a_n_search else 8

    candidates = []
    for ci, cd in enumerate(cluster_data):
        if ci == v009a_cluster_idx:
            continue
        # Must have at least some games and winrate
        if cd["games"] < 3:
            continue
        # More consistent: more Pokemon OR more search cards
        more_pokemon = cd["n_pokemon"] > ref_pokemon
        more_search = cd["n_search"] > ref_search
        if more_pokemon or more_search:
            candidates.append((ci, cd))

    # Sort candidates by: winner_main_decisions * winrate (weighted score)
    candidates.sort(key=lambda x: -(x[1]["winner_main_decisions"] * x[1]["winrate"]))

    if not candidates:
        print("\n  NO more-consistent Crustle clusters found in data.")
        print("  Our frozen v009a list appears to be near-best in the field.")
        print("  (All other Crustle clusters have fewer Pokemon and/or search cards.)")
    else:
        print(f"\n  Found {len(candidates)} more-consistent cluster(s) than v009a:\n")

    for rank, (ci, cd) in enumerate(candidates[:5], 1):
        print(f"\n  --- CANDIDATE #{rank} (cluster {ci}) ---")
        print(f"  Pokemon: {cd['n_pokemon']} (vs v009a: {ref_pokemon}, delta +{cd['n_pokemon']-ref_pokemon})")
        print(f"  Search/draw cards: {cd['n_search']} (vs v009a: {ref_search}, delta +{cd['n_search']-ref_search})")
        print(f"  Games: {cd['games']}  |  Winrate: {cd['winrate']:.1%}  |  "
              f"Winner-MAIN decisions: {cd['winner_main_decisions']}")
        print(f"  Exact variants in cluster: {cd['n_exact_variants']}")
        print(f"  Players: {', '.join(cd['players'])}")
        print(f"\n  Decklist:")
        print(format_decklist(cd["rep_deck"], card_info))

    # ─────────────────────────────────────────────────────────────────────
    # Step 8: Recommendation
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RECOMMENDATION")
    print("=" * 72)

    if not candidates:
        print("""
  Our frozen v009a Crustle list is already among the most consistent
  Crustle lists in the replay data. No other Crustle cluster has meaningfully
  more Pokemon or search cards.

  RECOMMENDATION: Keep v009a decklist for IL cloning — the limiting factor
  is not the decklist but data volume and model quality.
""")
    else:
        best_ci, best_cd = candidates[0]
        print(f"""
  Best clone target: Cluster #{best_ci}
  - Pokemon: {best_cd['n_pokemon']} (+{best_cd['n_pokemon']-ref_pokemon} vs v009a {ref_pokemon})
  - Search cards: {best_cd['n_search']} (+{best_cd['n_search']-ref_search} vs v009a {ref_search})
  - Winrate: {best_cd['winrate']:.1%}
  - Games: {best_cd['games']}
  - Winner-MAIN decisions: {best_cd['winner_main_decisions']}
  - Players: {', '.join(best_cd['players'][:5])}

  Rationale: Best balance of consistency improvement, winrate, and IL training
  volume. More Pokemon means fewer mulligans and more reliable game starts.
""")

    # ─────────────────────────────────────────────────────────────────────
    # Step 9: Full dump of all cluster summary data
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  FULL CLUSTER SUMMARY (for all Crustle clusters)")
    print("=" * 72)
    for ci, cd in enumerate(cluster_data):
        is_v009a = (ci == v009a_cluster_idx)
        print(f"\nCluster {ci}{' [v009a]' if is_v009a else ''}:")
        print(f"  Games={cd['games']}, WR={cd['winrate']:.1%}, "
              f"Pkmn={cd['n_pokemon']}, Search={cd['n_search']}, "
              f"WinnerMain={cd['winner_main_decisions']}, "
              f"Variants={cd['n_exact_variants']}")
        print(f"  Players: {', '.join(cd['players'])}")
        # Show Pokemon composition of representative deck
        rep_counts = Counter(cd["rep_deck"])
        pokemon_in_deck = [(cid, cnt) for cid, cnt in rep_counts.items()
                           if is_pokemon(cid, card_info)]
        pokemon_str = ", ".join(
            f"{cnt}x{card_info.get(cid,{}).get('name','?')}"
            for cid, cnt in sorted(pokemon_in_deck, key=lambda x: -x[1])
        )
        print(f"  Pokemon: {pokemon_str}")
        search_in_deck = [(cid, cnt) for cid, cnt in rep_counts.items()
                          if is_search_draw(cid, card_info)]
        search_str = ", ".join(
            f"{cnt}x{card_info.get(cid,{}).get('name','?')}"
            for cid, cnt in sorted(search_in_deck, key=lambda x: -x[1])
        )
        print(f"  Search/draw: {search_str}")


if __name__ == "__main__":
    main()
