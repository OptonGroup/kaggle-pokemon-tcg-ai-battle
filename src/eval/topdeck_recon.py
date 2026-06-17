"""
topdeck_recon.py — Deck clustering + IL viability analysis for top-player replays.

Answers:
  1. Deck archetype distribution
  2. Dominant deck (best 60-card reconstruction)
  3. Data volume for IL (games + decisions by context)
  4. Deck legality
  5. Win rate of dominant deck
  6. Recommendation

Usage:
    .venv/bin/python src/eval/topdeck_recon.py
"""

import json
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path

# Repo root
REPO = Path(__file__).parent.parent.parent
TOP_PLAYERS_DIR = REPO / "data" / "episodes" / "top_players"
EN_CARD_CSV = REPO / "data" / "EN_Card_Data.csv"
PARSE_MODULE = REPO / "src"
sys.path.insert(0, str(PARSE_MODULE))

from replays.parse import parse_replay, revealed_cards, walk_replays, _CONTEXT_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# Card name lookup
# ─────────────────────────────────────────────────────────────────────────────

def load_card_names() -> dict[int, str]:
    """Load card_id → name from EN_Card_Data.csv."""
    names: dict[int, str] = {}
    with open(EN_CARD_CSV) as f:
        header = f.readline().strip().split(",")
        id_idx = 0   # "Card ID"
        name_idx = 1  # "Card Name"
        for line in f:
            parts = line.strip().split(",")
            if len(parts) > name_idx:
                try:
                    cid = int(parts[id_idx])
                    names[cid] = parts[name_idx]
                except ValueError:
                    pass
    return names


def load_card_info() -> dict[int, dict]:
    """Load card_id → {name, stage, category} from EN_Card_Data.csv."""
    info: dict[int, dict] = {}
    with open(EN_CARD_CSV) as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
        for line in f:
            parts = line.strip().split(",")
            if not parts or not parts[0].strip():
                continue
            try:
                cid = int(parts[0].strip())
            except ValueError:
                continue
            row = {}
            for i, h in enumerate(header):
                row[h] = parts[i].strip() if i < len(parts) else ""
            info[cid] = row
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Deck extraction from replays
# ─────────────────────────────────────────────────────────────────────────────

def extract_deck_from_replay(path: Path) -> dict[int, list[int]]:
    """
    Extract the submitted deck for each player from a replay.
    The deck is the 60-card action in steps[1][player].action.
    Returns {player_idx: [card_id, ...]} (list of 60 ids).
    """
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


def deck_signature(deck: list[int], card_info: dict[int, dict]) -> str:
    """
    Create a canonical archetype signature:
    Use the top 3 most-frequent Pokémon card IDs (stage contains 'Pokémon').
    Falls back to top-3 non-energy card IDs if no Pokémon identified.
    """
    counts = Counter(deck)
    # Pokémon stages: "Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"
    POKEMON_STAGES = {"Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"}
    pokemon_ids = []
    for cid, cnt in counts.most_common():
        info = card_info.get(cid, {})
        stage = info.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
        if stage in POKEMON_STAGES:
            pokemon_ids.append((cid, cnt))

    if not pokemon_ids:
        # fallback: exclude energies, take top-3 by count
        non_energy = [
            (cid, cnt) for cid, cnt in counts.most_common()
            if card_info.get(cid, {}).get("Stage (Pokémon)/Type (Energy and Trainer)", "") not in
               {"Basic Energy", "Special Energy"}
        ]
        if not non_energy:
            return str(sorted([cid for cid, _ in counts.most_common(3)]))
        top3fb = sorted(non_energy[:3], key=lambda x: (-x[1], x[0]))
        return str([cid for cid, _ in top3fb])

    # Sort by count desc, then id asc for stability; take top-3
    pokemon_ids_sorted = sorted(pokemon_ids, key=lambda x: (-x[1], x[0]))
    top3 = pokemon_ids_sorted[:3]
    return str([cid for cid, _ in top3])


def archetype_name(sig: str, card_names: dict[int, str]) -> str:
    """Convert a signature string back to a human-readable archetype name."""
    import ast
    try:
        ids = ast.literal_eval(sig)
    except Exception:
        return sig
    parts = [card_names.get(cid, f"id{cid}") for cid in ids]
    return " / ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  TOP-DECK RECON: Deck Clustering + IL Viability")
    print("=" * 70)

    # Load supporting data
    card_names = load_card_names()
    card_info = load_card_info()
    print(f"\n  Loaded {len(card_names)} card name entries from EN_Card_Data.csv")

    # Walk replay files
    replay_paths = walk_replays([TOP_PLAYERS_DIR])
    print(f"  Found {len(replay_paths)} replay files in {TOP_PLAYERS_DIR}")

    if not replay_paths:
        print("ERROR: No replays found. Check path.")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Extract decks from all replays
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/6] Extracting decks from replays...")

    # per_game: list of {episodeId, player_idx, player_name, deck, won}
    game_records = []
    parse_errors = 0

    for path in replay_paths:
        try:
            with open(path) as f:
                data = json.load(f)
            episode_id = data.get("id") or data.get("info", {}).get("EpisodeId", 0)
            agents = data.get("info", {}).get("Agents", [])
            rewards = data.get("rewards", [])
            decks = extract_deck_from_replay(path)

            for pi, deck in decks.items():
                player_name = agents[pi]["Name"] if pi < len(agents) else f"player{pi}"
                won = rewards[pi] > 0 if pi < len(rewards) else False
                game_records.append({
                    "episodeId": episode_id,
                    "player_idx": pi,
                    "player_name": player_name,
                    "deck": deck,
                    "won": won,
                    "path": path,
                })
        except Exception as e:
            parse_errors += 1
            print(f"  WARN: {path.name}: {e}")

    print(f"  Extracted {len(game_records)} player-game deck records from {len(replay_paths) - parse_errors} games")
    print(f"  Parse errors: {parse_errors}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Cluster decks into archetypes
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/6] Clustering decks into archetypes...")

    for rec in game_records:
        rec["sig"] = deck_signature(rec["deck"], card_info)

    # Count archetypes
    sig_counts: Counter = Counter()
    sig_to_records: dict[str, list] = defaultdict(list)
    for rec in game_records:
        sig_counts[rec["sig"]] += 1
        sig_to_records[rec["sig"]].append(rec)

    print(f"\n  Found {len(sig_counts)} distinct deck archetypes (by top-3 Pokémon)\n")
    print(f"  {'Archetype':<45} {'Games':>6} {'Players':>8} {'WinRate':>9}  {'TopSig'}")
    print("  " + "-" * 90)

    sorted_sigs = sig_counts.most_common()
    for i, (sig, count) in enumerate(sorted_sigs[:20]):
        records = sig_to_records[sig]
        players = len(set(r["player_name"] for r in records))
        wins = sum(1 for r in records if r["won"])
        winrate = wins / count if count else 0
        name = archetype_name(sig, card_names)
        print(f"  {name:<45} {count:>6}  {players:>8}   {winrate:>7.1%}   {sig}")

    if len(sorted_sigs) > 20:
        rest = sum(c for _, c in sorted_sigs[20:])
        print(f"  ... {len(sorted_sigs)-20} more archetypes with {rest} total game-slots")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Dominant deck
    # ─────────────────────────────────────────────────────────────────────────
    top_sig, top_count = sorted_sigs[0]
    top_records = sig_to_records[top_sig]
    top_name = archetype_name(top_sig, card_names)

    print(f"\n[3/6] Dominant archetype: '{top_name}'  ({top_count} game-slots)")

    # Reconstruct best 60-card decklist by aggregating all decks in this archetype
    # Strategy: collect all submitted decks; the most common deck (or majority-vote per slot)
    all_top_decks = [tuple(sorted(r["deck"])) for r in top_records]
    deck_freq: Counter = Counter(all_top_decks)
    most_common_deck_tuple, most_common_freq = deck_freq.most_common(1)[0]
    most_common_deck = list(most_common_deck_tuple)

    print(f"\n  Distinct 60-card deck variants within archetype: {len(deck_freq)}")
    print(f"  Most common exact deck appears in: {most_common_freq}/{top_count} slots")

    # Also try majority-vote reconstruction
    # For each of the 60 positions (after sorting), take the mode
    all_top_deck_lists = [sorted(r["deck"]) for r in top_records]
    # Aggregate as multiset: most common count for each card ID
    combined_counter: Counter = Counter()
    for d in all_top_deck_lists:
        combined_counter.update(Counter(d))

    # Average count per card
    n_decks = len(all_top_deck_lists)
    avg_counts = {cid: cnt / n_decks for cid, cnt in combined_counter.items()}

    # Round to nearest int and try to fit in 60
    # Majority-vote: if avg >= 0.5, include; adjust to sum=60
    candidate_counts = {cid: round(cnt) for cid, cnt in avg_counts.items() if round(cnt) >= 1}
    total = sum(candidate_counts.values())

    print(f"\n  Majority-vote reconstruction: {len(candidate_counts)} distinct card IDs, total copies={total}")
    print(f"  (60 = target; diff = {total - 60:+d})")

    # Report the most common exact deck as the "best" decklist
    best_deck_counter = Counter(most_common_deck)
    print(f"\n  --- Best Reconstructed Decklist (most frequent exact submission) ---")
    print(f"  {'Card ID':>8} {'Name':<40} {'Count':>6}  {'Stage/Type'}")
    print("  " + "-" * 80)
    total_cards = 0
    for cid, cnt in sorted(best_deck_counter.items(), key=lambda x: -x[1]):
        info = card_info.get(cid, {})
        name = card_names.get(cid, f"[id={cid}]")
        stage = info.get("Stage (Pokémon)/Type (Energy and Trainer)", "?")
        print(f"  {cid:>8}  {name:<40} {cnt:>5}   {stage}")
        total_cards += cnt
    print(f"  {'':>8}  {'TOTAL':<40} {total_cards:>5}")

    # Confidence note
    if most_common_freq == n_decks:
        confidence = "EXACT — all players used identical deck"
    elif most_common_freq / n_decks >= 0.8:
        confidence = f"HIGH — {most_common_freq/n_decks:.0%} of players used exact same deck"
    elif most_common_freq / n_decks >= 0.5:
        confidence = f"MEDIUM — {most_common_freq/n_decks:.0%} used exact deck; others are variants"
    else:
        confidence = f"LOW — only {most_common_freq/n_decks:.0%} used exact deck; many variants"
    print(f"\n  Reconstruction confidence: {confidence}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Data volume for dominant-deck IL
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[4/6] Data volume for dominant-deck IL...")

    # Find all replay paths where at least one player pilots the dominant archetype
    top_paths = list(set(r["path"] for r in top_records))
    print(f"  Games (replays) containing dominant-deck player: {len(top_paths)}")

    # Parse decisions from those replays, but only for the player(s) using top deck
    top_player_episodes: dict[tuple, list] = {}  # (episodeId, player_idx) -> records
    for rec in top_records:
        key = (rec["episodeId"], rec["player_idx"])
        top_player_episodes[key] = rec

    # Parse decisions
    total_decisions = 0
    context_counter: Counter = Counter()
    win_decisions = 0
    loss_decisions = 0
    main_decisions = 0
    parse_errs2 = 0

    for path in top_paths:
        try:
            pairs = parse_replay(path)
            with open(path) as f:
                data = json.load(f)
            ep_id = data.get("id") or data.get("info", {}).get("EpisodeId", 0)

            for pair in pairs:
                pi = pair["player_index"]
                key = (ep_id, pi)
                if key not in top_player_episodes:
                    continue  # other player, not our archetype
                total_decisions += 1
                ctx = pair["select_context"]
                context_counter[ctx] += 1
                if pair["won"]:
                    win_decisions += 1
                else:
                    loss_decisions += 1
                if ctx == 0:
                    main_decisions += 1
        except Exception as e:
            parse_errs2 += 1
            print(f"  WARN: {path.name}: {e}")

    print(f"  Total decisions (dominant-deck player only): {total_decisions}")
    print(f"  Win decisions: {win_decisions}  |  Loss decisions: {loss_decisions}")
    print(f"  MAIN (ctx=0) decisions: {main_decisions}")
    print(f"\n  Context breakdown (dominant-deck decisions):")
    print(f"  {'Ctx':>5} {'Name':<30} {'Count':>7}  {'%':>6}")
    print("  " + "-" * 55)
    for ctx, cnt in context_counter.most_common(20):
        name = _CONTEXT_NAMES.get(ctx, f"CTX_{ctx}")
        pct = 100 * cnt / total_decisions if total_decisions else 0
        print(f"  {ctx:>5}  {name:<30} {cnt:>7}  {pct:>5.1f}%")

    # Threshold commentary
    print(f"\n  IL threshold: MAIN decisions >= 2000 = comfortable, 1000-2000 = workable, <500 = thin")
    if main_decisions >= 2000:
        threshold_label = "COMFORTABLE (>=2000)"
    elif main_decisions >= 1000:
        threshold_label = "WORKABLE (1000-2000)"
    elif main_decisions >= 500:
        threshold_label = "BORDERLINE (500-1000)"
    else:
        threshold_label = "THIN (<500)"
    print(f"  => {main_decisions} MAIN decisions: {threshold_label}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Deck legality check
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[5/6] Deck legality check for dominant deck...")

    # Known Ace Spec IDs (we know 1159 = Hero's Cape; flag any card with count=1
    # that looks like it could be Ace Spec)
    KNOWN_ACE_SPEC = {1159}  # Hero's Cape
    # Also try to detect from card name
    ACE_SPEC_KEYWORDS = ["ace spec", "computer search", "dowsing machine", "gold potion",
                         "hero's cape", "unfair stamp", "maximum belt"]

    issues = []

    # Total cards
    total_best = sum(best_deck_counter.values())
    if total_best != 60:
        issues.append(f"FAIL: Total card count = {total_best} (expected 60)")
    else:
        print(f"  Total cards: {total_best} ✓")

    # Max 4 copies of any non-basic-energy card
    for cid, cnt in best_deck_counter.items():
        info = card_info.get(cid, {})
        stage = info.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
        if stage == "Basic Energy":
            continue
        if cnt > 4:
            name = card_names.get(cid, f"id={cid}")
            issues.append(f"FAIL: {name} (id={cid}) appears {cnt} times (max 4 for non-energy)")

    # Ace Spec check: <=1 copy total of any Ace Spec card
    ace_spec_in_deck = []
    for cid in best_deck_counter:
        info = card_info.get(cid, {})
        name = card_names.get(cid, "").lower()
        rule = info.get("Rule", "").lower()
        if cid in KNOWN_ACE_SPEC or any(k in name for k in ACE_SPEC_KEYWORDS) or "ace spec" in rule:
            ace_spec_in_deck.append(cid)

    if len(ace_spec_in_deck) > 1:
        names_str = ", ".join(f"{card_names.get(c,'?')} (id={c})" for c in ace_spec_in_deck)
        issues.append(f"WARN: Multiple Ace Spec cards detected: {names_str}")
    elif len(ace_spec_in_deck) == 1:
        cid = ace_spec_in_deck[0]
        print(f"  Ace Spec detected: {card_names.get(cid,'?')} (id={cid}), count={best_deck_counter[cid]} ✓")
    else:
        print(f"  No known Ace Spec cards in deck")

    # Check for at least 1 Basic Pokémon
    basic_pokemon = []
    for cid in best_deck_counter:
        info = card_info.get(cid, {})
        stage = info.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
        if stage == "Basic Pokémon":
            basic_pokemon.append(cid)

    if not basic_pokemon:
        issues.append("WARN: No Basic Pokémon detected in deck (may be a card data parsing issue)")
    else:
        print(f"  Basic Pokémon: {len(basic_pokemon)} types detected ✓")

    # Flag any card whose count exceeds what's in the whole data (artifact check)
    # A count > 4 for non-energy is already flagged; flag count > 4 period for non-energy
    for cid, cnt in best_deck_counter.items():
        info = card_info.get(cid, {})
        stage = info.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
        if stage != "Basic Energy" and cnt > 4:
            name = card_names.get(cid, f"id={cid}")
            issues.append(f"ARTIFACT?: {name} (id={cid}) has count={cnt}")

    if not issues:
        print("  Legality: PASS — no issues found")
    else:
        for issue in issues:
            print(f"  {issue}")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Win rate of dominant deck
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[6/6] Win signal for dominant deck...")

    top_wins = sum(1 for r in top_records if r["won"])
    top_losses = sum(1 for r in top_records if not r["won"])
    top_winrate = top_wins / top_count if top_count else 0
    print(f"  Games as dominant-deck: {top_count} ({top_wins} W / {top_losses} L)")
    print(f"  Win rate: {top_winrate:.1%}")

    # Top players
    player_game_counts: Counter = Counter(r["player_name"] for r in top_records)
    player_win_counts: Counter = Counter(r["player_name"] for r in top_records if r["won"])
    print(f"\n  Players piloting dominant deck (top 10 by games):")
    print(f"  {'Player':<30} {'Games':>6} {'Wins':>6} {'WinRate':>8}")
    print("  " + "-" * 56)
    for player, games in player_game_counts.most_common(10):
        wins = player_win_counts.get(player, 0)
        wr = wins / games if games else 0
        print(f"  {player:<30} {games:>6} {wins:>6} {wr:>7.1%}")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Recommendation
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  RECOMMENDATION")
    print(f"{'=' * 70}")

    total_game_slots = len(game_records)
    top_share = top_count / total_game_slots if total_game_slots else 0

    print(f"""
  Dominant deck: '{top_name}'
  Share of all player-game slots: {top_count}/{total_game_slots} = {top_share:.1%}
  Win rate: {top_winrate:.1%}
  Reconstruction confidence: {confidence}
  MAIN decisions available: {main_decisions}  [{threshold_label}]
  Total decisions: {total_decisions}

  Viability assessment:
""")

    if top_share < 0.15:
        print("  DIVERSE FIELD — no single dominant archetype (top deck <15% of games).")
        print("  'Copy the best deck' is risky: insufficient data for a reliable IL ranker.")
        print(f"  The closest candidate is '{top_name}' but with only {main_decisions} MAIN decisions.")
        print("  SUGGESTION: Harvest more targeted replays of the top archetype, OR")
        print("  use a deck-agnostic heuristic/MCTS approach instead of IL.")
    elif main_decisions < 500:
        print(f"  Archetype '{top_name}' is dominant ({top_share:.0%}) but data is THIN.")
        print(f"  Only {main_decisions} MAIN decisions — too few for reliable IL training.")
        print("  SUGGESTION: Harvest 5-10x more replays specifically of this archetype.")
    elif main_decisions < 1000:
        print(f"  Archetype '{top_name}' is dominant ({top_share:.0%}) and data is BORDERLINE.")
        print(f"  {main_decisions} MAIN decisions — marginal for IL. Could work with strong regularization.")
        print("  SUGGESTION: Harvest more replays OR use IL as a re-ranking signal on top of heuristics.")
    else:
        print(f"  VIABLE: '{top_name}' is dominant ({top_share:.0%}) with {main_decisions} MAIN")
        print(f"  decisions. This is {'comfortable' if main_decisions >= 2000 else 'workable'} for IL training.")
        print(f"  Win rate {top_winrate:.0%} and {top_count} games confirm it's the strongest archetype.")
        print(f"  BUILD THIS DECK and train a deck-specific IL ranker on its {main_decisions} MAIN decisions.")
        if len(deck_freq) > 1:
            print(f"  NOTE: {len(deck_freq)} deck variants exist; use majority-vote reconstruction if")
            print(f"  exact decklist isn't confirmed.")

    print()


if __name__ == "__main__":
    main()
