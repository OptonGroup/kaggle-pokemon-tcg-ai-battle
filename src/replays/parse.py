"""
parse.py — replay parser for Pokemon TCG AI Battle imitation learning (P0-4).

Works entirely from raw JSON (no libcg required) so it runs on macOS without Docker.

Replay format summary
---------------------
- Top-level keys: configuration, info, rewards, steps, ...
- info.Agents[i].Name: player name; rewards[i]: +1=win, -1=loss for player i
- steps: list of "step frames", each frame is a list of 2 player dicts:
    step[i][player] = {action, info, observation, reward, status, ...}
  - observation.select: None (no decision) or dict with:
      context (int, matches SelectContext enum), minCount, maxCount, option (list of option dicts)
  - action: list of ints — the RESPONSE to step[i-1]'s observation
    (i.e. the chosen positional indices into step[i-1].select.option)
  So training pair = (obs = step[i].observation, chosen = step[i+1].action)
  when step[i].observation.select is not None and step[i+1].action is non-empty.

SelectContext notable values (from cg/api.py):
  0=MAIN, 1=SETUP_ACTIVE_POKEMON, 2=SETUP_BENCH, 38=DRAW_COUNT, 41=IS_FIRST, 42=MULLIGAN ...
  Skip IS_FIRST(41) and MULLIGAN(42) — trivial/not strategic.

Revealed cards
--------------
For each step, scan observation.current.players[i].{active, hand, bench, discard, prize}
to collect card ids visible to player i (from their own perspective in the replay).
"""

import json
import os
from collections import Counter
from pathlib import Path

# SelectContext values to SKIP (not strategically meaningful for IL)
_SKIP_CONTEXTS = {
    41,  # IS_FIRST: go first or second? (essentially random coin flip choice)
    42,  # MULLIGAN: redraw? (usually just yes)
}

# Human-readable context names for reporting
_CONTEXT_NAMES = {
    0: "MAIN",
    1: "SETUP_ACTIVE",
    2: "SETUP_BENCH",
    3: "SWITCH",
    4: "TO_ACTIVE",
    5: "TO_BENCH",
    6: "TO_FIELD",
    7: "TO_HAND",
    8: "DISCARD",
    9: "TO_DECK",
    10: "TO_DECK_BOTTOM",
    11: "TO_PRIZE",
    12: "NOT_MOVE",
    13: "DAMAGE_COUNTER",
    14: "DAMAGE_COUNTER_ANY",
    15: "DAMAGE",
    16: "REMOVE_DAMAGE_COUNTER",
    17: "HEAL",
    18: "EVOLVES_FROM",
    19: "EVOLVES_TO",
    20: "DEVOLVE",
    21: "ATTACH_FROM",
    22: "ATTACH_TO",
    23: "DETACH_FROM",
    24: "LOOK",
    25: "EFFECT_TARGET",
    26: "DISCARD_ENERGY_CARD",
    27: "DISCARD_TOOL_CARD",
    28: "SWITCH_ENERGY_CARD",
    29: "DISCARD_CARD_OR_ATTACHED_CARD",
    30: "DISCARD_ENERGY",
    31: "TO_HAND_ENERGY",
    32: "TO_DECK_ENERGY",
    33: "SWITCH_ENERGY",
    34: "SKILL_ORDER",
    35: "ATTACK",
    36: "DISABLE_ATTACK",
    37: "EVOLVE",
    38: "DRAW_COUNT",
    39: "DAMAGE_COUNTER_COUNT",
    40: "REMOVE_DAMAGE_COUNTER_COUNT",
    41: "IS_FIRST",
    42: "MULLIGAN",
    43: "ACTIVATE",
    44: "FIRST_EFFECT",
    45: "MORE_DEVOLVE",
    46: "COIN_HEAD",
    47: "AFFECT_SPECIAL_CONDITION",
    48: "RECOVER_SPECIAL_CONDITION",
}


def _context_name(ctx: int) -> str:
    return _CONTEXT_NAMES.get(ctx, f"CTX_{ctx}")


def _extract_card_ids_from_obs(obs: dict) -> list[int]:
    """Collect all card ids visible in a single player observation's current state."""
    cur = obs.get("current")
    if not cur:
        return []
    players = cur.get("players", [])
    my_idx = cur.get("yourIndex", 0)
    ids = []
    for pi, player in enumerate(players):
        # active
        for card in (player.get("active") or []):
            if isinstance(card, dict) and "id" in card:
                ids.append(card["id"])
            # energyCards on active
            for ec in card.get("energyCards", []) if isinstance(card, dict) else []:
                if isinstance(ec, dict) and "id" in ec:
                    ids.append(ec["id"])
        # bench
        for slot in (player.get("bench") or []):
            if isinstance(slot, dict) and "id" in slot:
                ids.append(slot["id"])
            for ec in slot.get("energyCards", []) if isinstance(slot, dict) else []:
                if isinstance(ec, dict) and "id" in ec:
                    ids.append(ec["id"])
        # hand (only for the player whose perspective this obs is)
        if pi == my_idx:
            for card in (player.get("hand") or []):
                if isinstance(card, dict) and "id" in card:
                    ids.append(card["id"])
        # discard
        for card in (player.get("discard") or []):
            if isinstance(card, dict) and "id" in card:
                ids.append(card["id"])
        # prize (visible if taken as prize)
        for card in (player.get("prize") or []):
            if isinstance(card, dict) and "id" in card:
                ids.append(card["id"])
    return ids


def parse_replay(path: str | os.PathLike) -> list[dict]:
    """
    Parse a single replay JSON file into training pairs.

    Returns a list of dicts, one per decision step where select is not None
    and a non-empty action was returned:
        {
          "episodeId": int,
          "player_index": int,
          "player_name": str,
          "obs_dict": dict,          # raw observation dict (includes select, current, etc.)
          "chosen_option_indices": list[int],  # positional indices into select.option
          "select_context": int,     # select.context value
          "n_options": int,          # len(select.option)
          "won": bool,               # did this player win?
        }

    Skips: deck-selection step (select=None), IS_FIRST/MULLIGAN contexts,
    steps where the action is empty or the deck (len=60).
    """
    with open(path) as f:
        data = json.load(f)

    episode_id = data.get("id") or data.get("info", {}).get("EpisodeId", 0)
    agents = data.get("info", {}).get("Agents", [])
    rewards = data.get("rewards", [])
    steps = data.get("steps", [])

    # Determine who won
    # rewards[i] pairs with agents[i]; +1=win, -1=loss
    n_players = len(agents)
    won_by_player = {}
    for i in range(min(n_players, len(rewards))):
        won_by_player[i] = rewards[i] > 0

    pairs = []
    n_steps = len(steps)

    # Training pair: obs = steps[i][player].observation, chosen = steps[i+1][player].action
    for i in range(n_steps - 1):
        for player_idx in range(n_players):
            obs = steps[i][player_idx]["observation"]
            sel = obs.get("select")
            if sel is None:
                continue

            ctx = sel.get("context", -1)
            if ctx in _SKIP_CONTEXTS:
                continue

            opts = sel.get("option", [])
            n_opts = len(opts)
            if n_opts == 0:
                continue

            # The action returned for this obs lives in the NEXT step
            next_step = steps[i + 1]
            if player_idx >= len(next_step):
                continue
            action = next_step[player_idx].get("action", [])

            # Skip empty actions (player was passive / no choice to make yet)
            if not action:
                continue

            # Skip if this looks like a deck submission (60 card IDs)
            if len(action) == 60 and all(isinstance(x, int) for x in action):
                continue

            # Validate indices are in-range
            if not all(isinstance(x, int) and 0 <= x < n_opts for x in action):
                # Some replays may have edge cases; skip invalid
                continue

            player_name = agents[player_idx]["Name"] if player_idx < len(agents) else f"player{player_idx}"
            pairs.append({
                "episodeId": episode_id,
                "player_index": player_idx,
                "player_name": player_name,
                "obs_dict": obs,
                "chosen_option_indices": list(action),
                "select_context": ctx,
                "n_options": n_opts,
                "won": won_by_player.get(player_idx, False),
            })

    return pairs


def revealed_cards(path: str | os.PathLike) -> dict[int, Counter]:
    """
    Returns a dict mapping player_index → Counter of card ids seen in any observation
    across the full game (from that player's perspective).

    This covers: active/bench/energies (both players, always visible),
    hand (own hand only), discard, prize cards taken.
    Opponent's hand and deck cards not visible are excluded.
    """
    with open(path) as f:
        data = json.load(f)

    steps = data.get("steps", [])
    n_players = len(data.get("info", {}).get("Agents", [])) or 2

    revealed: dict[int, Counter] = {i: Counter() for i in range(n_players)}

    for step in steps:
        for player_idx in range(min(n_players, len(step))):
            obs = step[player_idx].get("observation", {})
            ids = _extract_card_ids_from_obs(obs)
            revealed[player_idx].update(ids)

    return revealed


def _opp_card_id_serial_pairs(obs: dict, opp_idx: int) -> list[tuple[int, int]]:
    """(id, serial) pairs for every card belonging to the OPPONENT that is visible
    in this single observation: active/bench Pokemon + their energyCards/tools,
    discard pile, and stadium (if played by the opponent).

    Unlike _extract_card_ids_from_obs, this also reads "tools" and "stadium" (not
    covered there) and returns (id, serial) instead of a flat id list, so callers
    can dedup by physical card instance rather than by step-appearance.
    """
    cur = obs.get("current")
    if not cur:
        return []
    players = cur.get("players", [])
    if opp_idx >= len(players):
        return []
    opp = players[opp_idx]
    pairs = []
    for slot in (opp.get("active") or []) + (opp.get("bench") or []):
        if not isinstance(slot, dict):
            continue
        if "id" in slot and "serial" in slot:
            pairs.append((slot["id"], slot["serial"]))
        for zone in ("energyCards", "tools"):
            for c in (slot.get(zone) or []):
                if isinstance(c, dict) and "id" in c and "serial" in c:
                    pairs.append((c["id"], c["serial"]))
    for c in (opp.get("discard") or []):
        if isinstance(c, dict) and "id" in c and "serial" in c:
            pairs.append((c["id"], c["serial"]))
    for c in (cur.get("stadium") or []):
        if isinstance(c, dict) and c.get("playerIndex") == opp_idx and "id" in c and "serial" in c:
            pairs.append((c["id"], c["serial"]))
    return pairs


def opponent_reveal_progression(
    path: str | os.PathLike,
    milestones: tuple[int, ...] = (1, 2, 4, 8, 12, 20),
) -> list[dict]:
    """Load `path` and delegate to opponent_reveal_progression_from_data (see there
    for the full contract). Kept as the convenient single-file entry point."""
    with open(path) as f:
        data = json.load(f)
    return opponent_reveal_progression_from_data(data, milestones)


def true_decks_from_data(data: dict) -> dict[int, list[int]]:
    """
    Extract the submitted 60-card deck for each player from an already-loaded
    replay dict. The deck is the 60-card action in steps[1][player].action —
    same ground truth used by src/eval/topdeck_recon.py::extract_deck_from_replay
    and src/eval/crustle_list_recon.py::extract_deck_from_replay (kept here too,
    operating on pre-loaded data to avoid re-reading multi-MB replay files a
    second time when a caller also needs opponent_reveal_progression_from_data).
    Returns {player_idx: [card_id, ...]} (list of 60 ids), only for players
    whose steps[1] action is actually a 60-card list.
    """
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


def opponent_reveal_progression_from_data(
    data: dict,
    milestones: tuple[int, ...] = (1, 2, 4, 8, 12, 20),
) -> list[dict]:
    """
    For each player's perspective, walk the game in order and snapshot the
    OPPONENT's revealed card counts at each milestone (total revealed *physical
    copies*, deduped by (card_id, serial) — NOT step-appearances, so a Pokemon
    sitting on the bench for 50 steps still counts once).

    Returns a list of records, one per (player_index, milestone) actually reached
    in this game: {"player_index": int, "milestone": int, "revealed": {card_id: count}}.
    A game that reveals fewer than milestones[0] opponent cards yields no records
    for that player.

    Takes an already-parsed replay dict (see opponent_reveal_progression for the
    load-from-path convenience wrapper) so callers needing both this and
    true_decks_from_data only pay the json.load() cost once per file.
    """
    steps = data.get("steps", [])
    n_players = len(data.get("info", {}).get("Agents", [])) or 2

    records = []
    for player_idx in range(n_players):
        opp_idx = 1 - player_idx
        seen_serials: set[tuple[int, int]] = set()
        counts: Counter = Counter()
        next_milestone_i = 0
        for step in steps:
            if next_milestone_i >= len(milestones):
                break
            if player_idx >= len(step):
                continue
            obs = step[player_idx].get("observation", {})
            for cid, serial in _opp_card_id_serial_pairs(obs, opp_idx):
                key = (cid, serial)
                if key in seen_serials:
                    continue
                seen_serials.add(key)
                counts[cid] += 1
            total = sum(counts.values())
            while next_milestone_i < len(milestones) and total >= milestones[next_milestone_i]:
                records.append({
                    "player_index": player_idx,
                    "milestone": milestones[next_milestone_i],
                    "revealed": dict(counts),
                })
                next_milestone_i += 1
    return records


def walk_replays(dirs: list[str | os.PathLike]) -> list[Path]:
    """Return all *-replay.json files found recursively under the given directories."""
    paths = []
    for d in dirs:
        d = Path(d)
        if d.is_file() and d.name.endswith("-replay.json"):
            paths.append(d)
        elif d.is_dir():
            for p in sorted(d.rglob("*-replay.json")):
                paths.append(p)
    return paths


def aggregate_stats(replay_paths: list[Path]) -> dict:
    """
    Parse all replays and return aggregate statistics dict.
    """
    total_games = 0
    total_pairs = 0
    context_counts: Counter = Counter()
    winner_pairs = 0
    loser_pairs = 0
    total_options = 0
    avg_revealed_per_game: list[float] = []
    parse_errors = 0

    for path in replay_paths:
        try:
            pairs = parse_replay(path)
            rev = revealed_cards(path)

            total_games += 1
            total_pairs += len(pairs)
            winner_pairs += sum(1 for p in pairs if p["won"])
            loser_pairs += sum(1 for p in pairs if not p["won"])
            for p in pairs:
                context_counts[p["select_context"]] += 1
                total_options += p["n_options"]

            # avg distinct cards per player perspective
            for player_counter in rev.values():
                avg_revealed_per_game.append(len(player_counter))

        except Exception as e:
            parse_errors += 1
            print(f"  WARN: failed to parse {path.name}: {e}")

    avg_opts = total_options / total_pairs if total_pairs else 0
    avg_rev = sum(avg_revealed_per_game) / len(avg_revealed_per_game) if avg_revealed_per_game else 0

    return {
        "total_games": total_games,
        "total_pairs": total_pairs,
        "winner_pairs": winner_pairs,
        "loser_pairs": loser_pairs,
        "avg_options_per_decision": avg_opts,
        "avg_distinct_revealed_per_player_per_game": avg_rev,
        "context_counts": context_counts,
        "parse_errors": parse_errors,
    }


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Parse Pokemon TCG replay JSONs for IL")
    parser.add_argument(
        "dirs",
        nargs="*",
        default=["data/episodes/top_players", "data/episodes"],
        help="Directories (or files) to search for *-replay.json files",
    )
    parser.add_argument("--top-n-contexts", type=int, default=15, help="How many contexts to show")
    args = parser.parse_args()

    # Resolve paths relative to repo root (two levels up from this file)
    repo_root = Path(__file__).parent.parent.parent
    dirs = []
    for d in args.dirs:
        p = Path(d)
        if not p.is_absolute():
            p = repo_root / p
        dirs.append(p)

    paths = walk_replays(dirs)
    # Deduplicate by name (same file may appear from multiple dir searches)
    seen_names = set()
    unique_paths = []
    for p in paths:
        if p.name not in seen_names:
            seen_names.add(p.name)
            unique_paths.append(p)
    paths = unique_paths

    print(f"Found {len(paths)} replay files across {[str(d) for d in dirs]}")

    if not paths:
        print("No replays found.")
        return

    print("Parsing...")
    stats = aggregate_stats(paths)

    print("\n=== AGGREGATE STATS ===")
    print(f"  Total games parsed:              {stats['total_games']}")
    print(f"  Parse errors:                    {stats['parse_errors']}")
    print(f"  Total (obs, action) pairs:       {stats['total_pairs']}")
    print(f"  Pairs from winners:              {stats['winner_pairs']}")
    print(f"  Pairs from losers:               {stats['loser_pairs']}")
    winner_frac = stats['winner_pairs'] / stats['total_pairs'] if stats['total_pairs'] else 0
    print(f"  Winner fraction:                 {winner_frac:.1%}")
    print(f"  Avg options/decision:            {stats['avg_options_per_decision']:.2f}")
    print(f"  Avg distinct revealed cards/player/game: {stats['avg_distinct_revealed_per_player_per_game']:.1f}")

    print(f"\n  Pairs by SelectContext (top {args.top_n_contexts}):")
    for ctx, count in stats["context_counts"].most_common(args.top_n_contexts):
        name = _context_name(ctx)
        pct = 100 * count / stats['total_pairs'] if stats['total_pairs'] else 0
        print(f"    {ctx:3d} ({name:30s}): {count:6d}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
