#!/usr/bin/env python3
"""Forensic analyzer for v036_archaludon_patched live replays.

Parses data/episodes/episode-<id>-replay.json for the games listed in
data/episodes/v036_episode_ids.txt, determines W/L/D, tags the opponent
archetype USING THE SAME ID SETS v036's own main.py::detect_matchup() uses
(so the report reflects what the pilot itself would have classified the
matchup as), and flags concrete behavioral anomalies:

  - declined available attacks, retreat oscillation, hand clog (generic,
    reused pattern from analyze_v024.py)
  - Boss's Orders (1182) / Jumbo Ice Cream (1147) available in hand as a PLAY
    option but never chosen the whole game
  - Duraludon (169) / Archaludon ex (190) never attacked despite the deck's
    entire plan being built around one of them attacking

Caveat: archetype tagging here is a POST-HOC approximation over the whole
game (union of opponent Pokemon ever seen), not the pilot's live per-turn
detect_matchup() call (which only sees the board at decision time and can
theoretically flip earlier in the game before all archetype pieces are
revealed). Good enough to find WR-by-archetype patterns and blind spots.

Outputs:
  data/episodes/v036_analysis.json  (list of per-game dicts)
  Also prints an aggregate summary to stdout.
"""
import csv
import json
import os
from collections import Counter, defaultdict

ROOT = "/Users/max/kaggle-pokemon-tcg-ai-battle"
EP_LIST = os.path.join(ROOT, "data/episodes/v036_episode_ids.txt")
EPISODES_DIR = os.path.join(ROOT, "data/episodes")
OUT_JSON = os.path.join(ROOT, "data/episodes/v036_analysis.json")

# ── Mirrors submissions/v036_archaludon_patched/main.py card IDs / archetype
#    ID sets, so archetype tagging matches what the PILOT itself would see. ──
DURALUDON = 169
ARCHALUDON_EX = 190
CINDERACE = 666
RELICANTH = 57

CRUSTLE_LINE = {344, 345, 532}
STARMIE_LINE = {1030, 1031}
LUCARIO_LINE = {677, 678}
HOP_LINE = {288, 289, 299, 304, 307, 308, 309, 310, 878, 879}
GRIMMSNARL_LINE = {646, 647, 648}
DIPPLIN_LINE = {89, 90, 92, 93, 247}
ALAKAZAM_LINE = {741, 742, 743}

BOSS = 1182
JUMBO_ICE_CREAM = 1147

# detect_matchup() priority order in main.py (first match wins)
_ARCHETYPE_ORDER = [
    (CRUSTLE_LINE, "crustle"),
    (HOP_LINE, "hop"),
    (STARMIE_LINE, "starmie"),
    (LUCARIO_LINE, "lucario"),
    (ALAKAZAM_LINE, "alakazam"),
    (GRIMMSNARL_LINE, "grimmsnarl"),
    (DIPPLIN_LINE, "dipplin"),
]

POKEMON_STAGES = {"Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"}

# ---- Load card data (for human-readable names in the report) ----
card_by_id = {}
with open(os.path.join(ROOT, "data/EN_Card_Data.csv")) as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            cid = int(row["Card ID"])
        except ValueError:
            continue
        card_by_id[cid] = {
            "name": row["Card Name"],
            "stage": row["Stage (Pokémon)/Type (Energy and Trainer)"],
        }


def name_of(cid):
    c = card_by_id.get(cid)
    return c["name"] if c else f"UNKNOWN({cid})"


def is_pokemon(cid):
    c = card_by_id.get(cid)
    return bool(c and c["stage"] in POKEMON_STAGES)


def detect_matchup(opp_pokemon_ids):
    for line, label in _ARCHETYPE_ORDER:
        if opp_pokemon_ids & line:
            return label
    return "generic"


def find_agents_and_result(data):
    agents = data["info"]["Agents"]
    our_index = None
    for i, a in enumerate(agents):
        if a.get("Name") == "Maxim Simakov":
            our_index = i
            break
    if our_index is None:
        our_index = 0
    opp_index = 1 - our_index
    opp_name = agents[opp_index].get("Name")
    r = data["rewards"]
    our_r = r[our_index]
    if our_r > 0:
        outcome = "W"
    elif our_r < 0:
        outcome = "L"
    else:
        outcome = "D"
    return our_index, opp_index, opp_name, outcome


def analyze_episode(path):
    with open(path) as f:
        data = json.load(f)
    ep_id = data.get("info", {}).get("EpisodeId") or data.get("id") or os.path.basename(path)
    our_index, opp_index, opp_name, outcome = find_agents_and_result(data)
    steps = data["steps"]

    max_turn = 0
    first_player = None
    opp_pokemon_ids = set()
    our_pokemon_ids = set()
    our_bench_ever_nonzero = False
    our_max_bench = 0
    our_handcount_series = []
    our_active_serial_by_turn = {}

    for step in steps:
        for view in step:
            obs = view.get("observation", {})
            cur = obs.get("current")
            if not cur:
                continue
            t = cur.get("turn", 0)
            if t and t > max_turn:
                max_turn = t
            fp = cur.get("firstPlayer")
            if fp is not None and fp != -1:
                first_player = fp
            players = cur.get("players", [])
            if len(players) != 2:
                continue
            opp_p = players[opp_index]
            our_p = players[our_index]
            for zone in ("active", "bench", "discard"):
                for card in opp_p.get(zone) or []:
                    if card is None:
                        continue
                    cid = card.get("id")
                    if cid is not None and is_pokemon(cid):
                        opp_pokemon_ids.add(cid)
            for zone in ("active", "bench", "discard"):
                for card in our_p.get(zone) or []:
                    if card is None:
                        continue
                    cid = card.get("id")
                    if cid is not None and is_pokemon(cid):
                        our_pokemon_ids.add(cid)
            bench_n = len([c for c in (our_p.get("bench") or []) if c])
            our_max_bench = max(our_max_bench, bench_n)
            if bench_n > 0:
                our_bench_ever_nonzero = True
            hc = our_p.get("handCount")
            if hc is not None:
                our_handcount_series.append((t, hc, bench_n))
            our_active_serials = [c.get("serial") for c in (our_p.get("active") or []) if c]
            if our_active_serials:
                our_active_serial_by_turn.setdefault(t, our_active_serials[0])

    # Decision-level pass: our MAIN-context decisions, plus PLAY-option card ids
    # (both available and chosen) to detect Boss/Ice-Cream availability vs use.
    our_decisions = []
    boss_available_turns = []
    boss_played_turns = []
    ice_cream_available_turns = []
    ice_cream_played_turns = []

    for step in steps:
        for view in step:
            action = view.get("action")
            obs = view.get("observation", {})
            sel = obs.get("select")
            if not action or not sel:
                continue
            cur = obs.get("current")
            your_index = cur.get("yourIndex") if cur else None
            if your_index is None or your_index != our_index:
                continue
            if sel.get("type") != 0:  # MAIN
                continue
            options = sel.get("option") or []
            hand = cur["players"][your_index].get("hand") or []
            turn = cur.get("turn", 0)

            option_types = [o.get("type") for o in options]
            chosen_types = []
            for idx in action:
                if 0 <= idx < len(option_types):
                    chosen_types.append(option_types[idx])
            our_decisions.append({
                "turn": turn,
                "option_types": option_types,
                "chosen_types": chosen_types,
            })

            # PLAY options (type 7): option['index'] indexes into hand[]
            chosen_set = set(action)
            for oi, o in enumerate(options):
                if o.get("type") != 7:
                    continue
                hidx = o.get("index")
                if hidx is None or hidx >= len(hand) or hand[hidx] is None:
                    continue
                cid = hand[hidx].get("id")
                if cid == BOSS:
                    boss_available_turns.append(turn)
                    if oi in chosen_set:
                        boss_played_turns.append(turn)
                elif cid == JUMBO_ICE_CREAM:
                    ice_cream_available_turns.append(turn)
                    if oi in chosen_set:
                        ice_cream_played_turns.append(turn)

    cnt_attack = sum(1 for d in our_decisions if 13 in d["chosen_types"])
    cnt_retreat = sum(1 for d in our_decisions if 12 in d["chosen_types"])

    by_turn = defaultdict(list)
    for d in our_decisions:
        by_turn[d["turn"]].append(d)
    declined_attack_turns = []
    for turn, ds in by_turn.items():
        attack_available = any(13 in d["option_types"] for d in ds)
        attack_chosen = any(13 in d["chosen_types"] for d in ds)
        ended = any(14 in d["chosen_types"] for d in ds)
        if attack_available and not attack_chosen and ended:
            declined_attack_turns.append(turn)

    never_attacked_whole_game = (cnt_attack == 0)
    attack_ever_available = any(13 in d["option_types"] for d in our_decisions)

    turns_sorted = sorted(our_active_serial_by_turn.keys())
    serial_seq = [our_active_serial_by_turn[t] for t in turns_sorted]
    retreat_loop = False
    for i in range(len(serial_seq) - 3):
        a, b, c, d4 = serial_seq[i], serial_seq[i + 1], serial_seq[i + 2], serial_seq[i + 3]
        if a == c and b == d4 and a != b:
            retreat_loop = True
            break

    hand_clog = False
    max_handcount_seen = 0
    for (t, hc, bench_n) in our_handcount_series:
        if hc is not None:
            max_handcount_seen = max(max_handcount_seen, hc)
        if t >= 5 and bench_n == 0 and (hc or 0) >= 8:
            hand_clog = True

    boss_available_never_played = bool(boss_available_turns) and not boss_played_turns
    ice_cream_available_never_played = bool(ice_cream_available_turns) and not ice_cream_played_turns
    attacker_never_attacked = (
        (DURALUDON in our_pokemon_ids or ARCHALUDON_EX in our_pokemon_ids)
        and not any(13 in d["chosen_types"] for d in our_decisions)
    )

    opp_label = detect_matchup(opp_pokemon_ids)
    opp_names = sorted({name_of(cid) for cid in opp_pokemon_ids})
    our_names = sorted({name_of(cid) for cid in our_pokemon_ids})

    reason_note = []
    if outcome == "L":
        if not our_bench_ever_nonzero:
            reason_note.append("NEVER benched a second Pokemon (single-attacker whole game)")
        if hand_clog:
            reason_note.append(f"hand clog (up to {max_handcount_seen} cards, bench empty)")
        if declined_attack_turns:
            reason_note.append(f"declined available attack on turns {declined_attack_turns}")
        if retreat_loop:
            reason_note.append("retreat oscillation loop detected")
        if not attack_ever_available:
            reason_note.append("attack option never became available all game")
        if boss_available_never_played:
            reason_note.append(f"Boss's Orders in hand (turns {boss_available_turns}) but NEVER played")
        if attacker_never_attacked:
            reason_note.append("Duraludon/Archaludon ex on board but never attacked (energy misrouting?)")
        if opp_label == "generic":
            reason_note.append(f"opponent archetype UNRECOGNIZED (generic fallback) — opp pokemon: {opp_names}")
    else:
        if cnt_attack > 0:
            reason_note.append(f"attacked {cnt_attack}x")

    return {
        "episode_id": ep_id,
        "path": os.path.basename(path),
        "our_index": our_index,
        "opp_name": opp_name,
        "outcome": outcome,
        "num_turns_max": max_turn,
        "first_player": first_player,
        "we_went_first": (first_player == our_index) if first_player is not None else None,
        "opp_archetype": opp_label,
        "opp_pokemon_names": opp_names,
        "our_pokemon_names": our_names,
        "our_bench_ever_nonzero": our_bench_ever_nonzero,
        "our_max_bench": our_max_bench,
        "our_max_handcount": max_handcount_seen,
        "hand_clog_detected": hand_clog,
        "cnt_attack": cnt_attack,
        "cnt_retreat": cnt_retreat,
        "declined_attack_turns": declined_attack_turns,
        "attack_ever_available": attack_ever_available,
        "retreat_loop_detected": retreat_loop,
        "boss_available_turns": boss_available_turns,
        "boss_played_turns": boss_played_turns,
        "boss_available_never_played": boss_available_never_played,
        "ice_cream_available_turns": ice_cream_available_turns,
        "ice_cream_played_turns": ice_cream_played_turns,
        "ice_cream_available_never_played": ice_cream_available_never_played,
        "attacker_never_attacked": attacker_never_attacked,
        "reason_note": reason_note,
    }


def main():
    with open(EP_LIST) as f:
        ep_ids = [line.strip() for line in f if line.strip()]

    results = []
    for eid in ep_ids:
        path = os.path.join(EPISODES_DIR, f"episode-{eid}-replay.json")
        if not os.path.exists(path):
            print("MISSING FILE:", path)
            continue
        try:
            res = analyze_episode(path)
        except Exception as e:
            print("ERROR analyzing", eid, e)
            import traceback
            traceback.print_exc()
            continue
        results.append(res)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n = len(results)
    w = sum(1 for r in results if r["outcome"] == "W")
    l = sum(1 for r in results if r["outcome"] == "L")
    d = sum(1 for r in results if r["outcome"] == "D")
    print(f"\n=== TOTAL: {n} games, W-L-D = {w}-{l}-{d}, WR={w/n:.1%} ===\n")

    by_arch = defaultdict(lambda: {"W": 0, "L": 0, "D": 0})
    for r in results:
        by_arch[r["opp_archetype"]][r["outcome"]] += 1

    print("Archetype (by v036's own detect_matchup ID sets) | Games | W | L | D | WR")
    for arch, c in sorted(by_arch.items(), key=lambda kv: -(kv[1]["W"] + kv[1]["L"] + kv[1]["D"])):
        g = c["W"] + c["L"] + c["D"]
        wr = c["W"] / g if g else 0
        print(f"{arch} | {g} | {c['W']} | {c['L']} | {c['D']} | {wr:.0%}")

    print("\n--- Behavior flags (all games) ---")
    n_bench_zero = sum(1 for r in results if not r["our_bench_ever_nonzero"])
    n_hand_clog = sum(1 for r in results if r["hand_clog_detected"])
    n_declined = sum(1 for r in results if r["declined_attack_turns"])
    n_retreat_loop = sum(1 for r in results if r["retreat_loop_detected"])
    n_never_attack_avail = sum(1 for r in results if not r["attack_ever_available"])
    n_boss_wasted = sum(1 for r in results if r["boss_available_never_played"])
    n_attacker_silent = sum(1 for r in results if r["attacker_never_attacked"])
    n_generic = sum(1 for r in results if r["opp_archetype"] == "generic")
    print("never benched second pokemon:", n_bench_zero, [r["episode_id"] for r in results if not r["our_bench_ever_nonzero"]])
    print("hand clog detected:", n_hand_clog, [r["episode_id"] for r in results if r["hand_clog_detected"]])
    print("declined available attack in some turn:", n_declined, [(r["episode_id"], r["declined_attack_turns"]) for r in results if r["declined_attack_turns"]])
    print("retreat loop detected:", n_retreat_loop, [r["episode_id"] for r in results if r["retreat_loop_detected"]])
    print("attack option never available all game:", n_never_attack_avail, [r["episode_id"] for r in results if not r["attack_ever_available"]])
    print("Boss's Orders in hand but never played:", n_boss_wasted, [(r["episode_id"], r["outcome"]) for r in results if r["boss_available_never_played"]])
    print("Duraludon/Archaludon on board but never attacked:", n_attacker_silent, [(r["episode_id"], r["outcome"]) for r in results if r["attacker_never_attacked"]])
    print("opponent archetype = generic (unrecognized):", n_generic, [(r["episode_id"], r["outcome"], r["opp_pokemon_names"]) for r in results if r["opp_archetype"] == "generic"])

    print("\n--- Losses only: reason notes ---")
    for r in results:
        if r["outcome"] == "L":
            print(f"{r['episode_id']} | opp={r['opp_archetype']} ({r['opp_name']}) | turns={r['num_turns_max']} | 1st={r['we_went_first']} | notes={'; '.join(r['reason_note']) or '(no flag triggered)'}")

    print("\n--- Per game one-liners ---")
    for r in results:
        print(f"{r['episode_id']} | {r['outcome']} | turns={r['num_turns_max']} | 1st={r['we_went_first']} | opp={r['opp_archetype']}/{r['opp_name']} | atk={r['cnt_attack']} ret={r['cnt_retreat']} bench_max={r['our_max_bench']} hand_max={r['our_max_handcount']}")


if __name__ == "__main__":
    main()
