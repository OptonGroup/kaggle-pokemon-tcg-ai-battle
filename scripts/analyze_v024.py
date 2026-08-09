#!/usr/bin/env python3
"""Analyze v024 (card-embedding transformer + RL) live replays.

Parses data/episodes/episode-<id>-replay.json for the 39 v024 games,
determines W/L/D, opponent archetype, our board development, and
behavioral signals (declined attacks, retreat loops, hand clog, etc).

Outputs:
  data/episodes/v024_analysis.json  (list of per-game dicts)
  Also prints an aggregate summary to stdout.
"""
import csv
import json
import glob
import os
from collections import Counter, defaultdict

ROOT = "/Users/max/kaggle-pokemon-tcg-ai-battle"
EP_LIST = os.environ.get("V024_EP_LIST", "/tmp/tmp/v024_episodes.txt")
EPISODES_DIR = os.path.join(ROOT, "data/episodes")
OUT_JSON = os.path.join(ROOT, "data/episodes/v024_analysis.json")

POKEMON_STAGES = {"Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"}

# ---- Load card data ----
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


# Known archetype signature Pokemon (name substring, case-insensitive) -> archetype label
ARCHETYPE_SIGNATURES = [
    ("starmie", "Starmie"),
    ("mega lucario", "Mega Lucario ex"),
    ("lucario", "Lucario"),
    ("alakazam", "Alakazam"),
    ("archaludon", "Archaludon"),
    ("duraludon", "Archaludon"),
    ("grimmsnarl", "Grimmsnarl ex"),
    ("munkidori", "Grimmsnarl ex/Munkidori"),
    ("crustle", "Crustle/Dwebble (mirror)"),
    ("dwebble", "Crustle/Dwebble (mirror)"),
    ("cubchoo", "Cubchoo/Beartic"),
    ("beartic", "Cubchoo/Beartic"),
    ("iron hands", "Iron Hands/Electric"),
    ("raging bolt", "Raging Bolt/Electric"),
    ("pikachu", "Pikachu/Electric"),
    ("electrode", "Electrode/Electric"),
    ("festival lead", "Festival Lead"),
    ("gholdengo", "Gholdengo"),
    ("dragapult", "Dragapult ex"),
    ("charizard", "Charizard ex"),
    ("miraidon", "Miraidon ex"),
    ("iron valiant", "Iron Valiant"),
    ("gardevoir", "Gardevoir ex"),
    ("chien-pao", "Chien-Pao ex"),
    ("roaring moon", "Roaring Moon ex"),
]


def find_agents_and_result(data):
    agents = data["info"]["Agents"]
    our_index = None
    for i, a in enumerate(agents):
        if a.get("Name") == "Maxim Simakov":
            our_index = i
            break
    if our_index is None:
        # fallback: assume 0
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
    our_final_active_ids = set()
    our_bench_ever_nonzero = False
    our_max_bench = 0
    our_handcount_series = []  # (turn, handCount, benchCount)
    statuses_seen = set()

    # For behavior: collect our MAIN-level decisions in order, grouped by turn
    # our_decisions: list of dict(turn, option_types, chosen_types)
    our_decisions = []
    our_active_serial_by_turn = {}  # turn -> serial of our active (first snapshot seen that turn)

    for step in steps:
        for view in step:
            obs = view.get("observation", {})
            statuses_seen.add(view.get("status"))
            cur = obs.get("current")
            if cur:
                t = cur.get("turn", 0)
                if t and t > max_turn:
                    max_turn = t
                fp = cur.get("firstPlayer")
                if fp is not None and fp != -1:
                    first_player = fp
                players = cur.get("players", [])
                if len(players) == 2:
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
                    our_active_ids_now = {c.get("id") for c in (our_p.get("active") or []) if c}
                    our_final_active_ids = our_active_ids_now
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

            # decision-level: this view acted this step if action nonempty
            action = view.get("action")
            sel = obs.get("select")
            if action and sel and sel.get("type") == 0:  # MAIN
                # is this our decision? need to know which player made it.
                # cur may be None for this view at this exact step (rare); use cur if present else infer via other view
                pass

    # Second pass dedicated to decisions: need per-step both views to know whose decision,
    # since the deciding view is the one with nonempty action AND its own perspective (yourIndex in its cur).
    for step in steps:
        for view in step:
            action = view.get("action")
            obs = view.get("observation", {})
            sel = obs.get("select")
            if not action or not sel:
                continue
            cur = obs.get("current")
            your_index = cur.get("yourIndex") if cur else None
            if your_index is None:
                continue
            if your_index != our_index:
                continue
            if sel.get("type") != 0:
                continue
            options = sel.get("option") or []
            option_types = [o.get("type") for o in options]
            chosen_types = []
            for idx in action:
                if 0 <= idx < len(option_types):
                    chosen_types.append(option_types[idx])
            turn = cur.get("turn", 0)
            our_decisions.append({
                "turn": turn,
                "option_types": option_types,
                "chosen_types": chosen_types,
            })

    # Aggregate behavior counters
    cnt_attack = sum(1 for d in our_decisions if 13 in d["chosen_types"])
    cnt_retreat = sum(1 for d in our_decisions if 12 in d["chosen_types"])
    cnt_end = sum(1 for d in our_decisions if 14 in d["chosen_types"])
    cnt_attach = sum(1 for d in our_decisions if 8 in d["chosen_types"])
    cnt_evolve = sum(1 for d in our_decisions if 9 in d["chosen_types"])
    cnt_play = sum(1 for d in our_decisions if 7 in d["chosen_types"])
    cnt_ability = sum(1 for d in our_decisions if 10 in d["chosen_types"])

    # Declined-attack turns: group decisions by turn
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

    # retreat-loop: oscillation A,B,A,B in active serial across consecutive turns
    turns_sorted = sorted(our_active_serial_by_turn.keys())
    serial_seq = [our_active_serial_by_turn[t] for t in turns_sorted]
    retreat_loop = False
    for i in range(len(serial_seq) - 3):
        a, b, c, d4 = serial_seq[i], serial_seq[i+1], serial_seq[i+2], serial_seq[i+3]
        if a == c and b == d4 and a != b:
            retreat_loop = True
            break

    # hand clog: bench stays 0 while handCount grows beyond 8 in mid/late game
    hand_clog = False
    max_handcount_seen = 0
    for (t, hc, bench_n) in our_handcount_series:
        if hc is not None:
            max_handcount_seen = max(max_handcount_seen, hc)
        if t >= 5 and bench_n == 0 and (hc or 0) >= 8:
            hand_clog = True

    evolved_to_crustle = 345 in our_pokemon_ids
    stuck_on_dwebble_only = (344 in our_pokemon_ids) and (345 not in our_pokemon_ids)

    # opponent archetype detection
    opp_names = sorted({name_of(cid) for cid in opp_pokemon_ids})
    opp_label = None
    lower_names = [n.lower() for n in opp_names]
    for sig, label in ARCHETYPE_SIGNATURES:
        if any(sig in n for n in lower_names):
            opp_label = label
            break
    if opp_label is None:
        opp_label = "Unknown (" + (", ".join(opp_names[:3]) if opp_names else "no pokemon seen") + ")"

    our_names = sorted({name_of(cid) for cid in our_pokemon_ids})

    reason_note = []
    if outcome == "L":
        if not our_bench_ever_nonzero:
            reason_note.append("NEVER benched a second Pokemon (single-attacker whole game)")
        if hand_clog:
            reason_note.append(f"hand clog (up to {max_handcount_seen} cards, bench empty)")
        if stuck_on_dwebble_only:
            reason_note.append("never evolved Dwebble->Crustle")
        if declined_attack_turns:
            reason_note.append(f"declined available attack on turns {declined_attack_turns}")
        if retreat_loop:
            reason_note.append("retreat oscillation loop detected")
        if not attack_ever_available:
            reason_note.append("attack option never became available all game")
    else:
        if cnt_attack > 0:
            reason_note.append(f"attacked {cnt_attack}x")
        if evolved_to_crustle:
            reason_note.append("evolved to Crustle")

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
        "our_evolved_to_crustle": evolved_to_crustle,
        "our_stuck_on_dwebble_only": stuck_on_dwebble_only,
        "our_bench_ever_nonzero": our_bench_ever_nonzero,
        "our_max_bench": our_max_bench,
        "our_max_handcount": max_handcount_seen,
        "hand_clog_detected": hand_clog,
        "cnt_attack": cnt_attack,
        "cnt_retreat": cnt_retreat,
        "cnt_end": cnt_end,
        "cnt_attach": cnt_attach,
        "cnt_evolve": cnt_evolve,
        "cnt_play": cnt_play,
        "cnt_ability": cnt_ability,
        "declined_attack_turns": declined_attack_turns,
        "attack_ever_available": attack_ever_available,
        "retreat_loop_detected": retreat_loop,
        "statuses_seen": sorted(s for s in statuses_seen if s),
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

    # ---- Aggregate summary ----
    n = len(results)
    w = sum(1 for r in results if r["outcome"] == "W")
    l = sum(1 for r in results if r["outcome"] == "L")
    d = sum(1 for r in results if r["outcome"] == "D")
    print(f"\n=== TOTAL: {n} games, W-L-D = {w}-{l}-{d}, WR={w/n:.1%} ===\n")

    by_arch = defaultdict(lambda: {"W": 0, "L": 0, "D": 0})
    for r in results:
        by_arch[r["opp_archetype"]][r["outcome"]] += 1

    print("Archetype | Games | W | L | D | WR")
    for arch, c in sorted(by_arch.items(), key=lambda kv: -(kv[1]["W"]+kv[1]["L"]+kv[1]["D"])):
        g = c["W"] + c["L"] + c["D"]
        wr = c["W"] / g if g else 0
        print(f"{arch} | {g} | {c['W']} | {c['L']} | {c['D']} | {wr:.0%}")

    print("\n--- Deck check ---")
    always_crustle = all(set(r["our_pokemon_names"]) <= {"Dwebble", "Crustle"} for r in results)
    print("All games use only Dwebble/Crustle as our Pokemon:", always_crustle)
    for r in results:
        if not (set(r["our_pokemon_names"]) <= {"Dwebble", "Crustle"}):
            print("  DEVIATION", r["episode_id"], r["our_pokemon_names"])

    print("\n--- Behavior flags ---")
    n_bench_zero = sum(1 for r in results if not r["our_bench_ever_nonzero"])
    n_hand_clog = sum(1 for r in results if r["hand_clog_detected"])
    n_stuck_dwebble = sum(1 for r in results if r["our_stuck_on_dwebble_only"])
    n_declined = sum(1 for r in results if r["declined_attack_turns"])
    n_retreat_loop = sum(1 for r in results if r["retreat_loop_detected"])
    n_never_attack_avail = sum(1 for r in results if not r["attack_ever_available"])
    print("never benched second pokemon:", n_bench_zero, [r["episode_id"] for r in results if not r["our_bench_ever_nonzero"]])
    print("hand clog detected:", n_hand_clog, [r["episode_id"] for r in results if r["hand_clog_detected"]])
    print("stuck on dwebble only (never evolved):", n_stuck_dwebble, [r["episode_id"] for r in results if r["our_stuck_on_dwebble_only"]])
    print("declined available attack in some turn:", n_declined, [(r["episode_id"], r["declined_attack_turns"]) for r in results if r["declined_attack_turns"]])
    print("retreat loop detected:", n_retreat_loop, [r["episode_id"] for r in results if r["retreat_loop_detected"]])
    print("attack option never available all game:", n_never_attack_avail, [r["episode_id"] for r in results if not r["attack_ever_available"]])

    print("\n--- Statuses (non ACTIVE/INACTIVE/DONE = suspicious) ---")
    for r in results:
        extra = [s for s in r["statuses_seen"] if s not in ("ACTIVE", "INACTIVE", "DONE")]
        if extra:
            print(r["episode_id"], extra)

    print("\n--- Per game one-liners ---")
    for r in results:
        print(f"{r['episode_id']} | {r['outcome']} | turns={r['num_turns_max']} | 1st={r['we_went_first']} | opp={r['opp_archetype']} | atk={r['cnt_attack']} ret={r['cnt_retreat']} bench_max={r['our_max_bench']} hand_max={r['our_max_handcount']} | notes={'; '.join(r['reason_note'])}")


if __name__ == "__main__":
    main()
