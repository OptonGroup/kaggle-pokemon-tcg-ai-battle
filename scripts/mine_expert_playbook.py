#!/usr/bin/env python3
"""mine_expert_playbook.py — per-matchup expert playbook mining for the
Marnie's Grimmsnarl ex deck (jiatu.l / bono) from raw official daily episode
dumps (data/episodes/official/{day}/*.json).

Unlike scripts/scan_daily_dataset_for_experts.py (which mines (obs, action)
pairs from the per-player fog-of-war `steps[t][p]['observation']` stream for
BC training), this script uses the OMNISCIENT replay log stored once per file
at `steps[0][0]['visualize']` — a chronological list of "frames", each with:

    {
      "current": <full board state, BOTH players fully revealed>,
      "select":  {"context": "<StringName>", "minCount", "maxCount", "option": [...]},
      "selected": [chosen option indices],
      "logs":    [events resolved by the PREVIOUS frame's `selected`],
    }

`logs[i]['type']` vocabulary observed: MoveCard, Draw, Play, Shuffle, HpChange,
TurnStart, TurnEnd, Evolve, Attach, Attack, HasBasicPokemon, Switch, Coin,
Result, Confused, MoveAttached, Asleep.

Key empirically-verified event field semantics (see docs/gold/v062_expert_playbook_lines.md
for the recon that established these — do not re-derive from scratch):

  - Attach:  {cardId=energy id, cardIdTarget=creature id, playerIndex, serial, serialTarget}
  - Evolve:  {cardId=NEW stage id, cardIdTarget=FROM stage id, playerIndex, serial, serialTarget}
             Rare Candy shows as cardId==648 (Grimmsnarl ex) with cardIdTarget==646 (Impidimp,
             skipping Morgrem 647) in a single Evolve event.
  - Switch:  {cardIdActive/serialActive = card that WAS active, now bumped to bench;
              cardIdBench/serialBench  = card that WAS benched, now the new active;
              playerIndex = whose BOARD this happened on (i.e. the victim's side for a
              forced switch via Boss's Orders, or the retreating player's own side for a
              voluntary retreat)}
  - Attack:  {attackId, cardId=attacker, serial, playerIndex}
  - HpChange:{cardId, serial, playerIndex, value (signed HP delta), putDamageCounter}
             Adrena-Brain (Munkidori) shows as a HEAL frame (putDamageCounter=False,
             value>0, on the ACTING player's own card) followed within a few frames by a
             DAMAGE frame (putDamageCounter=True, value<0, on the opponent's card) with the
             same |value| — i.e. damage counters physically moved from ours to theirs.
  - Play:    {cardId, serial, playerIndex} — card entered play (Pokemon/Trainer/Stadium).
  - MoveCard:{cardId, serial, playerIndex, fromArea, toArea} — zone codes empirically:
             1=deck, 2=hand, 3=discard, 4=active, 5=bench, 6=prize.

`current.stadium` is a list (0 or 1 entries) of {"id","name","playerIndex","serial"} —
diffing frame i-1 vs frame i around a Play(stadium) event tells us whether an opponent's
stadium was replaced.

Purity filter: mirrors scan_daily_dataset_for_experts.py — the player's own 60-card deck
(steps[1][player_idx].action) must overlap the reference decklist (data/deck_scan/top10_decks.json)
by >= --min-overlap (default 54/60, multiset intersection).

Incremental by day: each day's extracted per-game facts are cached at
<out>/playbook_by_day/<day>.jsonl so a later run adding a new day (e.g. 2026-07-20) only
pays the parse cost for that day; the final aggregate merges every cached day file found.

Usage:
    python3 scripts/mine_expert_playbook.py --days 2026-07-14 2026-07-15 2026-07-16 \
        2026-07-17 2026-07-18 2026-07-20 --limit 100   # smoke test
    python3 scripts/mine_expert_playbook.py --days 2026-07-14 ... 2026-07-20  # full run

Note: this script lives in a git worktree, but the raw episode dataset and the
il_grimmsnarl_v2/ output directory are large, gitignored, and shared across
worktrees — they live under the MAIN checkout (/Users/max/kaggle-pokemon-tcg-ai-battle),
not this worktree. Defaults below point there explicitly; override with flags if needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median, mean

REPO = Path(__file__).resolve().parent.parent  # this worktree
MAIN_REPO = Path("/Users/max/kaggle-pokemon-tcg-ai-battle")  # shared data lives here


def true_decks_from_data(data: dict) -> dict[int, list[int]]:
    """Extract the submitted 60-card deck for each player from an already-loaded
    replay dict: steps[1][player_idx].action, when it's a 60-int-id list.
    (Inlined here — this worktree's src/replays/parse.py predates this helper;
    kept self-contained rather than depending on the main checkout's newer copy.)"""
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

# ── Card-id constants (see task brief / CLAUDE.md deck description) ──────────
GRIMMSNARL_LINE = {646, 647, 648}
GRIMMSNARL_EX = 648
GRIMMSNARL_MORGREM = 647
GRIMMSNARL_IMPIDIMP = 646
MUNKIDORI = 112
FROSLASS = 104
SNORUNT = 860
RARE_CANDY = 1079
BOSS_ORDERS = 1182
SPIKEMUTH_GYM = 1259
PETREL = 1219
ENERGY_D = 7

MATCHUP_PRIORITY = [
    ("crustle", {344, 345}),
    ("archaludon", {169, 190}),
    ("lucario", {677, 678}),
    ("tr_mewtwo", {431}),
    ("abomasnow", {722, 723}),
    ("mirror", {646, 647, 648}),
    ("alakazam", {741, 742, 743}),
]


def classify_matchup(opp_deck_ids: list[int]) -> str:
    s = set(opp_deck_ids)
    for name, ids in MATCHUP_PRIORITY:
        if s & ids:
            return name
    return "generic"


def multiset_overlap(a: list[int], b: list[int]) -> int:
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values())


def load_card_db(csv_path: Path) -> tuple[dict, set]:
    """id -> {"name": str, "moves": [(move_name, damage_int_or_None), ...], "category": str}"""
    db = {}
    stadium_ids = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                cid = int(row["Card ID"])
            except (KeyError, ValueError):
                continue
            name = row.get("Card Name", "")
            typ = row.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
            if typ == "Stadium":
                stadium_ids.add(cid)
            entry = db.setdefault(cid, {"name": name, "moves": [], "category": typ})
            mv = row.get("Move Name") or ""
            dmg_raw = (row.get("Damage") or "").strip()
            dmg = None
            if dmg_raw and dmg_raw not in ("n/a", ""):
                digits = "".join(ch for ch in dmg_raw if ch.isdigit())
                if digits:
                    dmg = int(digits)
            if mv and mv != "n/a":
                entry["moves"].append((mv, dmg))
    return db, stadium_ids


def card_name(db: dict, cid: int) -> str:
    e = db.get(cid)
    return e["name"] if e else f"id={cid}"


def guess_attack_name(db: dict, attacker_cid: int, observed_dmg: int | None) -> str:
    e = db.get(attacker_cid)
    if not e or not e["moves"]:
        return f"attack(cardId={attacker_cid})"
    if observed_dmg is not None:
        for mv, dmg in e["moves"]:
            if dmg == observed_dmg:
                return mv
    non_ability = [mv for mv, _ in e["moves"] if not mv.startswith("[Ability]")]
    if len(non_ability) == 1:
        return non_ability[0]
    return "/".join(non_ability) if non_ability else e["moves"][0][0]


# ── Per-game extraction ───────────────────────────────────────────────────

def hp_pct(entry: dict | None) -> float | None:
    if not entry:
        return None
    mx = entry.get("maxHp")
    if not mx:
        return None
    return round(100.0 * entry.get("hp", 0) / mx, 1)


def find_active_entry(cur_players: list, player_idx: int, serial: int | None = None,
                       cid: int | None = None) -> dict | None:
    if player_idx >= len(cur_players):
        return None
    active = cur_players[player_idx].get("active") or []
    for a in active:
        if serial is not None and a.get("serial") == serial:
            return a
        if serial is None and cid is not None and a.get("id") == cid:
            return a
    if active and serial is None and cid is None:
        return active[0]
    return None


def find_any_entry(cur_players: list, player_idx: int, serial: int) -> dict | None:
    """Look for a card with this serial anywhere on player_idx's board (active+bench)."""
    if player_idx >= len(cur_players):
        return None
    p = cur_players[player_idx]
    for a in (p.get("active") or []) + (p.get("bench") or []):
        if a.get("serial") == serial:
            return a
    return None


def analyze_game(v: list[dict], expert_idx: int, opp_idx: int) -> dict:
    facts = {
        "n_frames": len(v),
        "max_turn": max((fr["current"]["turn"] for fr in v), default=0),
        "adrena_brain_events": [],
        "attach_events": [],
        "boss_orders_events": [],
        "opp_boss_orders_events": [],  # opponent using Boss's Orders on us (context for retreat filter)
        "stadium_events": [],
        "petrel_fetches": [],
        "retreat_events": [],
        "munkidori_spans": [],
        "evolve_648_events": [],
    }

    # Track opponent's Play(BOSS_ORDERS) frame indices to distinguish forced vs voluntary Switch.
    opp_boss_frames: list[int] = []
    my_boss_frames: list[int] = []
    for i, fr in enumerate(v):
        for l in fr["logs"]:
            if l["type"] == "Play" and l["cardId"] == BOSS_ORDERS:
                if l["playerIndex"] == opp_idx:
                    opp_boss_frames.append(i)
                elif l["playerIndex"] == expert_idx:
                    my_boss_frames.append(i)

    def near_any(frame_idx: int, anchors: list[int], window: int = 4) -> bool:
        return any(0 <= frame_idx - a <= window for a in anchors)

    # ---- Adrena-Brain: pair heal(own, +) frames with damage(opp, -) frames ----
    heal_candidates = []  # (frame_idx, abs_value, source_cid, source_serial)
    dmg_candidates = []   # (frame_idx, abs_value, target_cid, target_serial)
    for i, fr in enumerate(v):
        types = {l["type"] for l in fr["logs"]}
        if "Attack" in types:
            continue  # exclude attack-driven HpChange (recoil / self-damage)
        for l in fr["logs"]:
            if l["type"] != "HpChange":
                continue
            if l["playerIndex"] == expert_idx and l.get("putDamageCounter") is False and l["value"] > 0:
                heal_candidates.append((i, l["value"], l["cardId"], l["serial"]))
            elif l["playerIndex"] == opp_idx and l.get("putDamageCounter") is True and l["value"] < 0:
                dmg_candidates.append((i, -l["value"], l["cardId"], l["serial"]))

    used_dmg = set()
    for hi, hval, h_cid, h_serial in heal_candidates:
        for di, (dj, dval, d_cid, d_serial) in enumerate(dmg_candidates):
            if di in used_dmg:
                continue
            if dj < hi or dj - hi > 4:
                continue
            if dval != hval:
                continue
            used_dmg.add(di)
            frame = v[dj]
            target_entry = find_any_entry(frame["current"]["players"], opp_idx, d_serial)
            zone = "unknown"
            if target_entry is not None:
                active_serials = {a.get("serial") for a in (frame["current"]["players"][opp_idx].get("active") or [])}
                zone = "active" if d_serial in active_serials else "bench"
            facts["adrena_brain_events"].append({
                "turn": frame["current"]["turn"],
                "amount": hval,
                "source_cid": h_cid,
                "target_cid": d_cid,
                "target_zone": zone,
            })
            break

    # ---- Attach routing ----
    attach_ordinal = 0
    for fr in v:
        for l in fr["logs"]:
            if l["type"] == "Attach" and l["playerIndex"] == expert_idx:
                attach_ordinal += 1
                tgt = l["cardIdTarget"]
                if tgt in GRIMMSNARL_LINE:
                    cls = "grimmsnarl_line"
                elif tgt == MUNKIDORI:
                    cls = "munkidori"
                else:
                    cls = "other"
                facts["attach_events"].append({
                    "turn": fr["current"]["turn"],
                    "ordinal": attach_ordinal,
                    "energy_cid": l["cardId"],
                    "target_cid": tgt,
                    "target_class": cls,
                })

    # ---- Boss's Orders (ours) ----
    for i in my_boss_frames:
        turn = v[i]["current"]["turn"]
        for j in range(i, min(i + 5, len(v))):
            found = False
            for l in v[j]["logs"]:
                if l["type"] == "Switch" and l["playerIndex"] == opp_idx:
                    pulled_serial = l["serialBench"]
                    pulled_cid = l["cardIdBench"]
                    entry = find_active_entry(v[j]["current"]["players"], opp_idx, serial=pulled_serial)
                    facts["boss_orders_events"].append({
                        "turn": turn,
                        "target_cid": pulled_cid,
                        "hp_pct": hp_pct(entry),
                    })
                    found = True
                    break
            if found:
                break

    # ---- Boss's Orders (opponent's, used on us) — needed for retreat-vs-forced filter, logged too ----
    for i in opp_boss_frames:
        turn = v[i]["current"]["turn"]
        for j in range(i, min(i + 5, len(v))):
            found = False
            for l in v[j]["logs"]:
                if l["type"] == "Switch" and l["playerIndex"] == expert_idx:
                    entry = find_active_entry(v[j]["current"]["players"], expert_idx, serial=l["serialBench"])
                    facts["opp_boss_orders_events"].append({
                        "turn": turn,
                        "target_cid": l["cardIdBench"],
                        "hp_pct": hp_pct(entry),
                    })
                    found = True
                    break
            if found:
                break

    # ---- Stadium ----
    for i, fr in enumerate(v):
        for l in fr["logs"]:
            if l["type"] == "Play" and l["cardId"] == SPIKEMUTH_GYM and l["playerIndex"] == expert_idx:
                before = v[i - 1]["current"].get("stadium") if i > 0 else []
                replaced_opp = bool(before) and before[0].get("playerIndex") == opp_idx
                facts["stadium_events"].append({
                    "turn": fr["current"]["turn"],
                    "replaced_opp": replaced_opp,
                })

    # ---- Petrel fetches ----
    for i, fr in enumerate(v):
        for l in fr["logs"]:
            if l["type"] == "Play" and l["cardId"] == PETREL and l["playerIndex"] == expert_idx:
                for j in range(i, min(i + 3, len(v))):
                    got = False
                    for l2 in v[j]["logs"]:
                        if l2["type"] == "MoveCard" and l2.get("fromArea") == 1 and l2.get("toArea") == 2 \
                                and l2["playerIndex"] == expert_idx:
                            facts["petrel_fetches"].append(l2["cardId"])
                            got = True
                            break
                    if got:
                        break

    # ---- Voluntary retreats (Switch on our board NOT anchored to opponent's Boss's Orders) ----
    for i, fr in enumerate(v):
        for l in fr["logs"]:
            if l["type"] == "Switch" and l["playerIndex"] == expert_idx:
                if near_any(i, opp_boss_frames, window=4):
                    continue  # this is the forced-switch consequence, already logged above
                before = v[i - 1]["current"] if i > 0 else fr["current"]
                prev_active = find_active_entry(before["players"], expert_idx, serial=l["serialActive"])
                facts["retreat_events"].append({
                    "turn": fr["current"]["turn"],
                    "hp_pct_before": hp_pct(prev_active),
                    "incoming_cid": l["cardIdBench"],
                })

    # ---- Munkidori active-duration spans ----
    prev_key = None
    span_start_turn = None
    prev_turn = None
    prev_hp = None
    for fr in v:
        cur = fr["current"]
        if expert_idx >= len(cur["players"]):
            continue
        active_list = cur["players"][expert_idx].get("active") or []
        a = active_list[0] if active_list else None
        key = (a["id"], a["serial"]) if a else None
        turn = cur["turn"]
        if key != prev_key:
            if prev_key is not None and prev_key[0] == MUNKIDORI:
                facts["munkidori_spans"].append({
                    "start_turn": span_start_turn,
                    "end_turn": prev_turn,
                    "n_ticks": (prev_turn - span_start_turn + 1) if span_start_turn is not None else None,
                    "died": prev_hp == 0,
                })
            if key is not None and key[0] == MUNKIDORI:
                span_start_turn = turn
            prev_key = key
        prev_turn = turn
        prev_hp = a["hp"] if a else None
    if prev_key is not None and prev_key[0] == MUNKIDORI:
        facts["munkidori_spans"].append({
            "start_turn": span_start_turn,
            "end_turn": prev_turn,
            "n_ticks": (prev_turn - span_start_turn + 1) if span_start_turn is not None else None,
            "died": prev_hp == 0,
        })

    # ---- Evolve into Grimmsnarl ex (648) ----
    for fr in v:
        for l in fr["logs"]:
            if l["type"] == "Evolve" and l["cardId"] == GRIMMSNARL_EX and l["playerIndex"] == expert_idx:
                facts["evolve_648_events"].append({
                    "turn": fr["current"]["turn"],
                    "via_rare_candy": l["cardIdTarget"] == GRIMMSNARL_IMPIDIMP,
                    "from_cid": l["cardIdTarget"],
                })

    return facts


# ── Narration for Archaludon recon ────────────────────────────────────────

NARRATE_TYPES = {"Play", "Attach", "Evolve", "Attack", "HpChange", "Switch", "Result", "Coin"}


def narrate_game(v: list[dict], db: dict, expert_idx: int, expert_name: str,
                  opp_idx: int, opp_name: str) -> str:
    lines = []
    lines.append(f"# {expert_name} (player {expert_idx}) vs {opp_name} (player {opp_idx})\n")
    lines.append(f"Result: {expert_name} WON\n")
    last_turn = None
    for fr in v:
        turn = fr["current"]["turn"]
        if turn != last_turn:
            who = "expert" if (turn % 2 == 1) == (expert_idx == 0) else "opponent"
            lines.append(f"\n## Turn {turn} ({expert_name if who == 'expert' else opp_name})\n")
            last_turn = turn
        for l in fr["logs"]:
            t = l["type"]
            if t not in NARRATE_TYPES:
                continue
            pidx = l.get("playerIndex")
            who_name = expert_name if pidx == expert_idx else (opp_name if pidx == opp_idx else "?")
            if t == "Play":
                lines.append(f"- {who_name} plays {card_name(db, l['cardId'])}")
            elif t == "Attach":
                lines.append(f"- {who_name} attaches {card_name(db, l['cardId'])} to {card_name(db, l['cardIdTarget'])} (#{l.get('serialTarget')})")
            elif t == "Evolve":
                rc = " [Rare Candy]" if l["cardId"] == GRIMMSNARL_EX and l["cardIdTarget"] == GRIMMSNARL_IMPIDIMP else ""
                lines.append(f"- {who_name} evolves {card_name(db, l['cardIdTarget'])} -> {card_name(db, l['cardId'])}{rc}")
            elif t == "Attack":
                dmg = None
                for l2 in fr["logs"]:
                    if l2["type"] == "HpChange" and l2.get("playerIndex") != pidx:
                        dmg = -l2["value"]
                        break
                mv = guess_attack_name(db, l["cardId"], dmg)
                dmg_s = f" ({dmg} dmg)" if dmg is not None else ""
                lines.append(f"- {who_name}'s {card_name(db, l['cardId'])} uses {mv}{dmg_s}")
            elif t == "HpChange":
                # NOTE: `putDamageCounter` is NOT a heal/damage flag — ordinary attack damage
                # also arrives with putDamageCounter=False (negative value). The sign of
                # `value` is what actually distinguishes damage (-) from healing/counter
                # removal (+); Adrena-Brain heals are recognized by putDamageCounter=False
                # AND value>0 specifically (see analyze_game(), which is unaffected by this
                # narration-only label and was cross-validated against raw select counts).
                sign = "+" if l["value"] > 0 else ""
                kind = "damage" if l["value"] < 0 else "heal/remove counters"
                lines.append(f"    HpChange: {card_name(db, l['cardId'])} ({who_name}) {sign}{l['value']} [{kind}]")
            elif t == "Switch":
                lines.append(f"- {who_name}: {card_name(db, l['cardIdBench'])} switches in, {card_name(db, l['cardIdActive'])} benched")
            elif t == "Result":
                lines.append(f"- GAME RESULT: reason={l.get('reason')} result={l.get('result')}")
            elif t == "Coin":
                pass
    return "\n".join(lines) + "\n"


# ── Aggregation ────────────────────────────────────────────────────────────

def pct(n, d):
    return round(100.0 * n / d, 1) if d else None


def safe_median(xs):
    xs = [x for x in xs if x is not None]
    return round(median(xs), 1) if xs else None


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return round(mean(xs), 2) if xs else None


def aggregate_bucket(games: list[dict]) -> dict:
    n = len(games)
    out = {"n_games": n}
    if n == 0:
        return out

    # Adrena-Brain
    ab_counts = [len(g["adrena_brain_events"]) for g in games]
    all_ab = [e for g in games for e in g["adrena_brain_events"]]
    first_ab_turns = [min((e["turn"] for e in g["adrena_brain_events"]), default=None) for g in games]
    zone_counter = Counter(e["target_zone"] for e in all_ab)
    out["adrena_brain"] = {
        "mean_activations_per_game": safe_mean(ab_counts),
        "median_activations_per_game": safe_median(ab_counts),
        "pct_games_with_ge1_activation": pct(sum(1 for c in ab_counts if c > 0), n),
        "median_turn_of_first_activation": safe_median(first_ab_turns),
        "target_zone_distribution_pct": {k: pct(v, len(all_ab)) for k, v in zone_counter.items()},
        "n_total_activations": len(all_ab),
    }

    # Attach routing
    all_attach = [e for g in games for e in g["attach_events"]]
    attach_counts = [len(g["attach_events"]) for g in games]
    class_counter = Counter(e["target_class"] for e in all_attach)
    ordinal_class = defaultdict(Counter)
    for e in all_attach:
        if e["ordinal"] <= 3:
            ordinal_class[e["ordinal"]][e["target_class"]] += 1
    first_munk_turns = []
    for g in games:
        munk_attaches = [e["turn"] for e in g["attach_events"] if e["target_class"] == "munkidori"]
        if munk_attaches:
            first_munk_turns.append(min(munk_attaches))
    out["attach_routing"] = {
        "mean_attaches_per_game": safe_mean(attach_counts),
        "target_class_distribution_pct": {k: pct(v, len(all_attach)) for k, v in class_counter.items()},
        "ordinal_1_2_3_target_class_pct": {
            str(k): {kk: pct(vv, sum(v.values())) for kk, vv in v.items()}
            for k, v in sorted(ordinal_class.items())
        },
        "pct_games_munkidori_ever_gets_energy": pct(len(first_munk_turns), n),
        "median_turn_first_energy_on_munkidori": safe_median(first_munk_turns),
        "n_total_attaches": len(all_attach),
    }

    # Boss's Orders
    all_bo = [e for g in games for e in g["boss_orders_events"]]
    bo_counts = [len(g["boss_orders_events"]) for g in games]
    bo_target_counter = Counter(e["target_cid"] for e in all_bo)
    out["boss_orders"] = {
        "mean_uses_per_game": safe_mean(bo_counts),
        "pct_games_with_ge1_use": pct(sum(1 for c in bo_counts if c > 0), n),
        "median_turn": safe_median([e["turn"] for e in all_bo]),
        "median_target_hp_pct": safe_median([e["hp_pct"] for e in all_bo]),
        "target_cid_distribution": dict(bo_target_counter.most_common(10)),
        "n_total_uses": len(all_bo),
    }

    # Stadium
    all_st = [e for g in games for e in g["stadium_events"]]
    first_st_turns = [min((e["turn"] for e in g["stadium_events"]), default=None) for g in games]
    out["stadium"] = {
        "pct_games_played_ge1": pct(sum(1 for g in games if g["stadium_events"]), n),
        "median_turn_first_play": safe_median(first_st_turns),
        "replace_rate_pct": pct(sum(1 for e in all_st if e["replaced_opp"]), len(all_st)),
        "n_total_plays": len(all_st),
    }

    # Petrel fetches
    all_fetch = [c for g in games for c in g["petrel_fetches"]]
    out["petrel"] = {
        "pct_games_played_ge1": pct(sum(1 for g in games if g["petrel_fetches"]), n),
        "n_total_fetches": len(all_fetch),
        "top_fetched_cids": Counter(all_fetch).most_common(10),
    }

    # Retreats
    all_rt = [e for g in games for e in g["retreat_events"]]
    rt_counts = [len(g["retreat_events"]) for g in games]
    out["retreats"] = {
        "mean_per_game": safe_mean(rt_counts),
        "median_hp_pct_at_retreat": safe_median([e["hp_pct_before"] for e in all_rt]),
        "n_total": len(all_rt),
    }
    all_opp_bo = [e for g in games for e in g["opp_boss_orders_events"]]
    out["opp_boss_orders_on_us"] = {
        "mean_per_game": safe_mean([len(g["opp_boss_orders_events"]) for g in games]),
        "median_target_hp_pct": safe_median([e["hp_pct"] for e in all_opp_bo]),
        "n_total": len(all_opp_bo),
    }

    # Munkidori active-duration
    all_spans = [s for g in games for s in g["munkidori_spans"]]
    out["munkidori_active_duration"] = {
        "pct_games_munkidori_ever_active": pct(sum(1 for g in games if g["munkidori_spans"]), n),
        "n_total_spans": len(all_spans),
        "mean_ticks_per_span": safe_mean([s["n_ticks"] for s in all_spans]),
        "median_ticks_per_span": safe_median([s["n_ticks"] for s in all_spans]),
        "pct_spans_ending_in_death": pct(sum(1 for s in all_spans if s["died"]), len(all_spans)),
    }

    # Evolves
    all_ev = [e for g in games for e in g["evolve_648_events"]]
    first_ev_turns = []
    first_ev_via_rc = []
    for g in games:
        if g["evolve_648_events"]:
            first = min(g["evolve_648_events"], key=lambda e: e["turn"])
            first_ev_turns.append(first["turn"])
            first_ev_via_rc.append(first["via_rare_candy"])
    out["evolve_grimmsnarl_ex"] = {
        "pct_games_648_never_appears": pct(sum(1 for g in games if not g["evolve_648_events"]), n),
        "median_turn_first_648": safe_median(first_ev_turns),
        "pct_first_648_via_rare_candy": pct(sum(1 for x in first_ev_via_rc if x), len(first_ev_via_rc)),
        "pct_all_648_evolves_via_rare_candy": pct(sum(1 for e in all_ev if e["via_rare_candy"]), len(all_ev)),
        "n_total_evolve_events": len(all_ev),
    }

    return out


# ── Main scan ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--targets", nargs="+", default=["jiatu.l", "bono"])
    ap.add_argument("--daily-root", default=str(MAIN_REPO / "data" / "episodes" / "official"))
    ap.add_argument("--ref-decks", default=str(MAIN_REPO / "data" / "deck_scan" / "top10_decks.json"))
    ap.add_argument("--card-csv", default=str(MAIN_REPO / "data" / "EN_Card_Data.csv"))
    ap.add_argument("--out", default=str(MAIN_REPO / "data" / "il_grimmsnarl_v2"))
    ap.add_argument("--min-overlap", type=int, default=54)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit; else cap files scanned per day (smoke test)")
    ap.add_argument("--archaludon-recon-n", type=int, default=3)
    ap.add_argument("--archaludon-recon-dir", default=str(MAIN_REPO / "data" / "il_grimmsnarl_v2" / "archaludon_recon"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_day_dir = out_dir / "playbook_by_day"
    by_day_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = Path(args.archaludon_recon_dir)
    recon_dir.mkdir(parents=True, exist_ok=True)

    ref_decks = json.loads(Path(args.ref_decks).read_text())
    targets = set(args.targets)
    needles = {t: t.encode("utf-8") for t in targets}
    db, stadium_ids = load_card_db(Path(args.card_csv))

    t0 = time.time()

    for day in args.days:
        day_dir = Path(args.daily_root) / day
        if not day_dir.is_dir():
            print(f"[mine] SKIP missing day dir: {day_dir}")
            continue
        files = sorted(day_dir.glob("*.json"))
        if args.limit:
            files = files[: args.limit]
        print(f"[mine] {day}: {len(files)} JSON files")

        day_out_path = by_day_dir / f"{day}.jsonl"
        n_scanned = 0
        n_grep_hits = 0
        n_kept = 0
        n_impure = 0
        n_no_visualize = 0
        recon_written = 0

        with open(day_out_path, "w") as out_f:
            for fp in files:
                n_scanned += 1
                if n_scanned % 1000 == 0:
                    print(f"[mine]   {day}: scanned={n_scanned} grep_hits={n_grep_hits} kept={n_kept}")
                try:
                    raw = fp.read_bytes()
                except Exception as e:
                    print(f"[mine]   read error {fp.name}: {e}")
                    continue
                hit_targets = [t for t, needle in needles.items() if needle in raw]
                if not hit_targets:
                    continue
                n_grep_hits += 1
                try:
                    data = json.loads(raw)
                except Exception as e:
                    print(f"[mine]   json error {fp.name}: {e}")
                    continue

                agents = data.get("info", {}).get("Agents", [])
                rewards = data.get("rewards", [])
                names = [a.get("Name") for a in agents]
                true_decks = true_decks_from_data(data)

                steps = data.get("steps", [])
                if not steps or not steps[0] or "visualize" not in steps[0][0]:
                    n_no_visualize += 1
                    continue
                v = steps[0][0]["visualize"]

                for expert_idx, name in enumerate(names):
                    if name not in targets or name not in hit_targets:
                        continue
                    deck = true_decks.get(expert_idx)
                    if deck is None or name not in ref_decks:
                        n_impure += 1
                        continue
                    overlap = multiset_overlap(deck, ref_decks[name])
                    if overlap < args.min_overlap:
                        n_impure += 1
                        continue

                    opp_idx = 1 - expert_idx
                    opp_deck = true_decks.get(opp_idx, [])
                    matchup = classify_matchup(opp_deck)
                    won = expert_idx < len(rewards) and rewards[expert_idx] > 0

                    facts = analyze_game(v, expert_idx, opp_idx)
                    record = {
                        "episode_id": data.get("id") or data.get("info", {}).get("EpisodeId"),
                        "day": day,
                        "file": fp.name,
                        "expert_name": name,
                        "expert_idx": expert_idx,
                        "opp_name": names[opp_idx] if opp_idx < len(names) else None,
                        "matchup": matchup,
                        "won": won,
                        "deck_overlap": overlap,
                        **facts,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    n_kept += 1

                    if matchup == "archaludon" and won and recon_written < args.archaludon_recon_n:
                        try:
                            md = narrate_game(v, db, expert_idx, name, opp_idx, names[opp_idx])
                            recon_path = recon_dir / f"game_{record['episode_id']}.md"
                            recon_path.write_text(md)
                            recon_written += 1
                            print(f"[mine]   wrote archaludon recon: {recon_path}")
                        except Exception as e:
                            print(f"[mine]   recon narration failed for {fp.name}: {e}")

        print(f"[mine] {day} DONE: scanned={n_scanned} grep_hits={n_grep_hits} kept={n_kept} "
              f"impure={n_impure} no_visualize={n_no_visualize} elapsed={time.time()-t0:.1f}s")

    # ── Merge all cached day files and aggregate ──────────────────────────
    all_records = []
    for p in sorted(by_day_dir.glob("*.jsonl")):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))

    print(f"[mine] merged {len(all_records)} game records across {len(list(by_day_dir.glob('*.jsonl')))} day files")

    by_bucket = defaultdict(lambda: {"won": [], "lost": []})
    for rec in all_records:
        bucket = by_bucket[rec["matchup"]]
        bucket["won" if rec["won"] else "lost"].append(rec)

    matchup_summary = {}
    for matchup, buckets in by_bucket.items():
        matchup_summary[matchup] = {
            "won": aggregate_bucket(buckets["won"]),
            "lost": aggregate_bucket(buckets["lost"]),
        }

    playbook = {
        "meta": {
            "days": sorted({r["day"] for r in all_records}),
            "targets": sorted(targets),
            "n_games_total": len(all_records),
            "min_overlap": args.min_overlap,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "n_games_by_matchup_outcome": {
            m: {"won": len(b["won"]), "lost": len(b["lost"])} for m, b in by_bucket.items()
        },
        "matchups": matchup_summary,
    }

    out_path = out_dir / "expert_playbook.json"
    out_path.write_text(json.dumps(playbook, indent=2, ensure_ascii=False))
    print(f"[mine] wrote {out_path}")

    print("\n=== n_games by matchup x outcome ===")
    for m, counts in sorted(playbook["n_games_by_matchup_outcome"].items(),
                             key=lambda kv: -(kv[1]["won"] + kv[1]["lost"])):
        print(f"  {m:12s} won={counts['won']:4d} lost={counts['lost']:4d}")


if __name__ == "__main__":
    main()
