#!/usr/bin/env python3
"""fidelity_expert_topk.py — OFFLINE top-1 decision fidelity of the FULL v074 pilot
(submissions/<cand>/main.py's `agent()`: loop-breaker + 6 guards + scorer + fallback,
NOT just scoring_grimmsnarl.py) against REAL jiatu.l/bono expert decisions.

This is the docs/gold/BREAK_CEILING_900.md §4 "first cheap experiment": for every
decision point an expert (rating-1130 owner jiatu.l / bono) actually faced, feed the
expert's exact fog-of-war observation to our pilot and check whether the pilot's chosen
option index(es) equal the expert's actual choice. Broken down by decision category
(energy-attach / retreat / boss-target / adrena-target / attack / ...). Fully offline,
deterministic: no live games, no gate, no submit — the noise floors that plague live
(+-50 mu) and the N=1000 gate (+-1-2pp) do not apply; we compare concrete decisions on
concrete boards, not game outcomes.

WHY the (obs, action) substrate is byte-faithful, not a reconstruction:
`steps[t][p]['observation']` in an official daily dump is the EXACT int-encoded,
fog-of-war obs_dict the live engine handed the agent (verified: select.context=41 int,
option[k]={'type':1} int -- matches the pilot's `int(sel['context'])` code), and
`steps[t+1][p]['action']` is the option indices the expert returned. src/replays/parse.py
`parse_replay()` pairs them correctly (the action lives in step t+1, a kaggle-environments
quirk) and already skips deck-submission / IS_FIRST / MULLIGAN / empty-action steps. So we
feed the pilot precisely what it would see live and compare to what a 1130-rated human did.

Engine: the pilot needs `from cg.api import to_observation_class/all_card_data/all_attack`
(card DB + typed-obs conversion for the scorer). We register the native macOS engine
(cg_native, libcg.dylib) under the `cg`/`cg.api`/... module names BEFORE loading the
pilot, exactly like scripts/gate_behavioral_fidelity.py. NO battle_start/battle_select is
ever called -- this is a pure static-obs replay, so the engine-RNG / threading caveats of
the self-play gates do not apply here.

Path attribution (which guard/scorer/fallback made each pick) is read from the DELTA of the
pilot's own module-level counters (_n_promo_guard, _n_ko_guard, _n_adrena_trigger,
_n_munkidori_energy_guard, _n_energy_guard, _n_model_decisions, _n_loop_breaker,
_n_trivial_fallback) across each agent() call -- robust, unlike parsing the interleaved
[dbg] stderr. The Adrena-sequence continuation path increments none of these, so it is
detected from the captured `guard=adrena_seq` stderr line as a fallback signal.

Controls reported per category (the green/red instrument, per BREAK_CEILING §4):
  * exact_match_rate   : set(our) == set(expert), the headline fidelity.
  * chance_floor_mean  : mean over decisions of 1/C(n_options, k) -- the match rate a
                         uniform-random LEGAL answer would score. match ~ floor => the
                         pilot is no better than chance in that category (GREEN: big
                         addressable gap). match >> floor => pilot tracks the expert.
  * lift               : exact_match_rate - chance_floor_mean.
  * fallback_share     : fraction of decisions the pilot answered via the trivial fallback
                         (path=trivial_fallback) -- high fallback CONTAMINATES the number
                         (the pilot isn't really "deciding"); reported so it can't hide.
Split additionally by expert name (jiatu.l vs bono -- independent replication) and by
whether the expert WON that game (agreement-on-wins vs agreement-on-losses).

Usage:
    .venv/bin/python scripts/fidelity_expert_topk.py --limit 3          # smoke (3 dumps)
    .venv/bin/python scripts/fidelity_expert_topk.py                    # full 07-20 corpus
    .venv/bin/python scripts/fidelity_expert_topk.py --candidate v074_fullmetal \
        --days 2026-07-20 --out out/v074_fidelity/topk_0720.json
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
MAIN_REPO = Path("/Users/max/kaggle-pokemon-tcg-ai-battle")
SRC = WORKTREE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ── Native engine shim (macOS arm64) — must precede any pilot import ──────────
import cg_native  # noqa: E402
import cg_native.api  # noqa: E402
import cg_native.game  # noqa: E402
import cg_native.sim  # noqa: E402
import cg_native.utils  # noqa: E402

sys.modules.setdefault("cg", cg_native)
sys.modules.setdefault("cg.api", cg_native.api)
sys.modules.setdefault("cg.game", cg_native.game)
sys.modules.setdefault("cg.sim", cg_native.sim)
sys.modules.setdefault("cg.utils", cg_native.utils)

from replays.parse import parse_replay, _CONTEXT_NAMES  # noqa: E402


def true_decks_from_data(data: dict) -> dict:
    """Submitted 60-card deck per player from steps[1][pi].action (inlined -- this
    worktree's src/replays/parse.py predates the shared helper; same logic as
    scripts/mine_expert_playbook.py)."""
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

# ── OptionType / SelectContext int constants (mirror submissions/*/main.py) ───
OPT_NUMBER, OPT_YES, OPT_CARD = 0, 1, 3
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY = 7, 8, 9, 10
OPT_RETREAT, OPT_ATTACK, OPT_END = 12, 13, 14
CTX_MAIN, CTX_SWITCH, CTX_TO_ACTIVE = 0, 3, 4
CTX_DAMAGE_COUNTER, CTX_REMOVE_DAMAGE_COUNTER, CTX_REMOVE_DAMAGE_COUNTER_COUNT = 13, 16, 40
AREA_HAND = 2
BOSS_ORDERS_ID = 1182

# Pilot module-level counters snapshotted per agent() call for path attribution.
# Exactly one increments per call (agent() returns after the first firing branch),
# except the Adrena-sequence continuation, which increments none -> stderr-detected.
PATH_COUNTERS = [
    "_n_promo_guard", "_n_ko_guard", "_n_adrena_trigger", "_n_munkidori_energy_guard",
    "_n_energy_guard", "_n_model_decisions", "_n_loop_breaker", "_n_trivial_fallback",
]
# agent() firing order; first counter that moved is the path.
PATH_ORDER = [
    ("_n_promo_guard", "promotion"),
    ("_n_ko_guard", "ko_lethal"),
    ("_n_adrena_trigger", "adrena_trigger"),
    ("_n_munkidori_energy_guard", "munkidori_energy"),
    ("_n_energy_guard", "energy_attach"),
    ("_n_loop_breaker", "loop_breaker"),
    ("_n_model_decisions", "scorer"),
    ("_n_trivial_fallback", "trivial_fallback"),
]


def multiset_overlap(a, b) -> int:
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values())


def load_pilot(sub_name: str):
    """Load submissions/<sub_name>/main.py ONCE as a module (cg shim already in
    place). The pilot resets its own per-game state whenever agent() is called
    with select=None, so a single load is reused across all games."""
    sub_dir = WORKTREE / "submissions" / sub_name
    main_path = sub_dir / "main.py"
    assert main_path.exists(), f"pilot not found: {main_path}"
    if str(sub_dir) not in sys.path:
        sys.path.insert(0, str(sub_dir))
    old_cwd = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(f"_pilot_{sub_name}", main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = str(main_path)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(old_cwd)
    return mod


def _snapshot(pilot) -> dict:
    return {k: int(getattr(pilot, k, 0) or 0) for k in PATH_COUNTERS}


def _path_of(before: dict, after: dict, stderr_text: str) -> str:
    if "guard=adrena_seq" in stderr_text:
        return "adrena_seq"
    for key, label in PATH_ORDER:
        if after.get(key, 0) > before.get(key, 0):
            return label
    return "unknown"


def _resolve_hand_card_id(obs: dict, opt: dict):
    """For a MAIN Play/Attach option sourced from hand (area==HAND, index), return the
    hand card's id (own hand is visible in the player's own fog-of-war obs). Used only
    to distinguish a Boss's Orders (1182) play from other supporter/item plays."""
    try:
        if opt.get("area") != AREA_HAND:
            return None
        idx = opt.get("index")
        if not isinstance(idx, int):
            return None
        cur = obs.get("current") or {}
        me = int(cur.get("yourIndex") or 0)
        players = cur.get("players") or []
        if me >= len(players):
            return None
        hand = players[me].get("hand") or []
        if 0 <= idx < len(hand):
            h = hand[idx]
            return int(h.get("id")) if isinstance(h, dict) and h.get("id") is not None else None
    except Exception:
        return None
    return None


def categorize(obs: dict, chosen: list) -> str:
    """Map one decision point to a category label. MAIN decisions split by the CHOSEN
    option's type (that is what makes it an attack vs an energy-attach vs a Boss play);
    non-MAIN by SelectContext. Opponent-board targeting (playerIndex != me) in a
    switch/to-active context is the Boss's-Orders / gust drag target."""
    sel = obs.get("select") or {}
    ctx = sel.get("context", -1)
    opts = sel.get("option") or []
    ci = chosen[0] if chosen else None
    opt = opts[ci] if (isinstance(ci, int) and 0 <= ci < len(opts)) else None
    ctype = opt.get("type") if isinstance(opt, dict) else None

    if ctx == CTX_MAIN:
        if ctype == OPT_ATTACK:
            return "attack"
        if ctype == OPT_ATTACH:
            return "energy-attach"
        if ctype == OPT_ABILITY:
            return "ability(adrena-trigger)"
        if ctype == OPT_RETREAT:
            return "retreat"
        if ctype == OPT_EVOLVE:
            return "evolve"
        if ctype == OPT_END:
            return "end-turn"
        if ctype == OPT_PLAY:
            cid = _resolve_hand_card_id(obs, opt)
            return "boss-play" if cid == BOSS_ORDERS_ID else "play-support"
        return f"main-other(t={ctype})"

    if ctx in (CTX_SWITCH, CTX_TO_ACTIVE):
        me = int((obs.get("current") or {}).get("yourIndex") or 0)
        opl = opt.get("playerIndex") if isinstance(opt, dict) else None
        if opl is not None and int(opl) != me:
            return "boss/gust-target(opp)"
        return "promotion/switch(own)"

    if ctx == CTX_DAMAGE_COUNTER:
        return "adrena-target(place-on-opp)"
    if ctx == CTX_REMOVE_DAMAGE_COUNTER:
        return "adrena-source(remove-own)"
    if ctx == CTX_REMOVE_DAMAGE_COUNTER_COUNT:
        return "adrena-count"

    return f"ctx:{_CONTEXT_NAMES.get(ctx, ctx)}"


# Map granular categories -> the 5 BREAK_CEILING §4 headline buckets (+positioning/other).
HEADLINE = {
    "energy-attach": "energy",
    "retreat": "retreat/positioning",
    "promotion/switch(own)": "retreat/positioning",
    "boss/gust-target(opp)": "boss-target",
    "boss-play": "boss-target",
    "adrena-target(place-on-opp)": "adrena-target",
    "adrena-source(remove-own)": "adrena-target",
    "adrena-count": "adrena-target",
    "ability(adrena-trigger)": "adrena-target",
    "attack": "attack",
}


# ── Semantic (index-free) option resolution ──────────────────────────────────
# Raw index match conflates strategic disagreement with fungible-index noise: an
# expert choosing energies [7,6,5] and the pilot choosing [0,1,2] pick the SAME cards
# (identical {D} energies) at different indices. The semantic signature resolves each
# chosen option to the CARD / ATTACK / ACTION it refers to, so duplicates collapse and
# only genuine card-level disagreement is counted. Reported only over "resolvable"
# decisions (every chosen option resolved to a concrete card id / attack id), which is
# the strategically meaningful subset (own hand+board + opp public board; own-deck
# searches stay unresolvable and are excluded rather than falsely scored as agreement).
ZONE = {1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench", 6: "prize"}


def _resolve_ref(obs: dict, opt: dict):
    """(kind, key, resolved_bool) for one option index, resolved to the CARD / ACTION it
    refers to (index-free), keyed per the engine option schemas empirically confirmed on
    the 07-20 corpus:
      PLAY(7)   = {index,type}         -> hand[index]           (WHICH card is played)
      ATTACH(8) = {area,index,inPlay*} -> inPlay dest mon       (WHOM energy goes to; the
                                          source energy is fungible {D}, so ignore it)
      EVOLVE(9) = {area,index,inPlay*} -> (evolver card, target mon)
      ABILITY(10)= {area,index}        -> the mon whose ability
      CARD(3)/energy-subselect(4,5,6)  -> referenced (playerIndex,area,index) card
      ATTACK(13)-> attackId ; NUMBER(0)-> number ; RETREAT/END/YES/NO -> the action
      any option carrying cardId directly (e.g. SKILL_ORDER type 15) -> that cardId
    """
    t = opt.get("type")
    cur = obs.get("current") or {}
    me = int(cur.get("yourIndex") or 0)
    players = cur.get("players") or []

    def card_at(pi, area, idx):
        if area in ZONE and isinstance(idx, int) and isinstance(pi, int) and 0 <= pi < len(players):
            zone = players[pi].get(ZONE[area]) or []
            if 0 <= idx < len(zone):
                c = zone[idx]
                return c.get("id") if isinstance(c, dict) else c
        return None

    if opt.get("cardId") is not None:
        return ("card", (None, int(opt["cardId"])), True)
    if t == OPT_ATTACK:
        aid = opt.get("attackId")
        return ("atk", aid, aid is not None)
    if t == OPT_NUMBER:
        num = opt.get("number")
        return ("num", num, num is not None)
    if t in (OPT_RETREAT, OPT_END, OPT_YES, 2):  # 2 == NO
        return ("act", t, True)
    if t == OPT_PLAY:
        cid = card_at(me, AREA_HAND, opt.get("index"))
        return ("play", cid, cid is not None)
    if t == OPT_ATTACH:
        tgt = card_at(me, opt.get("inPlayArea"), opt.get("inPlayIndex"))
        return ("attach_to", tgt, tgt is not None)
    if t == OPT_EVOLVE:
        what = card_at(me, opt.get("area", AREA_HAND), opt.get("index"))
        onto = card_at(me, opt.get("inPlayArea"), opt.get("inPlayIndex"))
        return ("evolve", (what, onto), what is not None and onto is not None)
    if t == OPT_ABILITY:
        who = card_at(me, opt.get("area"), opt.get("index"))
        return ("abil", who, who is not None)
    pi = opt.get("playerIndex")
    pi = int(pi) if pi is not None else me
    cid = card_at(pi, opt.get("area"), opt.get("index"))
    if cid is not None:
        return ("card", (pi != me, cid), True)
    return ("t", t, False)


def answer_sig(obs: dict, indices):
    """(signature, resolvable) — signature is an index-free multiset of what the chosen
    options refer to; resolvable is True iff every chosen option resolved concretely."""
    opts = (obs.get("select") or {}).get("option") or []
    parts = []
    resolvable = True
    for i in (indices or []):
        if isinstance(i, int) and 0 <= i < len(opts):
            kind, key, ok = _resolve_ref(obs, opts[i])
            parts.append(repr((kind, key)))
            resolvable = resolvable and ok
        else:
            parts.append(f"oob:{i}")
            resolvable = False
    return tuple(sorted(parts)), resolvable


def _chance_floor(n_options: int, k: int, min_c: int, max_c: int):
    """Probability a uniform-random LEGAL answer equals the expert's exact set.
    Well-defined only when the decision requires exactly k distinct picks
    (minCount==maxCount==k, 1<=k<=n). Returns None otherwise (excluded from floor mean)."""
    if min_c == max_c == k and 1 <= k <= n_options:
        try:
            return 1.0 / math.comb(n_options, k)
        except Exception:
            return None
    return None


def evaluate(pilot, dumps: list, targets: set, ref_decks: dict, min_overlap: int,
             rows_out) -> dict:
    stats = defaultdict(lambda: {
        "n": 0, "match": 0, "floor_sum": 0.0, "floor_n": 0,
        "paths": Counter(), "n_single": 0, "nopt_sum": 0,
        "sem_res": 0, "sem_match": 0,
    })
    by_name = defaultdict(lambda: {"n": 0, "match": 0})
    by_won = defaultdict(lambda: {"n": 0, "match": 0})
    totals = {"episodes_scanned": 0, "episodes_kept": 0, "episodes_impure": 0,
              "decisions": 0, "match": 0, "agent_exceptions": 0}

    def record(cat, matched, floor, path, n_options, k, name, won, sem_matched, resolvable):
        s = stats[cat]
        s["n"] += 1
        s["match"] += int(matched)
        s["nopt_sum"] += n_options
        if k == 1:
            s["n_single"] += 1
        if floor is not None:
            s["floor_sum"] += floor
            s["floor_n"] += 1
        if resolvable:
            s["sem_res"] += 1
            s["sem_match"] += int(sem_matched)
        s["paths"][path] += 1
        by_name[name]["n"] += 1
        by_name[name]["match"] += int(matched)
        by_won[won]["n"] += 1
        by_won[won]["match"] += int(matched)
        totals["decisions"] += 1
        totals["match"] += int(matched)

    for fp in dumps:
        try:
            raw = fp.read_bytes()
        except Exception:
            continue
        hit = [t for t in targets if t.encode("utf-8") in raw]
        if not hit:
            continue
        totals["episodes_scanned"] += 1
        try:
            data = json.loads(raw)
        except Exception:
            continue
        agents = data.get("info", {}).get("Agents", [])
        names = [a.get("Name") for a in agents]
        rewards = data.get("rewards", [])
        true_decks = true_decks_from_data(data)

        # Which target-player indices in THIS episode are pure enough to score?
        expert_idxs = []
        for pi, nm in enumerate(names):
            if nm not in targets or nm not in hit:
                continue
            deck = true_decks.get(pi)
            if deck is None or nm not in ref_decks:
                totals["episodes_impure"] += 1
                continue
            if multiset_overlap(deck, ref_decks[nm]) < min_overlap:
                totals["episodes_impure"] += 1
                continue
            expert_idxs.append(pi)
        if not expert_idxs:
            continue

        try:
            pairs = parse_replay(fp)
        except Exception:
            continue

        for pi in expert_idxs:
            totals["episodes_kept"] += 1
            name = names[pi]
            # Reset pilot per-game state exactly as live (select=None boundary).
            try:
                pilot.agent({"select": None})
            except Exception:
                pass
            ep_pairs = [p for p in pairs if p["player_index"] == pi]
            for p in ep_pairs:
                obs = p["obs_dict"]
                sel = obs.get("select") or {}
                opts = sel.get("option") or []
                n_options = len(opts)
                expert = list(p["chosen_option_indices"])
                k = len(expert)
                min_c = int(sel.get("minCount") or 1)
                max_c = int(sel.get("maxCount") or 1)

                before = _snapshot(pilot)
                buf = io.StringIO()
                try:
                    with contextlib.redirect_stderr(buf):
                        our = pilot.agent(obs)
                except Exception:
                    totals["agent_exceptions"] += 1
                    our = None
                after = _snapshot(pilot)
                path = _path_of(before, after, buf.getvalue())

                matched = (our is not None and set(our) == set(expert))
                floor = _chance_floor(n_options, k, min_c, max_c)
                cat = categorize(obs, expert)
                won = bool(p.get("won"))
                exp_sig, exp_res = answer_sig(obs, expert)
                our_sig, our_res = answer_sig(obs, our if our is not None else [])
                resolvable = bool(exp_res and our_res)
                sem_matched = (our is not None and exp_sig == our_sig)
                record(cat, matched, floor, path, n_options, k, name, won, sem_matched, resolvable)
                if rows_out is not None:
                    rows_out.write(json.dumps({
                        "episodeId": p.get("episodeId"), "name": name, "won": won,
                        "turn": (obs.get("current") or {}).get("turn"),
                        "category": cat, "headline": HEADLINE.get(cat, "other"),
                        "ctx": sel.get("context"), "n_options": n_options, "k": k,
                        "expert": expert, "our": (list(our) if our is not None else None),
                        "matched": bool(matched), "sem_matched": bool(sem_matched),
                        "resolvable": resolvable, "floor": floor, "path": path,
                        "exp_ref": list(exp_sig), "our_ref": list(our_sig),
                    }) + "\n")

    return _finalize(stats, by_name, by_won, totals)


def _finalize(stats, by_name, by_won, totals) -> dict:
    cats = {}
    for cat, s in stats.items():
        n = s["n"]
        mr = s["match"] / n if n else 0.0
        floor_mean = (s["floor_sum"] / s["floor_n"]) if s["floor_n"] else None
        sem_rate = (s["sem_match"] / s["sem_res"]) if s["sem_res"] else None
        cats[cat] = {
            "headline": HEADLINE.get(cat, "other"),
            "n_decisions": n,
            "exact_match_rate": round(mr, 4),
            "semantic_match_rate": round(sem_rate, 4) if sem_rate is not None else None,
            "n_resolvable": s["sem_res"],
            "chance_floor_mean": round(floor_mean, 4) if floor_mean is not None else None,
            "lift": round(mr - floor_mean, 4) if floor_mean is not None else None,
            "n_single_select": s["n_single"],
            "mean_n_options": round(s["nopt_sum"] / n, 2) if n else None,
            "fallback_share": round(s["paths"].get("trivial_fallback", 0) / n, 4) if n else None,
            "path_distribution": dict(s["paths"].most_common()),
        }

    # headline roll-up
    head = defaultdict(lambda: {"n": 0, "match": 0, "floor_sum": 0.0, "floor_n": 0,
                                "fallback": 0, "sem_res": 0, "sem_match": 0})
    for cat, s in stats.items():
        h = head[HEADLINE.get(cat, "other")]
        h["n"] += s["n"]
        h["match"] += s["match"]
        h["floor_sum"] += s["floor_sum"]
        h["floor_n"] += s["floor_n"]
        h["fallback"] += s["paths"].get("trivial_fallback", 0)
        h["sem_res"] += s["sem_res"]
        h["sem_match"] += s["sem_match"]
    headline = {}
    for hname, h in head.items():
        n = h["n"]
        mr = h["match"] / n if n else 0.0
        fm = (h["floor_sum"] / h["floor_n"]) if h["floor_n"] else None
        sr = (h["sem_match"] / h["sem_res"]) if h["sem_res"] else None
        headline[hname] = {
            "n_decisions": n,
            "exact_match_rate": round(mr, 4),
            "semantic_match_rate": round(sr, 4) if sr is not None else None,
            "n_resolvable": h["sem_res"],
            "chance_floor_mean": round(fm, 4) if fm is not None else None,
            "lift": round(mr - fm, 4) if fm is not None else None,
            "fallback_share": round(h["fallback"] / n, 4) if n else None,
        }

    return {
        "totals": totals,
        "overall_match_rate": round(totals["match"] / totals["decisions"], 4) if totals["decisions"] else None,
        "headline_categories": headline,
        "granular_categories": dict(sorted(cats.items(), key=lambda kv: -kv[1]["n_decisions"])),
        "by_expert_name": {k: {"n": v["n"], "match_rate": round(v["match"] / v["n"], 4) if v["n"] else None}
                           for k, v in by_name.items()},
        "by_expert_won": {("won" if k else "lost"): {"n": v["n"], "match_rate": round(v["match"] / v["n"], 4) if v["n"] else None}
                          for k, v in by_won.items()},
    }


def _print_table(report: dict) -> None:
    t = report["totals"]
    print(f"\n=== OVERALL top-1 fidelity: {report['overall_match_rate']} "
          f"({t['match']}/{t['decisions']} decisions, {t['episodes_kept']} expert-games kept, "
          f"{t['episodes_impure']} impure, {t['agent_exceptions']} agent-exc) ===\n")
    print(f"{'HEADLINE':<22} {'n':>7} {'idx':>6} {'sem':>6} {'semN':>6} {'floor':>6} {'lift':>7} {'fb%':>5}")
    for h, d in sorted(report["headline_categories"].items(), key=lambda kv: -kv[1]["n_decisions"]):
        fl = "" if d["chance_floor_mean"] is None else f"{d['chance_floor_mean']:.3f}"
        lf = "" if d["lift"] is None else f"{d['lift']:+.3f}"
        sm = "" if d["semantic_match_rate"] is None else f"{d['semantic_match_rate']:.3f}"
        print(f"{h:<22} {d['n_decisions']:>7} {d['exact_match_rate']:>6.3f} {sm:>6} "
              f"{d['n_resolvable']:>6} {fl:>6} {lf:>7} {(d['fallback_share'] or 0)*100:>4.1f}")
    print(f"\n{'GRANULAR':<28} {'n':>7} {'idx':>6} {'sem':>6} {'semN':>6} {'floor':>6} {'lift':>7} {'fb%':>5}")
    for c, d in report["granular_categories"].items():
        if d["n_decisions"] < 20:
            continue
        fl = "" if d["chance_floor_mean"] is None else f"{d['chance_floor_mean']:.3f}"
        lf = "" if d["lift"] is None else f"{d['lift']:+.3f}"
        sm = "" if d["semantic_match_rate"] is None else f"{d['semantic_match_rate']:.3f}"
        print(f"{c:<28} {d['n_decisions']:>7} {d['exact_match_rate']:>6.3f} {sm:>6} "
              f"{d['n_resolvable']:>6} {fl:>6} {lf:>7} {(d['fallback_share'] or 0)*100:>4.1f}")
    print("\nby expert:", report["by_expert_name"])
    print("by won/lost:", report["by_expert_won"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", default="v074_fullmetal")
    ap.add_argument("--days", nargs="+", default=["2026-07-20"])
    ap.add_argument("--daily-root", default=str(MAIN_REPO / "data" / "episodes" / "official"))
    ap.add_argument("--ref-decks", default=str(MAIN_REPO / "data" / "deck_scan" / "top10_decks.json"))
    ap.add_argument("--targets", nargs="+", default=["jiatu.l", "bono"])
    ap.add_argument("--min-overlap", type=int, default=54)
    ap.add_argument("--limit", type=int, default=0, help="0 = all dumps; else cap dumps scanned (smoke)")
    ap.add_argument("--out", default=str(WORKTREE / "out" / "v074_fidelity" / "topk_0720.json"))
    ap.add_argument("--rows-out", default=None, help="optional JSONL of per-decision rows for audit")
    args = ap.parse_args()

    ref_decks = json.loads(Path(args.ref_decks).read_text())
    targets = set(args.targets)
    pilot = load_pilot(args.candidate)

    dumps = []
    for day in args.days:
        d = Path(args.daily_root) / day
        if not d.is_dir():
            print(f"[fidelity] SKIP missing day: {d}", file=sys.stderr)
            continue
        files = sorted(d.glob("*.json"))
        if args.limit:
            # smoke: scan only until we have `limit` target-hitting dumps
            picked = []
            for fp in files:
                try:
                    if any(t.encode() in fp.read_bytes() for t in targets):
                        picked.append(fp)
                except Exception:
                    pass
                if len(picked) >= args.limit:
                    break
            dumps.extend(picked)
        else:
            dumps.extend(files)
    print(f"[fidelity] candidate={args.candidate} days={args.days} dumps={len(dumps)} "
          f"targets={sorted(targets)}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_f = open(args.rows_out, "w") if args.rows_out else None

    t0 = time.perf_counter()
    report = evaluate(pilot, dumps, targets, ref_decks, args.min_overlap, rows_f)
    report["wall_sec"] = round(time.perf_counter() - t0, 1)
    report["candidate"] = args.candidate
    report["days"] = args.days
    if rows_f:
        rows_f.close()

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    _print_table(report)
    print(f"\n[fidelity] wrote {out_path}  ({report['wall_sec']}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
