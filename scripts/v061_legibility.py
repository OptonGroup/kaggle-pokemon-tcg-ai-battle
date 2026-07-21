#!/usr/bin/env python3
"""v061_legibility.py — offline "does the deterministic pilot reproduce the expert?" diagnostic.

Forked from scripts/diag_v060_per_expert.py, but instead of scoring a LightGBM ranker over a
MAIN-only feature matrix, it runs the FULL v061 candidate decision pipeline (guards on the raw
obs_dict + the deterministic scorer on the typed obs) against every expert-won decision in
data/il_grimmsnarl_v2/pairs.jsonl — ALL SelectContexts, not just MAIN — and reports, per
decision-taxonomy bucket, how often the candidate's chosen option set overlaps / exactly
matches the expert's actual choice.

Baseline to beat: the v060 BC ranker's MAIN-only top-1 = 0.5410 (results/gates/
v060_per_expert_diag.json). Checkpoint-1 thresholds (see submissions/.../DESIGN.md + plan):
  - "solved" buckets (forced_lethal_attack, adrena_trigger, adrena_target, energy_attach) >= 0.90
  - MAIN-only top-1 >= 0.55
  - combo-wall buckets (boss_target, supporter_choice, punkup_placement) — report only.

Two subtleties this harness handles that a naive "call agent() per pair" would get wrong:
  1. The Adrena-Brain sub-steps (ctx REMOVE_DAMAGE_COUNTER=16 / _COUNT=40 / DAMAGE_COUNTER=13)
     are handled in-game by a stateful sequence guard that only fires after its own MAIN-context
     trigger. On isolated pairs that trigger never ran, so we drive `_adrena_sequence_guard`
     directly with `_adrena_state` forced to the step matching the context.
  2. Module-level guard state (`_adrena_state`, opp-attack tracking) is reset before every pair
     so decisions are evaluated independently.

Output: prints a table + writes results/gates/v061_legibility.json (diffable vs v060 diag).

Run:  .venv/bin/python scripts/v061_legibility.py [--limit N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUB_DIR = REPO / "submissions" / "v061_grimmsnarl_heuristic"
# pairs.jsonl is untracked in git; read it from the primary working copy if the worktree
# doesn't carry it.
_PAIRS_CANDIDATES = [
    REPO / "data" / "il_grimmsnarl_v2" / "pairs.jsonl",
    Path("/Users/max/kaggle-pokemon-tcg-ai-battle/data/il_grimmsnarl_v2/pairs.jsonl"),
]
PAIRS_PATH = next((p for p in _PAIRS_CANDIDATES if p.exists()), _PAIRS_CANDIDATES[0])

EXPERT_NAMES = {"jiatu.l", "bono"}
MAIN_CONTEXT = 0
OUT_PATH = REPO / "results" / "gates" / "v061_legibility.json"
V060_MAIN_TOP1_BASELINE = 0.5410

# Adrena-Brain sub-step contexts -> the _adrena_state value the in-game sequence guard would
# be in when it reaches that decision (see submissions/.../main.py _adrena_sequence_guard).
ADRENA_STEP_CTX = {16: "await_from", 40: "await_count", 13: "await_target"}

# Card taxonomy (kept in sync with the deck; only what classification needs).
SUPPORTERS = {1227, 1219, 1182, 1231, 1197}
ITEMS = {1086, 1152, 1079, 1097, 1080, 1122}
SPIKEMUTH = 1259
MUNKIDORI = 112


def _load_module(path: Path, name: str):
    """Import a .py file as a module, with the submission dir on sys.path first so its
    intra-package imports (cg, scoring_grimmsnarl) resolve exactly as on Kaggle."""
    sub = str(SUB_DIR)
    if sub not in sys.path:
        sys.path.insert(0, sub)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _action_signature(obs, opt, option_card):
    """A coarse 'what action is this' signature so functionally-identical choices count as a
    match even when their option INDEX differs (multiple Munkidori offering the same ability;
    the same energy attachable to the same target from different hand copies; within-turn
    ordering). Strict index-top1 penalizes these; this doesn't."""
    try:
        t = int(opt.type) if opt.type is not None else -1
    except Exception:
        t = -1
    cid = None
    try:
        c = option_card(obs, opt) if option_card else None
        cid = getattr(c, "id", None)
    except Exception:
        pass
    if t == 8:                      # ATTACH -> which card (D energy) matters, not which target
        return ("ATTACH", cid)
    if t == 10:                     # ABILITY -> which Pokémon's ability (e.g. any Munkidori)
        return ("ABILITY", cid)
    if t == 13:                     # ATTACK -> which attack
        return ("ATTACK", getattr(opt, "attackId", None))
    if t == 7:                      # PLAY -> which card
        return ("PLAY", cid)
    if t == 9:                      # EVOLVE -> which card evolved into
        return ("EVOLVE", cid)
    if t in (1, 2, 14, 12):         # YES/NO/END/RETREAT -> the type alone
        return ("T", t)
    if t == 0:                      # NUMBER -> the number value
        return ("NUM", getattr(opt, "number", None))
    return ("T", t)                 # CARD/target selects -> fall back to type (index compared separately)


def functional_match(obs, pick, chosen, option_card):
    """True if ANY of our picked options shares an action signature with ANY expert-chosen
    option. For CARD/target-select contexts (signature ('T', 3) etc.) this reduces to 'same
    action type', so those still effectively rely on the index-overlap metric reported
    alongside."""
    try:
        opts = obs.select.option
        exp_sigs = {_action_signature(obs, opts[i], option_card) for i in chosen if i < len(opts)}
        our_sigs = {_action_signature(obs, opts[i], option_card) for i in pick if i < len(opts)}
        return bool(exp_sigs & our_sigs)
    except Exception:
        return False


def classify(ctx, ctype, chosen_card_id, is_opp_target, main_mod, obs_dict, fired_guard):
    """Assign a decision to a taxonomy bucket. `fired_guard` is the name of the guard that
    produced the candidate pick (or None if the scorer did) — lets the "solved" buckets align
    with the actual mechanism, which is what checkpoint-1 measures."""
    if ctx in ADRENA_STEP_CTX:
        return "adrena_target"
    if ctx in (1, 2, 41, 42):
        return "setup"
    if ctx == 7:
        return "item_search"
    if ctx in (21, 22):
        return "punkup_placement"        # ATTACH_FROM / ATTACH_TO (Punk Up energy placement)
    if ctx == 15:
        return "damage_splash"           # Shadow Bullet +30 bench splash target
    if ctx == 43:
        return "activate"                # Punk Up YES/NO & other ability activations
    if ctx == 37:
        return "evolve_timing"
    if ctx == 34:
        return "skill_order"
    if ctx == 30:
        return "discard_energy"
    if ctx == 33:
        return "switch_energy"
    if ctx == 8:
        return "discard"
    if ctx == 38:
        return "draw_count"
    if ctx in (3, 4, 5, 6):
        return "boss_target" if is_opp_target else "promotion"
    if ctx == MAIN_CONTEXT:
        if fired_guard == "ko_guard":
            return "forced_lethal_attack"
        if fired_guard == "adrena_trigger":
            return "adrena_trigger"
        if fired_guard == "energy_guard":
            return "energy_attach"
        if ctype == 13:                  # ATTACK (non-lethal — ko_guard would have caught lethal)
            return "attack_choice"
        if ctype == 8:                   # ATTACH
            return "energy_attach"
        if ctype == 9:                   # EVOLVE
            return "evolve_timing"
        if ctype == 10:                  # ABILITY (Munkidori Adrena-Brain trigger, non-lethal)
            return "adrena_trigger"
        if ctype == 7:                   # PLAY
            if chosen_card_id in SUPPORTERS:
                return "supporter_choice"
            if chosen_card_id in ITEMS or chosen_card_id == SPIKEMUTH:
                return "item_play"
            return "pokemon_play"
        if ctype == 12:                  # RETREAT
            return "retreat"
        if ctype == 14:                  # END
            return "end_turn"
        return "main_other"
    return "other"


def candidate_pick(main_mod, ctx, obs_dict):
    """Return (pick_indices:list[int], fired_guard:str|None) for one isolated decision.

    Resets per-pair guard state, then either drives the Adrena-Brain sequence guard directly
    (for its sub-step contexts) or runs the full agent() pipeline (guards + scorer) and
    attributes which guard, if any, produced the pick."""
    # reset per-decision state
    main_mod._adrena_state = None
    scorer = getattr(main_mod, "_scorer", None)
    if scorer is not None and hasattr(scorer, "reset_game_state"):
        try:
            scorer.reset_game_state()
        except Exception:
            pass

    if ctx in ADRENA_STEP_CTX:
        main_mod._adrena_state = ADRENA_STEP_CTX[ctx]
        try:
            pick = main_mod._adrena_sequence_guard(obs_dict)
        except Exception:
            pick = None
        main_mod._adrena_state = None
        if pick is not None:
            return list(pick), "adrena_sequence"
        # fall through to full pipeline if the guard declined (desync)
        return list(main_mod.agent(obs_dict) or []), None

    # Full pipeline. Attribute the guard by probing the same order agent() uses.
    fired = None
    try:
        if main_mod._promotion_guard(obs_dict) is not None:
            fired = "promotion"
        elif main_mod._ko_guard(obs_dict) is not None:
            fired = "ko_guard"
        elif main_mod._adrena_trigger_guard(obs_dict) is not None:
            fired = "adrena_trigger"
            main_mod._adrena_state = None  # probing set it; undo so agent() re-fires cleanly
        elif main_mod._energy_guard(obs_dict) is not None:
            fired = "energy_guard"
    except Exception:
        fired = None
    main_mod._adrena_state = None
    try:
        pick = list(main_mod.agent(obs_dict) or [])
    except Exception:
        pick = []
    main_mod._adrena_state = None
    return pick, fired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap pairs scanned (0 = all)")
    args = ap.parse_args()

    if not PAIRS_PATH.exists():
        print(f"FATAL: pairs.jsonl not found at {PAIRS_PATH}", file=sys.stderr)
        sys.exit(2)

    # Import the candidate. main.py itself imports scoring_grimmsnarl as _scorer, so reuse that
    # exact module instance (avoid a second copy with divergent module-level state).
    main_mod = _load_module(SUB_DIR / "main.py", "v061_main")
    scorer = getattr(main_mod, "_scorer", None)
    to_obs = main_mod.to_observation_class
    option_card = getattr(scorer, "option_card", None) if scorer is not None else None

    buckets = defaultdict(lambda: {"n": 0, "top1": 0, "exact": 0, "func": 0})
    main_top1 = main_top1_n = 0
    guard_attr = defaultdict(int)
    n_scanned = n_eval = 0

    with open(PAIRS_PATH) as f:
        for line in f:
            if args.limit and n_scanned >= args.limit:
                break
            n_scanned += 1
            try:
                pair = json.loads(line)
            except Exception:
                continue
            if pair.get("player_name") not in EXPERT_NAMES or not pair.get("won"):
                continue
            ctx = pair["select_context"]
            n_opts = pair["n_options"]
            if n_opts == 0:
                continue
            obs_dict = pair["obs_dict"]
            chosen = {int(i) for i in pair["chosen_option_indices"] if int(i) < n_opts}
            if not chosen:
                continue

            pick, fired = candidate_pick(main_mod, ctx, obs_dict)
            pick = [i for i in pick if 0 <= i < n_opts]
            guard_attr[fired or "scorer"] += 1

            # resolve chosen card id + whether the chosen target is opponent-side (Boss gust)
            chosen_card_id = None
            is_opp_target = False
            ctype = None
            func = False
            try:
                obs = to_obs(obs_dict)
                opts = obs.select.option
                first = next(iter(chosen))
                copt = opts[first]
                ctype = int(copt.type) if copt.type is not None else None
                if option_card is not None:
                    c = option_card(obs, copt)
                    chosen_card_id = getattr(c, "id", None)
                me = obs.current.yourIndex
                pi = getattr(copt, "playerIndex", None)
                is_opp_target = pi is not None and int(pi) != int(me)
                func = functional_match(obs, pick, chosen, option_card)
            except Exception:
                pass

            bucket = classify(ctx, ctype, chosen_card_id, is_opp_target, main_mod, obs_dict, fired)
            top1 = bool(set(pick) & chosen)          # our pick overlaps the expert's choice
            exact = set(pick) == chosen
            b = buckets[bucket]
            b["n"] += 1
            b["top1"] += int(top1)
            b["exact"] += int(exact)
            b["func"] += int(func)
            if ctx == MAIN_CONTEXT:
                main_top1_n += 1
                main_top1 += int(top1)
            n_eval += 1

    # ---- report ----
    def rate(a, b):
        return (a / b) if b else 0.0

    SOLVED = {"forced_lethal_attack", "adrena_trigger", "adrena_target", "energy_attach"}
    COMBO_WALL = {"boss_target", "supporter_choice", "punkup_placement"}

    rows = sorted(buckets.items(), key=lambda kv: -kv[1]["n"])
    print(f"\npairs scanned={n_scanned}  evaluated={n_eval}")
    print(f"MAIN-only top-1 = {rate(main_top1, main_top1_n):.4f}  (n={main_top1_n}; "
          f"v060 BC ranker baseline={V060_MAIN_TOP1_BASELINE})")
    print(f"\n{'bucket':22s} {'n':>7s} {'top1':>8s} {'func':>8s} {'exact':>8s}  flag")
    out_buckets = {}
    for name, b in rows:
        t1 = rate(b["top1"], b["n"])
        fn = rate(b["func"], b["n"])
        ex = rate(b["exact"], b["n"])
        flag = ""
        if name in SOLVED:
            # judge the "solved" threshold on the FUNCTIONAL match (same action, ignoring
            # benign index/source/ordering differences), which is what "did the pilot do the
            # right thing" actually means for a deterministic agent.
            if fn < 0.75:
                flag = "SOLVED<0.75 REVISIT"
            elif fn < 0.90:
                flag = "SOLVED<0.90 revisit"
            else:
                flag = "solved-ok(func)"
        elif name in COMBO_WALL:
            flag = "combo-wall(report)"
        print(f"{name:22s} {b['n']:7d} {t1:8.4f} {fn:8.4f} {ex:8.4f}  {flag}")
        out_buckets[name] = {"n": b["n"], "top1": t1, "func": fn, "exact": ex,
                             "top1_count": b["top1"], "func_count": b["func"],
                             "exact_count": b["exact"]}

    print(f"\nguard attribution: {dict(guard_attr)}")

    # checkpoint-1 verdict (solved buckets judged on FUNCTIONAL match — same action, ignoring
    # index/source/ordering artifacts; MAIN top-1 on strict index vs the BC-ranker baseline).
    solved_min_func = min((out_buckets[n]["func"] for n in SOLVED if n in out_buckets), default=0.0)
    main_t1 = rate(main_top1, main_top1_n)
    verdict = {
        "main_top1": main_t1,
        "main_top1_pass": main_t1 >= 0.55,
        "solved_buckets_min_func": solved_min_func,
        "solved_buckets_pass": all(out_buckets.get(n, {}).get("func", 0) >= 0.90 for n in SOLVED
                                   if n in out_buckets),
    }
    print(f"\nCHECKPOINT-1: MAIN top-1 {main_t1:.4f} (pass>=0.55: {verdict['main_top1_pass']}); "
          f"solved buckets min FUNC {solved_min_func:.4f} (pass>=0.90: {verdict['solved_buckets_pass']})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "pairs_scanned": n_scanned,
            "evaluated": n_eval,
            "main_top1": main_t1,
            "main_top1_n": main_top1_n,
            "v060_bc_ranker_main_top1_baseline": V060_MAIN_TOP1_BASELINE,
            "buckets": out_buckets,
            "guard_attribution": dict(guard_attr),
            "checkpoint1": verdict,
        }, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
