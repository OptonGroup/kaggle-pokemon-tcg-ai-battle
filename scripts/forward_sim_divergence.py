#!/usr/bin/env python3
"""forward_sim_divergence.py — resolve the sequencing confound left open by
scripts/fidelity_expert_topk.py (docs/gold/BREAK_CEILING_900.md §4-РЕЗУЛЬТАТ).

The pointwise fidelity harness fed the pilot the EXPERT's continuation, so at an
attack/promotion divergence it could not tell "pilot forgoes the KO forever" from
"pilot follows its own documented develop-first-attack-last design and takes the KO
LATER in its own turn" (auditor C left this as a 25%-75% range). This script closes
that gap: at each divergence it hands the pilot the SAME board via the engine search
API (search_begin from the obs's own search_begin_input) and lets the PILOT DRIVE ITS
OWN TURN FORWARD (search_step with the pilot's choices) until the pilot's turn ends —
then reports whether the pilot eventually took an attack / the KO on its own, or ended
its turn without it (a genuine, non-sequencing miss).

Engine bridge (verified): the search state's typed Observation is converted back to the
raw obs_dict the pilot consumes via deenum(dataclasses.asdict(obs)); on a static obs the
pilot answers identically to the raw-dict path. Native macOS engine (cg_native), single
process, no threads (SIGSEGV risk noted in project_native_engine).

Hidden-info for search_begin: within OUR OWN turn the opponent never acts, so opponent
deck/hand/prize contents do not affect our decisions; they are filled with valid
placeholders of the correct COUNT (the engine validates counts, not contents, and never
draws them during our turn). Our own deck is reconstructed as (full 60-card deck −
visible), padded with basic {D} energy. Faithfulness is gated per rollout: the pilot's
FIRST decision inside the rollout MUST equal its pointwise decision on the original obs
(same board) — rollouts that fail this gate are counted separately, not scored.

Usage:
    .venv/bin/python scripts/forward_sim_divergence.py --limit-eps 60 --max-div 200
    .venv/bin/python scripts/forward_sim_divergence.py --smoke   # 5 cases, verbose
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib.util
import io
import contextlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
MAIN_REPO = Path("/Users/max/kaggle-pokemon-tcg-ai-battle")
SRC = WORKTREE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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

from cg.api import to_observation_class, search_begin, search_step, search_release  # noqa: E402
from replays.parse import parse_replay  # noqa: E402

OPT_ATTACK = 13
OPT_END = 14
CTX_MAIN = 0
CTX_SWITCH, CTX_TO_ACTIVE = 3, 4
BASIC_D_ENERGY = 7
IMPIDIMP = 646  # a Basic Pokemon, valid placeholder for opp_deck's "needs a basic" rule
ROLLOUT_STEP_CAP = 60


def _deenum(x):
    if isinstance(x, enum.Enum):
        return int(x.value) if isinstance(x.value, int) else x.value
    if isinstance(x, dict):
        return {k: _deenum(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_deenum(v) for v in x]
    return x


def bridge(typed_obs) -> dict:
    """Typed search Observation -> raw obs_dict the pilot consumes."""
    return _deenum(dataclasses.asdict(typed_obs))


def load_pilot(sub_name: str):
    sub_dir = (WORKTREE / "submissions" / sub_name).resolve()
    main_path = sub_dir / "main.py"
    if str(sub_dir) not in sys.path:
        sys.path.insert(0, str(sub_dir))
    old = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(f"_pilot_{sub_name}", str(main_path))
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = str(main_path)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(old)
    return mod, list(mod.my_deck)


def _agent_quiet(pilot, obs_dict):
    with contextlib.redirect_stderr(io.StringIO()):
        return pilot.agent(obs_dict)


def _ids(lst):
    return [int(c["id"]) for c in (lst or []) if isinstance(c, dict) and c.get("id") is not None]


def _board_ids(pl):
    out = []
    for a in ("active", "bench"):
        out += _ids(pl.get(a))
    return out


def reconstruct_hidden(obs_dict: dict, my_full_deck: list[int]):
    """Best-effort predicted hidden card lists for search_begin, correct COUNTS."""
    cur = obs_dict["current"]
    me = int(cur["yourIndex"])
    opp = 1 - me
    mp = cur["players"][me]
    op = cur["players"][opp]

    visible_mine = _ids(mp.get("hand")) + _board_ids(mp) + _ids(mp.get("discard"))
    remain = list((Counter(my_full_deck) - Counter(visible_mine)).elements())

    deck_count = int(mp.get("deckCount") or 0)
    my_prize_n = len(mp.get("prize") or [])
    pool = list(remain)
    while len(pool) < deck_count + my_prize_n:
        pool.append(BASIC_D_ENERGY)
    your_prize = pool[:my_prize_n]
    your_deck = pool[my_prize_n:my_prize_n + deck_count]
    while len(your_deck) < deck_count:
        your_deck.append(BASIC_D_ENERGY)

    opp_deck = [IMPIDIMP] * int(op.get("deckCount") or 0)
    opp_prize = [IMPIDIMP] * len(op.get("prize") or [])
    opp_hand = [IMPIDIMP] * int(op.get("handCount") or 0)
    opp_active_l = op.get("active") or []
    opp_active = []
    if opp_active_l and (not isinstance(opp_active_l[0], dict) or opp_active_l[0] is None):
        opp_active = [IMPIDIMP]
    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active


def _opp_active_hp(obs_dict):
    cur = obs_dict.get("current") or {}
    me = int(cur.get("yourIndex") or 0)
    op = (cur.get("players") or [{}, {}])[1 - me]
    act = op.get("active") or []
    if act and isinstance(act[0], dict):
        return act[0].get("hp"), act[0].get("serial")
    return None, None


def _my_prize_n(obs_dict):
    cur = obs_dict.get("current") or {}
    me = int(cur.get("yourIndex") or 0)
    return len((cur.get("players") or [{}, {}])[me].get("prize") or [])


def rollout(pilot, obs_dict: dict, my_full_deck: list[int]) -> dict:
    """Drive the pilot's OWN turn forward from obs_dict. Returns what it did."""
    typed = to_observation_class(obs_dict)
    yd, yp, od, op_, oh, oa = reconstruct_hidden(obs_dict, my_full_deck)
    our_index = int(obs_dict["current"]["yourIndex"])
    try:
        state = search_begin(typed, yd, yp, od, op_, oh, oa)
    except Exception as e:
        return {"status": "search_begin_failed", "error": f"{type(e).__name__}: {e}"}

    sid = state.searchId
    result = {"status": "ok", "attacked": False, "took_ko": False, "ended_turn": False,
              "first_choice_ok": None, "steps": 0, "first_choice": None}
    start_prize = _my_prize_n(obs_dict)
    try:
        first = True
        for _ in range(ROLLOUT_STEP_CAP):
            obs_t = state.observation
            cur = obs_t.current
            if cur is not None and getattr(cur, "result", -1) != -1:
                break
            if cur is None or getattr(cur, "yourIndex", our_index) != our_index:
                result["ended_turn"] = True  # turn passed to opponent
                break
            sel = obs_t.select
            if sel is None:
                break
            od_dict = bridge(obs_t)
            n_opt = len(od_dict.get("select", {}).get("option") or [])
            if n_opt == 0:
                break
            choice = _agent_quiet(pilot, od_dict)
            if first:
                result["first_choice"] = list(choice) if choice else choice
                first = False
            # classify the chosen option(s)
            opts = od_dict["select"]["option"]
            ctx = od_dict["select"].get("context")
            chose_attack = any(isinstance(i, int) and 0 <= i < len(opts)
                               and opts[i].get("type") == OPT_ATTACK for i in (choice or []))
            chose_end = any(isinstance(i, int) and 0 <= i < len(opts)
                            and opts[i].get("type") == OPT_END for i in (choice or []))
            hp_before, serial_before = _opp_active_hp(od_dict)
            try:
                state = search_step(sid, choice)
            except Exception as e:
                result["status"] = f"search_step_failed: {type(e).__name__}: {e}"
                break
            result["steps"] += 1
            if chose_attack:
                result["attacked"] = True
                # KO check: opp active HP hit 0 / active replaced / we gained a prize
                nb = bridge(state.observation) if state.observation is not None else {}
                hp_after, serial_after = _opp_active_hp(nb)
                took_prize = _my_prize_n(nb) < start_prize if nb else False
                if took_prize or (hp_after is not None and hp_after <= 0) or \
                   (serial_before is not None and serial_after is not None and serial_after != serial_before):
                    result["took_ko"] = True
            if chose_end:
                result["ended_turn"] = True
                break
    finally:
        try:
            search_release(sid)
        except Exception:
            pass
    return result


def iter_divergences(pilot, episodes, targets, ref_decks, min_overlap, max_div, want_cats):
    """Yield (obs_dict, expert_action, category, meta) for attack/promotion divergences."""
    def multiset_overlap(a, b):
        return sum((Counter(a) & Counter(b)).values())

    def true_decks(data):
        steps = data.get("steps", [])
        if len(steps) < 2:
            return {}
        out = {}
        for pi in range(min(2, len(steps[1]))):
            a = steps[1][pi].get("action", [])
            if len(a) == 60 and all(isinstance(x, int) for x in a):
                out[pi] = a
        return out

    n = 0
    for fp in episodes:
        try:
            raw = fp.read_bytes()
        except Exception:
            continue
        if not any(t.encode() in raw for t in targets):
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        names = [a.get("Name") for a in data.get("info", {}).get("Agents", [])]
        td = true_decks(data)
        experts = []
        for pi, nm in enumerate(names):
            if nm not in targets or nm not in ref_decks:
                continue
            d = td.get(pi)
            if d is None or multiset_overlap(d, ref_decks[nm]) < min_overlap:
                continue
            experts.append(pi)
        if not experts:
            continue
        try:
            pairs = parse_replay(fp)
        except Exception:
            continue
        for pi in experts:
            for p in pairs:
                if p["player_index"] != pi:
                    continue
                obs = p["obs_dict"]
                sel = obs.get("select") or {}
                ctx = sel.get("context")
                opts = sel.get("option") or []
                expert = list(p["chosen_option_indices"])
                ei = expert[0] if expert else None
                exp_type = opts[ei].get("type") if (isinstance(ei, int) and 0 <= ei < len(opts)) else None
                # category of interest
                cat = None
                if ctx == CTX_MAIN and exp_type == OPT_ATTACK:
                    cat = "attack"
                elif ctx in (CTX_SWITCH, CTX_TO_ACTIVE):
                    cat = "promotion/switch"
                if cat is None or cat not in want_cats:
                    continue
                # Reset pilot per decision so each divergence + its rollout is a clean,
                # reproducible unit (attack/promotion choices are board-driven; sticky
                # in-game state is re-derived from the board). The rollout then continues
                # from THIS post-decision state, so its first step reproduces `our`.
                _agent_quiet(pilot, {"select": None})
                our = _agent_quiet(pilot, obs)
                if our is not None and set(our) == set(expert):
                    continue  # agreed -> not a divergence
                yield obs, expert, cat, {"episodeId": p.get("episodeId"), "turn": obs.get("current", {}).get("turn"),
                                         "name": names[pi], "won": p.get("won"), "our_pointwise": list(our) if our else our}
                n += 1
                if n >= max_div:
                    return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", default="v074_fullmetal")
    ap.add_argument("--day", default="2026-07-20")
    ap.add_argument("--daily-root", default=str(MAIN_REPO / "data" / "episodes" / "official"))
    ap.add_argument("--ref-decks", default=str(MAIN_REPO / "data" / "deck_scan" / "top10_decks.json"))
    ap.add_argument("--targets", nargs="+", default=["jiatu.l", "bono"])
    ap.add_argument("--min-overlap", type=int, default=54)
    ap.add_argument("--limit-eps", type=int, default=80, help="cap episodes scanned")
    ap.add_argument("--max-div", type=int, default=200, help="cap divergences rolled out")
    ap.add_argument("--cats", nargs="+", default=["attack", "promotion/switch"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(WORKTREE / "out" / "v074_fidelity" / "forward_sim.json"))
    args = ap.parse_args()

    if args.smoke:
        args.limit_eps = 20
        args.max_div = 5

    ref_decks = json.loads(Path(args.ref_decks).read_text())
    targets = set(args.targets)
    pilot, my_deck = load_pilot(args.candidate)

    day_dir = Path(args.daily_root) / args.day
    episodes = sorted(day_dir.glob("*.json"))
    # keep only enough target-hitting episodes
    picked = []
    for fp in episodes:
        try:
            if any(t.encode() in fp.read_bytes() for t in targets):
                picked.append(fp)
        except Exception:
            pass
        if len(picked) >= args.limit_eps:
            break

    print(f"[fwdsim] candidate={args.candidate} episodes={len(picked)} cats={args.cats} "
          f"max_div={args.max_div}", file=sys.stderr)

    tally = {c: {"n": 0, "attacked": 0, "took_ko": 0, "ended_no_attack": 0,
                 "gate_fail": 0, "sim_fail": 0, "first_ok": 0} for c in args.cats}
    samples = []
    t0 = time.perf_counter()
    for obs, expert, cat, meta in iter_divergences(pilot, picked, targets, ref_decks,
                                                   args.min_overlap, args.max_div, set(args.cats)):
        r = rollout(pilot, obs, my_deck)
        T = tally[cat]
        T["n"] += 1
        if r["status"] != "ok":
            T["sim_fail"] += 1
            if args.smoke:
                print(f"  [{cat}] SIM_FAIL {r['status']}", file=sys.stderr)
            continue
        # gate: first rollout decision must equal the pointwise pilot decision on the same board
        first_ok = (r.get("first_choice") is not None and meta.get("our_pointwise") is not None
                    and set(r["first_choice"]) == set(meta["our_pointwise"]))
        r["first_choice_ok"] = first_ok
        if first_ok:
            T["first_ok"] += 1
        else:
            T["gate_fail"] += 1
        if r["attacked"] if cat == "attack" else r["attacked"]:
            T["attacked"] += 1
        if r["took_ko"]:
            T["took_ko"] += 1
        if not r["attacked"] and r["ended_turn"]:
            T["ended_no_attack"] += 1
        if args.smoke or len(samples) < 40:
            samples.append({"cat": cat, **meta, "expert": expert,
                            "rollout": {k: r[k] for k in ("attacked", "took_ko", "ended_turn",
                                                           "steps", "first_choice", "first_choice_ok")}})
        if args.smoke:
            print(f"  [{cat}] ep{meta['episodeId']} t{meta['turn']} exp={expert} "
                  f"first={r['first_choice']}(ok={first_ok}) attacked={r['attacked']} "
                  f"ko={r['took_ko']} ended={r['ended_turn']} steps={r['steps']}", file=sys.stderr)

    report = {"candidate": args.candidate, "day": args.day, "cats": args.cats,
              "wall_sec": round(time.perf_counter() - t0, 1), "tally": {}, "samples": samples}
    for c, T in tally.items():
        n = T["n"]
        scored = T["first_ok"]  # only gate-passing rollouts are trustworthy
        report["tally"][c] = {
            **T,
            "attack_rate_over_gatepass": round(T["attacked"] / scored, 4) if scored else None,
            "ko_rate_over_gatepass": round(T["took_ko"] / scored, 4) if scored else None,
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n=== FORWARD-SIM: does the pilot take the action LATER in its own turn? ===")
    for c, T in report["tally"].items():
        print(f"\n[{c}] divergences={T['n']} gate_pass={T['first_ok']} gate_fail={T['gate_fail']} "
              f"sim_fail={T['sim_fail']}")
        if T["first_ok"]:
            print(f"   of gate-passing: attacked-later={T['attacked']} "
                  f"({T['attack_rate_over_gatepass']:.1%}) | took-KO={T['took_ko']} "
                  f"({T['ko_rate_over_gatepass']:.1%}) | ended-turn-without-attack={T['ended_no_attack']}")
    print(f"\n[fwdsim] wrote {args.out}  ({report['wall_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
