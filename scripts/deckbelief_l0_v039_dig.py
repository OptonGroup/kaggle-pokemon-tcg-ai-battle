"""deckbelief_l0_v039_dig.py — PLAN.md §L0 follow-up: v047_deckbelief's regression vs v039_ft
(-4.8pp in the N=1000 pool gate) can't be explained by belief-net misclassification
(scripts/deckbelief_l0_diagnose.py found it fires wrong only 1.6%/0% of win/loss games for this
opponent -- v039_ft's own deck genuinely contains the Crustle line, so the board scan almost always
finds it directly and the belief net barely ever gets a turn).

The ONLY other behavioral difference between v047 and v044 is that detect_matchup()'s label
*persists* once board evidence goes quiet (_last_matchup survives instead of reverting to "generic"
the instant the identifying Pokemon leaves the active+bench). v039_ft's deck is Crustle +
Mega Kangaskhan ex (see submissions/v039_ft/deck.csv / project memory) -- the Crustle line can leave
the board (KO'd, or just not yet evolved) while Mega Kangaskhan is the active threat, a case where
v044 would say "generic" (safe default) but v047 keeps saying "crustle" (stale, PERSISTED -- not
misclassified, just outdated) and applies crustle-specific opp_max_damage/score_play/score_target
branches to a board that's no longer actually running Crustle mechanics that turn.

This script monkeypatches the already-imported v047_deckbelief module's detect_matchup (no edits to
the submission source) to record, at EVERY call (not just on label change), whether the returned
label came from live board evidence or from persistence, split by final win/loss -- the same
forensic-loss-trap discipline as the sibling script.

Usage:
    .venv/bin/python scripts/deckbelief_l0_v039_dig.py --games 300 --workers 8
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

MAX_MOVES_PER_GAME = 2000
CANDIDATE = "v047_deckbelief"
OPPONENT = "v039_ft"


def _load_submission_module(sub_dir_name: str, tag: str):
    sub_dir = os.path.join(ROOT, "submissions", sub_dir_name)
    main_path = os.path.join(sub_dir, "main.py")
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    original_cwd = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(f"_l0_v039dig_{sub_dir_name}_{tag}", main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)
    return mod


def _load_deck_csv(path: str) -> list[int]:
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip()]


def _play_one_game(task: dict) -> dict:
    from cg_native.game import battle_start, battle_finish, battle_select
    from cg_native.api import to_observation_class
    import cg_native, cg_native.api, cg_native.game, cg_native.sim, cg_native.utils
    sys.modules.setdefault("cg", cg_native)
    sys.modules.setdefault("cg.api", cg_native.api)
    sys.modules.setdefault("cg.game", cg_native.game)
    sys.modules.setdefault("cg.sim", cg_native.sim)
    sys.modules.setdefault("cg.utils", cg_native.utils)

    player_sub = task["player_sub"]
    opp_sub = task["opp_sub"]
    our_first = task["our_first"]
    seed = task["seed"]
    random.seed(seed)

    player_mod = _load_submission_module(player_sub, "player")
    opp_mod = _load_submission_module(opp_sub, "opp")

    # Instrument player_mod.detect_matchup in-place: every call, compare the persisted return value
    # against a fresh board-only read (the exact v044 behavior for that same obs) without touching
    # the submission source on disk.
    calls = {"total": 0, "persisted": 0, "persisted_crustle_over_generic": 0}
    orig_detect = player_mod.detect_matchup

    @functools.wraps(orig_detect)
    def traced_detect(obs):
        board_label = player_mod._detect_matchup_board_only(obs)
        result = orig_detect(obs)
        calls["total"] += 1
        if result != board_label:
            calls["persisted"] += 1
            if board_label == "generic" and result == "crustle":
                calls["persisted_crustle_over_generic"] += 1
        return result

    player_mod.detect_matchup = traced_detect

    player_deck = list(getattr(player_mod, "my_deck", None) or _load_deck_csv(
        os.path.join(ROOT, "submissions", player_sub, "deck.csv")))
    opp_deck = list(getattr(opp_mod, "my_deck", None) or _load_deck_csv(
        os.path.join(ROOT, "submissions", opp_sub, "deck.csv")))

    if our_first:
        deck0, deck1 = player_deck, opp_deck
        agent0, agent1 = player_mod.agent, opp_mod.agent
    else:
        deck0, deck1 = opp_deck, player_deck
        agent0, agent1 = opp_mod.agent, player_mod.agent

    obs_dict, start = battle_start(list(deck0), list(deck1))
    if obs_dict is None:
        return dict(outcome="crash", moves=0, **calls)

    move_count = 0
    winner = None
    n_exc = 0
    try:
        while move_count < MAX_MOVES_PER_GAME:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                winner = obs.current.result
                break
            your_index = obs.current.yourIndex if obs.current else 0
            agent_fn = agent0 if your_index == 0 else agent1
            try:
                choice = agent_fn(obs_dict)
            except Exception:
                n_exc += 1
                break
            obs_dict = battle_select(choice)
            move_count += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass

    our_side = 0 if our_first else 1
    if n_exc > 0 or winner is None:
        outcome = "aborted"
    elif winner == our_side:
        outcome = "our_win"
    elif winner == 2:
        outcome = "draw"
    else:
        outcome = "opp_win"

    return dict(outcome=outcome, moves=move_count, **calls)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260720)
    args = ap.parse_args()

    queue = []
    for i in range(args.games):
        queue.append(dict(player_sub=CANDIDATE, opp_sub=OPPONENT,
                           our_first=(i % 2 == 0),
                           seed=args.seed + (hash(("v039dig", i)) % 1_000_000)))
    random.Random(args.seed).shuffle(queue)

    ctx = multiprocessing.get_context("spawn")
    print(f"[gate] {len(queue)} games, workers={args.workers}", flush=True)
    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = {ex.submit(_play_one_game, t): t for t in queue}
        n_done = 0
        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for f in done:
                futures.pop(f)
                try:
                    res = f.result()
                except Exception:
                    res = dict(outcome="crash", moves=0, total=0, persisted=0,
                               persisted_crustle_over_generic=0)
                results.append(res)
                n_done += 1
                if n_done % 40 == 0:
                    print(f"  ... {n_done}/{len(queue)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== v039_ft persistence-dig report ({time.time()-t0:.0f}s) ===\n")
    wins = [r for r in results if r["outcome"] == "our_win"]
    losses = [r for r in results if r["outcome"] == "opp_win"]
    n_win, n_loss = len(wins), len(losses)
    print(f"W={n_win} L={n_loss} D={sum(1 for r in results if r['outcome']=='draw')} "
          f"aborted={sum(1 for r in results if r['outcome'] in ('aborted','crash'))} "
          f"WR={n_win/(n_win+n_loss):.3f}" if (n_win + n_loss) else "no completed games")

    def frac(rows, key, denom_key="total"):
        tot_num = sum(r[key] for r in rows)
        tot_den = sum(r[denom_key] for r in rows)
        return tot_num / tot_den if tot_den else float("nan")

    def per_game_mean(rows, key, denom_key="total"):
        vals = [r[key] / r[denom_key] for r in rows if r[denom_key] > 0]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"\ndetect_matchup() calls per game: win mean={sum(r['total'] for r in wins)/n_win:.1f} "
          f"loss mean={sum(r['total'] for r in losses)/n_loss:.1f}" if n_win and n_loss else "")

    print(f"\n'persisted' fraction (returned label != fresh board-only read, i.e. v047 diverges "
          f"from what v044 would say RIGHT NOW at this exact decision):")
    print(f"  pooled:   win={frac(wins,'persisted'):.3f}  loss={frac(losses,'persisted'):.3f}")
    print(f"  per-game: win={per_game_mean(wins,'persisted'):.3f}  "
          f"loss={per_game_mean(losses,'persisted'):.3f}")

    print(f"\n'persisted crustle over live-generic' fraction (the specific case: v047 says "
          f"'crustle' -- and fires the crustle-specific opp_max_damage=120/score branches -- "
          f"while the board RIGHT NOW shows no Crustle-line Pokemon, i.e. v044 would say "
          f"'generic' and use its default logic instead):")
    print(f"  pooled:   win={frac(wins,'persisted_crustle_over_generic'):.3f}  "
          f"loss={frac(losses,'persisted_crustle_over_generic'):.3f}")
    print(f"  per-game: win={per_game_mean(wins,'persisted_crustle_over_generic'):.3f}  "
          f"loss={per_game_mean(losses,'persisted_crustle_over_generic'):.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
