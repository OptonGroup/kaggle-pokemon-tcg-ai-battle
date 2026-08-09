"""deckbelief_l0_diagnose.py — PLAN.md §L0: diagnose the N=1000-gate regression of
v047_deckbelief vs v044_archaludon_abomasnow (WR=0.447, see results/gates/v047_full_pool_gate_1000.log).

Mechanism under test: detect_matchup()'s board-only pass (_detect_matchup_board_only, byte-identical
to v044) covers only 8 LINE signatures (crustle/starmie/lucario/hop/grimmsnarl/dipplin/abomasnow/
alakazam). Neither archaludon_mirror ({169,190,666}) nor dragapult ({119,120,121}) is one of them, so
against an opponent running either archetype the board scan returns "generic" for the whole game and
control passes to the belief net (_classify_belief) once >=8 opponent cards are revealed. The belief
net's own prototype set has 9 entries (the 8 above + "archaludon"), but NO downstream consumer
(opp_max_damage/score_play/score_target) has an explicit "archaludon" branch -- so even a *correct*
"archaludon" label behaves exactly like "generic" downstream. If the belief net instead mislabels an
archaludon_mirror or dragapult opponent as one of the 8 branches that DO have specific downstream logic
(most plausibly "crustle" -- see the false-positive documented inline in main.py's _classify_belief,
where a zero-evidence board produced "crustle" at similarity 0.44-0.67), v047 starts applying
crustle-specific (or whichever-specific) damage ceilings / scoring to an opponent that isn't running
that deck at all -- actively worse than the "generic" default v044 always used for these matchups.

This script plays v047_deckbelief against the 3 proven-regression opponents from the N=1000 gate
(v036_archaludon_patched, v039_ft, v046_judgefix), captures every "OPPONENT DECK: ... (source=...)"
debug line _track_and_log_matchup already emits unconditionally (_DEBUG_LOG=True in
submissions/v047_deckbelief/main.py), and cross-checks belief-net labels against each opponent's
*true* archetype (known exactly -- self-play, not a live replay -- from a LINE-set scan of the
opponent's own deck.csv). Reports misclassification rate split win vs loss (forensic-loss-trap
discipline: a signal only counts if it differs between the two groups).

Usage:
    .venv/bin/python scripts/deckbelief_l0_diagnose.py --games 200 --workers 4
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import multiprocessing
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

MAX_MOVES_PER_GAME = 2000
CANDIDATE_DEFAULT = "v047_deckbelief"
OPPONENTS = ["v036_archaludon_patched", "v039_ft", "v046_judgefix"]

# Same LINE sets as submissions/v047_deckbelief/main.py (8 board branches) plus the two candidate
# L2 additions -- used here ONLY to label the opponent's *true* archetype from its deck.csv, not to
# change any agent behavior.
TRUE_LINES = {
    "crustle": {344, 345, 532},
    "starmie": {1030, 1031},
    "lucario": {677, 678},
    "hop": {288, 289, 299, 304, 307, 308, 309, 310, 878, 879},
    "grimmsnarl": {646, 647, 648},
    "dipplin": {89, 90, 92, 93, 247},
    "abomasnow": {722, 723},
    "alakazam": {741, 742, 743},
    "archaludon_mirror": {169, 190, 666},
    "dragapult": {119, 120, 121},
}

# Labels that actually change behavior downstream (opp_max_damage/score_play/score_target all
# branch on exactly these 8 -- see grep 'matchup ==' in submissions/v047_deckbelief/main.py). The
# belief net's 9th prototype, "archaludon", has NO consumer anywhere -- a mislabel to "archaludon"
# is behaviorally identical to staying "generic", so it's tracked separately from the 8 that can
# actually misfire matchup-specific scoring/damage-ceiling logic onto the wrong opponent.
ACTIONABLE_LABELS = {"alakazam", "crustle", "hop", "lucario", "starmie", "grimmsnarl", "dipplin",
                      "abomasnow"}

DBG_RE = re.compile(
    r"\[dbg\] turn=(?P<turn>\d+) OPPONENT DECK: \S+ -> (?P<label>\S+) "
    r"\(source=(?P<source>[a-z-]+)"
)


def _load_submission_module(sub_dir_name: str, tag: str):
    sub_dir = os.path.join(ROOT, "submissions", sub_dir_name)
    main_path = os.path.join(sub_dir, "main.py")
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    original_cwd = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(f"_l0_diag_{sub_dir_name}_{tag}", main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)
    return mod


def _load_deck_csv(path: str) -> list[int]:
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip()]


def _true_archetype(deck_ids: set[int]) -> str:
    hits = [name for name, line in TRUE_LINES.items() if deck_ids & line]
    return hits[0] if len(hits) == 1 else ("ambiguous:" + "+".join(hits) if hits else "generic")


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

    buf = io.StringIO()
    obs_dict = None
    move_count = 0
    winner = None
    n_exc = 0
    with contextlib.redirect_stderr(buf):
        obs_dict, start = battle_start(list(deck0), list(deck1))
        if obs_dict is None:
            return dict(cell=task["cell"], outcome="crash", moves=0, dbg="")
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

    dbg_text = "\n".join(
        line for line in buf.getvalue().splitlines() if "OPPONENT DECK" in line
    )
    return dict(cell=task["cell"], outcome=outcome, moves=move_count, dbg=dbg_text,
                opp_sub=opp_sub, our_first=our_first)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--opponents", type=str, default=",".join(OPPONENTS))
    ap.add_argument("--candidate", type=str, default=CANDIDATE_DEFAULT)
    args = ap.parse_args()

    CANDIDATE = args.candidate
    opponents = [o for o in args.opponents.split(",") if o]

    # True archetype per opponent submission is fixed (static deck.csv) -- compute once.
    true_archetype = {}
    for opp in opponents:
        ids = set(_load_deck_csv(os.path.join(ROOT, "submissions", opp, "deck.csv")))
        true_archetype[opp] = _true_archetype(ids)
        print(f"[setup] {opp}: true archetype = {true_archetype[opp]}", flush=True)

    queue = []
    for opp in opponents:
        for i in range(args.games):
            queue.append(dict(cell=opp, player_sub=CANDIDATE, opp_sub=opp,
                               our_first=(i % 2 == 0),
                               seed=args.seed + (hash((opp, i)) % 1_000_000)))
    random.Random(args.seed).shuffle(queue)

    ctx = multiprocessing.get_context("spawn")
    print(f"[gate] {len(queue)} games total, workers={args.workers}", flush=True)
    t0 = time.time()
    results: dict[str, list[dict]] = {o: [] for o in opponents}

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = {ex.submit(_play_one_game, t): t for t in queue}
        n_done = 0
        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for f in done:
                t = futures.pop(f)
                try:
                    res = f.result()
                except Exception as e:
                    res = dict(cell=t["cell"], outcome="crash", moves=0, dbg="",
                               opp_sub=t["opp_sub"], our_first=t["our_first"])
                results[res["cell"]].append(res)
                n_done += 1
                if n_done % 40 == 0:
                    print(f"  ... {n_done}/{len(queue)} done ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== L0 misclassification report ({time.time()-t0:.0f}s total) ===\n")
    for opp in opponents:
        rows = results[opp]
        truth = true_archetype[opp]
        wins = sum(1 for r in rows if r["outcome"] == "our_win")
        losses = sum(1 for r in rows if r["outcome"] == "opp_win")
        n = wins + losses
        print(f"--- {opp} (true={truth}): W={wins} L={losses} D="
              f"{sum(1 for r in rows if r['outcome']=='draw')} "
              f"aborted={sum(1 for r in rows if r['outcome'] in ('aborted','crash'))} "
              f"WR={wins/n:.3f}" if n else f"--- {opp}: no completed games")

        # Per-game: did belief-net ever fire, and if so, was any fired label != truth?
        belief_fired = {"our_win": 0, "opp_win": 0}
        belief_wrong = {"our_win": 0, "opp_win": 0}
        belief_actionable_wrong = {"our_win": 0, "opp_win": 0}
        label_counts = Counter()
        wrong_label_counts = Counter()
        actionable_wrong_label_counts = Counter()
        for r in rows:
            if r["outcome"] not in ("our_win", "opp_win"):
                continue
            bucket = r["outcome"]
            fired_this_game = False
            wrong_this_game = False
            actionable_wrong_this_game = False
            for line in r["dbg"].splitlines():
                m = DBG_RE.search(line)
                if not m or m.group("source") != "belief-net":
                    continue
                fired_this_game = True
                label = m.group("label")
                label_counts[label] += 1
                if label != truth:
                    wrong_this_game = True
                    wrong_label_counts[f"{truth}->{label}"] += 1
                    if label in ACTIONABLE_LABELS:
                        actionable_wrong_this_game = True
                        actionable_wrong_label_counts[f"{truth}->{label}"] += 1
            if fired_this_game:
                belief_fired[bucket] += 1
            if wrong_this_game:
                belief_wrong[bucket] += 1
            if actionable_wrong_this_game:
                belief_actionable_wrong[bucket] += 1

        n_win = sum(1 for r in rows if r["outcome"] == "our_win")
        n_loss = sum(1 for r in rows if r["outcome"] == "opp_win")
        print(f"    belief-net fired:      win {belief_fired['our_win']}/{n_win} "
              f"({belief_fired['our_win']/n_win:.1%})" if n_win else "    belief-net fired: win n/a",
              end="  ")
        print(f"loss {belief_fired['opp_win']}/{n_loss} "
              f"({belief_fired['opp_win']/n_loss:.1%})" if n_loss else "loss n/a")
        print(f"    belief-net WRONG (!=truth): win {belief_wrong['our_win']}/{n_win} "
              f"({belief_wrong['our_win']/n_win:.1%})" if n_win else "    belief-net wrong: win n/a",
              end="  ")
        print(f"loss {belief_wrong['opp_win']}/{n_loss} "
              f"({belief_wrong['opp_win']/n_loss:.1%})" if n_loss else "loss n/a")
        print(f"    belief-net ACTIONABLE-WRONG (one of the 8 branch-consumed labels, wrong): "
              f"win {belief_actionable_wrong['our_win']}/{n_win} "
              f"({belief_actionable_wrong['our_win']/n_win:.1%})" if n_win
              else "    belief-net actionable-wrong: win n/a", end="  ")
        print(f"loss {belief_actionable_wrong['opp_win']}/{n_loss} "
              f"({belief_actionable_wrong['opp_win']/n_loss:.1%})" if n_loss else "loss n/a")
        if wrong_label_counts:
            print(f"    wrong-label breakdown: {dict(wrong_label_counts.most_common())}")
        if actionable_wrong_label_counts:
            print(f"    ACTIONABLE wrong-label breakdown: "
                  f"{dict(actionable_wrong_label_counts.most_common())}")
        if label_counts:
            print(f"    all belief-net labels fired: {dict(label_counts.most_common())}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
