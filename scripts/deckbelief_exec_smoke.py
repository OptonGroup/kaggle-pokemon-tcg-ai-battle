#!/usr/bin/env python3
"""deckbelief_exec_smoke.py — Kaggle live-safety exec-smoke for v047_deckbelief.

Mirrors budget_smoke.py's loader (exec() into a bare namespace, no __file__)
— the three historical live-only bugs (missing __file__, lazy import
silently failing, numpy/torch ABI mismatch) all reproduce ONLY under this
loading pattern, never under a plain import or importlib.spec_from_file
(which sets __file__ and is what deckbelief_gate_v047.py/eval_vs_pool.py use
— they don't cover this class of bug).

On top of "0 exceptions" this additionally proves the belief-net path is
actually exercised (not silently degrading to the v044 board-only fallback,
which would still show 0 exceptions and would mean we shipped v044 behavior
under a v047 label).

    python3 scripts/deckbelief_exec_smoke.py --games 5
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

KAGGLE_AGENT_DIR = Path("/kaggle_simulations/agent")


def stage_at_kaggle_path(sub_dir: Path):
    """Copy the submission dir to /kaggle_simulations/agent, the real Kaggle
    staging location main.py's own ROOT-bootstrap checks first. Two agents
    can't both occupy this path in the same process (only relevant for the
    agent under test — the opponent's deck is read directly from disk
    instead of round-tripping through its own agent(select=None), mirroring
    how deckbelief_gate_v047.py/eval_vs_pool.py already do it)."""
    if KAGGLE_AGENT_DIR.exists():
        shutil.rmtree(KAGGLE_AGENT_DIR)
    shutil.copytree(sub_dir, KAGGLE_AGENT_DIR)
    return KAGGLE_AGENT_DIR


def load_agent_no_file(load_dir: Path):
    """Load main.py the way Kaggle's exec() loader does: no __file__, cwd
    left untouched (Kaggle's cwd is not the agent dir — that's the whole
    point of the ROOT-bootstrap's /kaggle_simulations/agent check)."""
    sys.path.insert(0, str(load_dir))
    src = (load_dir / "main.py").read_text()
    ns: dict = {"__name__": "__main__"}
    exec(compile(src, "<string>", "exec"), ns)  # noqa: S102 — intentional, mirrors Kaggle
    assert "__file__" not in ns, "exec namespace unexpectedly has __file__ — not mirroring Kaggle"
    return ns


def load_deck_csv(path: Path) -> list[int]:
    return [int(x.strip()) for x in path.read_text().strip().split("\n") if x.strip()]


def load_opponent_agent(sub_dir: Path):
    """Plain load for the opponent side — only used to generate moves (never
    deck-select, so its own read_deck_csv() cwd-quirk is irrelevant here).
    We're not smoke-testing the opponent, just need its real playing policy
    (sys.path must include sub_dir for its own bundled modules, e.g.
    v017_alakazam_heur's features_alakazam.py, to import correctly)."""
    import importlib.util
    import os
    sys.path.insert(0, str(sub_dir))
    old_cwd = os.getcwd()
    os.chdir(sub_dir)
    try:
        spec = importlib.util.spec_from_file_location("_smoke_opponent", sub_dir / "main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(old_cwd)
    return mod.agent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v047_deckbelief")
    p.add_argument("--opponent", default="v044_archaludon_abomasnow")
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--max-moves", type=int, default=200)
    args = p.parse_args()

    from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

    sub_dir = REPO / "submissions" / args.version
    opp_dir = REPO / "submissions" / args.opponent

    staged = stage_at_kaggle_path(sub_dir)
    print(f"[smoke] staged {args.version} at {staged}, loading via exec(), no __file__; "
          f"opponent={args.opponent}", flush=True)
    ns = load_agent_no_file(staged)
    agent = ns["agent"]

    print(f"[smoke] _torch_ok={ns.get('_torch_ok')}", flush=True)
    if ns.get("_torch_ok") is not True:
        print("[smoke] FAIL: torch import guard tripped under exec() — belief net "
              "will silently degrade to v044 board-only fallback on Kaggle", flush=True)
        return 1

    # Opponent's deck is read directly (bypassing its own agent(select=None))
    # — we're only smoke-testing v047's loading, not the opponent's.
    opp_deck = load_deck_csv(opp_dir / "deck.csv")
    opp_agent = load_opponent_agent(opp_dir)

    total_exceptions = 0
    belief_used_any_game = False

    for g in range(args.games):
        deck = agent({"select": None})
        assert len(deck) == 60, f"deck has {len(deck)} cards"
        assert len(opp_deck) == 60, f"opp deck has {len(opp_deck)} cards"

        obs_dict, _start = battle_start(list(deck), list(opp_deck))
        for move in range(args.max_moves):
            sel = obs_dict.get("select")
            if sel is None:
                break
            obs_dict["remainingOverageTime"] = 600.0
            try:
                if obs_dict["current"]["yourIndex"] == 0:
                    choice = agent(obs_dict)
                else:
                    choice = opp_agent(obs_dict)
            except Exception as e:
                total_exceptions += 1
                print(f"[smoke] EXCEPTION game={g} move={move}: {type(e).__name__}: {e}", flush=True)
                raise
            obs_dict = battle_select(choice)
            if obs_dict.get("current", {}).get("result", -1) >= 0:
                break
        battle_finish()

        model_loaded = ns.get("_belief_model") is not None
        load_failed = ns.get("_belief_model_load_failed")
        last_matchup = ns.get("_last_matchup")
        print(f"[smoke] game={g} done, belief_model_loaded={model_loaded} "
              f"load_failed={load_failed} last_matchup={last_matchup}", flush=True)
        if model_loaded:
            belief_used_any_game = True

    print(f"[smoke] total_exceptions={total_exceptions} belief_model_ever_loaded={belief_used_any_game}", flush=True)
    if total_exceptions > 0:
        print("[smoke] FAIL: exceptions during play", flush=True)
        return 1
    if not belief_used_any_game:
        print("[smoke] FAIL: belief net never loaded across all games", flush=True)
        return 1
    print("[smoke] PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
