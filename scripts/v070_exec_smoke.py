#!/usr/bin/env python3
"""v070_exec_smoke.py — Kaggle live-safety exec-smoke for v070_grimmsnarl_hybrid.

Adaptation of scripts/v061_exec_smoke.py for the neural (torch) v070 pilot. Mirrors
Kaggle's real agent loader (exec() into a bare namespace, no __file__, files staged at
/kaggle_simulations/agent/) — the historical live-only bugs (missing __file__, lazy
import silently failing, numpy/ABI mismatch) reproduce ONLY under this loading pattern,
never under a plain import / spec_from_file_location (which sets __file__).

Run inside the linux/amd64 Docker image (pokemon-tcg-sim-torch:latest) so the bundled
cg/libcg.so (Linux) path is exercised, not the local arm64 cg_native build, and so torch
loads exactly as it will on Kaggle.

    docker run --rm --platform linux/amd64 -v $(pwd):/workspace -w /workspace \
        pokemon-tcg-sim-torch:latest python3 scripts/v070_exec_smoke.py --games 6
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

KAGGLE_AGENT_DIR = Path("/kaggle_simulations/agent")


def stage_at_kaggle_path(sub_dir: Path):
    KAGGLE_AGENT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if KAGGLE_AGENT_DIR.exists():
        shutil.rmtree(KAGGLE_AGENT_DIR)
    shutil.copytree(sub_dir, KAGGLE_AGENT_DIR)
    return KAGGLE_AGENT_DIR


def load_agent_no_file(load_dir: Path):
    """Load main.py the way Kaggle's exec() loader does: no __file__, cwd untouched."""
    sys.path.insert(0, str(load_dir))
    src = (load_dir / "main.py").read_text()
    ns: dict = {"__name__": "__main__"}
    exec(compile(src, "<string>", "exec"), ns)  # noqa: S102 — intentional, mirrors Kaggle
    assert "__file__" not in ns, "exec namespace unexpectedly has __file__ — not mirroring Kaggle"
    return ns


def load_deck_csv(path: Path) -> list[int]:
    return [int(x.strip()) for x in path.read_text().strip().split("\n") if x.strip()]


def load_opponent_agent(sub_dir: Path):
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
    p.add_argument("--version", default="v070_grimmsnarl_hybrid")
    p.add_argument("--opponent", default="v009a_crustle")
    p.add_argument("--games", type=int, default=6)
    p.add_argument("--max-moves", type=int, default=300)
    args = p.parse_args()

    from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

    sub_dir = REPO / "submissions" / args.version
    opp_dir = REPO / "submissions" / args.opponent

    # cwd deliberately NOT the staged dir — /tmp, like Kaggle's harness process cwd.
    os.chdir("/tmp")

    staged = stage_at_kaggle_path(sub_dir)
    print(f"[smoke] staged {args.version} at {staged}, loading via exec(), no __file__, "
          f"cwd={os.getcwd()}; opponent={args.opponent}", flush=True)
    ns = load_agent_no_file(staged)
    agent = ns["agent"]
    print(f"[smoke] main.py loaded OK, DECISION_LOG={ns.get('DECISION_LOG')}", flush=True)

    opp_deck = load_deck_csv(opp_dir / "deck.csv")
    opp_agent = load_opponent_agent(opp_dir)

    total_exceptions = 0
    total_moves = 0
    for g in range(args.games):
        deck = agent({"select": None})
        assert isinstance(deck, list) and len(deck) == 60 and all(isinstance(c, int) for c in deck), \
            f"deck-select returned invalid deck: {deck!r}"
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
            assert isinstance(choice, list) and all(isinstance(i, int) for i in choice), \
                f"agent returned non-list-of-int: {choice!r}"
            obs_dict = battle_select(choice)
            total_moves += 1
            if obs_dict.get("current", {}).get("result", -1) >= 0:
                break
        battle_finish()
        print(f"[smoke] game {g} done, moves so far total={total_moves}", flush=True)

    print(f"[smoke] total_moves={total_moves} total_exceptions={total_exceptions}", flush=True)
    if total_exceptions > 0:
        print("[smoke] FAIL: exceptions during play", flush=True)
        return 1
    print("[smoke] PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
