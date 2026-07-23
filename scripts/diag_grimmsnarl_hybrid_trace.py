"""Диагностика (разовая, не для гейта): проверяет, что
submissions/external_prvsiyan_grimmsnarl_hybrid реально гоняет нейросеть
(bc_agent.predict_action -> модельный путь), а не молча падает в
_fallback_action из-за исключения при загрузке чекпоинта/инференсе.

Играет несколько игр vs v017_alakazam_heur напрямую (single-process, без
ProcessPoolExecutor) с монки-патчем bc_agent.predict_action, который считает
вызовы модельного пути vs fallback-пути и логирует первое исключение (если
есть) целиком (traceback), а не проглатывает его как gate_v064_regression.py.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
from cg_native.api import to_observation_class  # noqa: E402
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

CAND_DIR = os.path.join(ROOT, "submissions", "external_prvsiyan_grimmsnarl_hybrid")
OPP_DIR = os.path.join(ROOT, "submissions", "v017_alakazam_heur")

sys.path.insert(0, CAND_DIR)

import importlib.util


def load_module(sub_dir, name):
    main_path = os.path.join(sub_dir, "main.py")
    original_cwd = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(name, main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)
    return mod


print("[diag] loading candidate module (external_prvsiyan_grimmsnarl_hybrid)...", flush=True)
t0 = time.time()
cand_mod = load_module(CAND_DIR, "_diag_cand")
print(f"[diag] loaded in {time.time()-t0:.2f}s", flush=True)

bc_agent = cand_mod.bc_agent if hasattr(cand_mod, "bc_agent") else sys.modules.get("bc_agent")
print(f"[diag] cand_mod._MODEL is None? {cand_mod._MODEL is None}", flush=True)
print(f"[diag] cand_mod._MODEL type: {type(cand_mod._MODEL)}", flush=True)

# Monkeypatch bc_agent.predict_action to count model-path vs fallback-path calls.
orig_predict_action = bc_agent.predict_action
counters = {"model_path": 0, "fallback_path": 0, "deck_select": 0, "exceptions": 0}
first_exc = {"tb": None}


def traced_predict_action(obs, model):
    select = obs.get("select")
    if not isinstance(select, dict):
        counters["deck_select"] += 1
        return orig_predict_action(obs, model)
    options = select.get("option")
    if not options or model is None:
        counters["fallback_path"] += 1
    else:
        counters["model_path"] += 1
    try:
        return orig_predict_action(obs, model)
    except Exception:
        counters["exceptions"] += 1
        if first_exc["tb"] is None:
            first_exc["tb"] = traceback.format_exc()
        raise


bc_agent.predict_action = traced_predict_action
# main.py did `from bc_agent import load_policy, predict_action` -> also patch that binding
cand_mod.predict_action = traced_predict_action


def traced_agent(obs_dict):
    base_action = traced_predict_action(obs_dict, cand_mod._MODEL)
    punk_up = cand_mod._punk_up_override(obs_dict)
    if punk_up is not None:
        return punk_up
    target = cand_mod._lowest_hp_override(obs_dict, base_action)
    return target if target is not None else base_action


cand_agent = traced_agent
cand_deck = getattr(cand_mod, "my_deck", None) or None
if cand_deck is None:
    with open(os.path.join(CAND_DIR, "deck.csv")) as f:
        cand_deck = [int(x.strip()) for x in f if x.strip()]

print(f"[diag] loading opponent module (v017_alakazam_heur)...", flush=True)
opp_mod = load_module(OPP_DIR, "_diag_opp")
opp_agent = opp_mod.agent
with open(os.path.join(OPP_DIR, "deck.csv")) as f:
    opp_deck = [int(x.strip()) for x in f if x.strip()]

N_GAMES = 8
MAX_MOVES = 2000

results = []
for game_idx in range(N_GAMES):
    our_first = (game_idx % 2 == 0)
    if our_first:
        deck0, deck1 = list(cand_deck), list(opp_deck)
        agent0, agent1 = cand_agent, opp_agent
    else:
        deck0, deck1 = list(opp_deck), list(cand_deck)
        agent0, agent1 = opp_agent, cand_agent

    t_g0 = time.time()
    obs_dict, start = battle_start(deck0, deck1)
    move_count = 0
    winner = None
    outcome = None
    game_exc = None
    try:
        while move_count < MAX_MOVES:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                winner = obs.current.result
                break
            your_index = obs.current.yourIndex if obs.current else 0
            agent_fn = agent0 if your_index == 0 else agent1
            try:
                choice = agent_fn(obs_dict)
            except Exception:
                game_exc = traceback.format_exc()
                outcome = "exception"
                break
            obs_dict = battle_select(choice)
            move_count += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass

    our_side = 0 if our_first else 1
    if outcome is None:
        if winner is None:
            outcome = "timeout"
        elif winner == our_side:
            outcome = "our_win"
        elif winner == 2:
            outcome = "draw"
        else:
            outcome = "opp_win"

    elapsed = time.time() - t_g0
    print(f"[diag] game {game_idx}: our_first={our_first} outcome={outcome} moves={move_count} "
          f"elapsed={elapsed:.2f}s model_calls={counters['model_path']} fallback_calls={counters['fallback_path']}",
          flush=True)
    if game_exc:
        print(f"[diag] game {game_idx} EXCEPTION TRACEBACK:\n{game_exc}", flush=True)
    results.append(outcome)

print("\n=== SUMMARY ===", flush=True)
print(f"outcomes: {results}", flush=True)
print(f"counters: {counters}", flush=True)
if first_exc["tb"]:
    print(f"first predict_action exception traceback:\n{first_exc['tb']}", flush=True)
print(f"our_wins={results.count('our_win')} opp_wins={results.count('opp_win')} "
      f"draws={results.count('draw')} exceptions={results.count('exception')} timeouts={results.count('timeout')}",
      flush=True)
