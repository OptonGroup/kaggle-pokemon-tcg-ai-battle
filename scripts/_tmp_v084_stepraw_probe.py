"""_tmp_v084_stepraw_probe.py — [v084 v2] Единственная цель: доказать, что низкоуровневый
raw search_step (json.loads(lib.SearchStep(...)) -- ДО json_to_dataclass) даёт obs_dict,
совместимый с bc_features_2b.build_features, и что можно держать >1 search_begin-мира живыми
одновременно (нужно UCB1-аллокации v2-ядра). НЕ гейт, НЕ измерение силы — только техническая
проверка перед переписыванием search_gs.py.

Стенд: macOS ARM -> cg_native через sys.modules-алиас (паттерн scripts/gate_scout_guard.py:124-137).

Запуск:
    .venv/bin/python scripts/_tmp_v084_stepraw_probe.py
"""
from __future__ import annotations

import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

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

from cg.game import battle_start, battle_finish, battle_select  # noqa: E402
from cg.api import to_observation_class, search_begin, search_step, search_end  # noqa: E402
import cg.api as capi  # noqa: E402

SUB_DIR = os.path.join(ROOT, "submissions", "v084_bc2b_search")
if SUB_DIR not in sys.path:
    sys.path.insert(0, SUB_DIR)

import importlib.util


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        os.chdir(SUB_DIR)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


bcf = _load_module("_probe_bc_features_2b", os.path.join(SUB_DIR, "bc_features_2b.py"))
main_mod = _load_module("_probe_main", os.path.join(SUB_DIR, "main.py"))
my_deck = list(main_mod.my_deck)

MAIN_CTX = int(capi.SelectContext.MAIN)
OK = "OK"
FAIL = "FAIL"


def step_raw(search_id: int, select: list) -> tuple:
    """[v084 v2] Low-level SearchStep: bypass json_to_dataclass (cg/api.py:597-627 is the typed
    wrapper this ports the error-code table from), return the raw parsed dict straight off the
    engine's JSON so it is byte-identical in shape to what main.py's agent() receives as
    obs_dict for a REAL decision (agent() gets called with the raw dict the harness/engine
    hands it -- see cg/game.py's battle_select, never a dataclass). This is the exact
    unlock cited in the task: json.loads(bs)["state"]["observation"].
    """
    lib = capi.lib
    import ctypes
    bs = lib.SearchStep(capi.agent_ptr, search_id, (ctypes.c_int * len(select))(*select), len(select))
    result = json.loads(bs.decode())
    err = result.get("error", 0)
    if err != 0:
        table = {
            1: "There is no element with the specified search_id.",
            2: "Released item.",
            3: "Cannot be selected because the battle has ended.",
            4: "Must be minCount <= len(select) <= maxCount.",
            5: "Must be 0 <= select elements < len(option).",
            6: "Duplicate select elements.",
            30: "agent_ptr broken.",
        }
        raise ValueError(table.get(err, f"unknown error {err}"))
    state = result["state"]
    return int(state["searchId"]), state["observation"]


def sample_hidden_dummy(obs_root) -> dict:
    """Minimal stand-in for search_gs._sample_hidden -- just enough to satisfy search_begin's
    length checks with legal card ids (own deck padding). Production sampler is unaffected by
    this probe; this is only to reach a search_begin call at all."""
    st = obs_root.current
    me_i = st.yourIndex
    me = st.players[me_i]
    op = st.players[1 - me_i]
    seen_mine = set()
    for c in (me.hand or []):
        seen_mine.add(c.id)
    your_deck = (my_deck * 3)[: me.deckCount]
    your_prize = (my_deck * 3)[: len(me.prize)]
    opp_deck = (my_deck * 3)[: op.deckCount]
    opp_prize = (my_deck * 3)[: len(op.prize)]
    opp_hand = (my_deck * 3)[: op.handCount]
    opp_active = []
    if op.active and op.active[0] is None:
        opp_active = [my_deck[0]]
    return dict(your_deck=your_deck, your_prize=your_prize, opponent_deck=opp_deck,
                opponent_prize=opp_prize, opponent_hand=opp_hand, opponent_active=opp_active)


def find_first_main_decision(seed: int):
    random.seed(seed)
    deck0 = list(my_deck)
    deck1 = list(my_deck)
    obs_dict, _ = battle_start(deck0, deck1)
    moves = 0
    while obs_dict is not None and moves < 2000:
        sel = obs_dict.get("select")
        if sel is None:
            break  # terminal / no more decisions
        cur = obs_dict.get("current") or {}
        ctx = sel.get("context")
        n_opts = len(sel.get("option") or [])
        turn = int(cur.get("turn") or 0)
        if (ctx is not None and int(ctx) == MAIN_CTX and 3 <= n_opts <= 24 and turn >= 2):
            return obs_dict
        try:
            choice = main_mod.agent(obs_dict)
        except Exception as e:
            print(f"  [warn] agent() raised at move {moves}: {e!r}")
            choice = [0] if n_opts else []
        obs_dict = battle_select(choice)
        moves += 1
    return None


def main():
    print("=== v084 v2 _step_raw probe ===")
    battle_finish()  # defensive: clear any stray prior session in this interpreter
    obs_dict = None
    for seed in range(1000, 1020):
        try:
            obs_dict = find_first_main_decision(seed)
        except Exception as e:
            print(f"  seed={seed} setup failed: {e!r}")
            obs_dict = None
        if obs_dict is not None:
            print(f"seed={seed}: found MAIN decision, turn="
                  f"{(obs_dict.get('current') or {}).get('turn')} "
                  f"n_opts={len(obs_dict['select']['option'])}")
            break
    if obs_dict is None:
        print(f"{FAIL}: no MAIN decision (3<=n<=24, turn>=2) found in 20 seeds")
        battle_finish()
        return

    n_options = len(obs_dict["select"]["option"])
    obs_root = to_observation_class(obs_dict)
    me_i = obs_root.current.yourIndex
    print(f"root: yourIndex={me_i} n_options={n_options} "
          f"search_begin_input_present={obs_dict.get('search_begin_input') is not None}")

    # ── Test 1: single search_begin + one raw step, key/shape probe ─────────
    try:
        hidden = sample_hidden_dummy(obs_root)
        root = search_begin(obs_root, **hidden)
        root_sid = root.searchId
        print(f"{OK}: search_begin -> searchId={root_sid}")
    except Exception as e:
        print(f"{FAIL}: search_begin raised {e!r}")
        battle_finish()
        return

    try:
        sid2, raw_obs = step_raw(root_sid, [0])
        print(f"{OK}: step_raw(root_sid, [0]) -> searchId={sid2}")
        print(f"  raw_obs top-level keys: {sorted(raw_obs.keys())}")
        print(f"  raw_obs['current'] keys: {sorted((raw_obs.get('current') or {}).keys())}")
        sel2 = raw_obs.get("select")
        print(f"  raw_obs['select'] present: {sel2 is not None}"
              + (f", keys={sorted(sel2.keys())}, n_options={len(sel2.get('option') or [])}"
                 if sel2 else ""))
    except Exception as e:
        print(f"{FAIL}: step_raw raised {e!r}")
        search_end()
        battle_finish()
        return

    # ── Test 2: build_features on the raw dict from step_raw ────────────────
    try:
        if sel2 is not None and (sel2.get("option") or []):
            opts2 = sel2["option"]
            feat = bcf.build_features(raw_obs, opts2[0], 0, len(opts2))
            print(f"{OK}: build_features(raw_obs, opt0, 0, {len(opts2)}) -> "
                  f"len={len(feat)} (expect N_FEATURES={bcf.N_FEATURES}) "
                  f"first5={feat[:5]}")
            assert len(feat) == bcf.N_FEATURES, "feature vector length mismatch"
        else:
            print("  (no select at this depth -- terminal/forced node, skip build_features check "
                  "on this branch, try another candidate index below)")
    except Exception as e:
        print(f"{FAIL}: build_features on raw step_raw output raised {e!r}")

    # ── Test 3: two concurrent search_begin worlds (no intervening search_end),
    #    branch independently from each, verify no cross-contamination ──────
    try:
        hidden_b = sample_hidden_dummy(obs_root)
        root_b = search_begin(obs_root, **hidden_b)
        root_b_sid = root_b.searchId
        print(f"{OK}: second concurrent search_begin -> searchId={root_b_sid} "
              f"(distinct from first: {root_b_sid != root_sid})")
        sid_a2, obs_a2 = step_raw(root_sid, [0])
        sid_b2, obs_b2 = step_raw(root_b_sid, [0])
        print(f"{OK}: branched both worlds independently -> "
              f"world A child sid={sid_a2}, world B child sid={sid_b2} "
              f"(distinct: {sid_a2 != sid_b2})")
        cur_a = obs_a2.get("current") or {}
        cur_b = obs_b2.get("current") or {}
        print(f"  world A turn={cur_a.get('turn')} yourIndex={cur_a.get('yourIndex')}; "
              f"world B turn={cur_b.get('turn')} yourIndex={cur_b.get('yourIndex')}")
    except Exception as e:
        print(f"{FAIL}: concurrent-worlds test raised {e!r}")

    # ── Test 4: search_end() once, then a fresh search_begin afterwards still works ──
    try:
        search_end()
        print(f"{OK}: search_end() completed")
        hidden_c = sample_hidden_dummy(obs_root)
        root_c = search_begin(obs_root, **hidden_c)
        print(f"{OK}: post-search_end search_begin still works -> searchId={root_c.searchId}")
        search_end()
    except Exception as e:
        print(f"{FAIL}: post-search_end search_begin raised {e!r}")

    # ── Test 5: turn-history module-global contamination risk (bc_features_2b) ──
    try:
        snap_before = (bcf._turn_hist_turn, list(bcf._turn_hist_buf),
                       bcf._turn_hist_last_sig, bcf._turn_hist_ability_used)
        # Simulate what a rollout would do: call build_features on a handful of
        # (possibly same-turn-number) hypothetical states.
        if sel2 is not None and (sel2.get("option") or []):
            for _ in range(3):
                bcf.build_features(raw_obs, sel2["option"][0], 0, len(sel2["option"]))
        snap_after_sim = (bcf._turn_hist_turn, list(bcf._turn_hist_buf),
                          bcf._turn_hist_last_sig, bcf._turn_hist_ability_used)
        changed = snap_before != snap_after_sim
        print(f"{'OK (confirms risk, as expected)' if changed else 'OK (no change)'}: "
              f"module-level turn-history globals {'DID' if changed else 'did NOT'} mutate "
              f"after simulated-world build_features calls -- "
              f"before={snap_before[:1]}/{snap_before[2]} "
              f"after={snap_after_sim[:1]}/{snap_after_sim[2]}")
        # restore, as the real search_gs.decide() must do around every rollout
        bcf._turn_hist_turn, bcf._turn_hist_buf, bcf._turn_hist_last_sig, bcf._turn_hist_ability_used = snap_before
        snap_restored = (bcf._turn_hist_turn, list(bcf._turn_hist_buf),
                         bcf._turn_hist_last_sig, bcf._turn_hist_ability_used)
        print(f"{OK if snap_restored == snap_before else FAIL}: snapshot/restore round-trip "
              f"{'matches' if snap_restored == snap_before else 'MISMATCH'}")
    except Exception as e:
        print(f"{FAIL}: turn-history contamination probe raised {e!r}")

    battle_finish()
    print("\n=== probe done ===")


if __name__ == "__main__":
    main()
