"""eval_vs_pool.py — multi-opponent local gate (cg_native, non-god-mode).

Task 16/18 (variant B, infra-before-GPU-run): plays "our agent" — either an
existing submissions/vNNN/ dir OR a raw PPO checkpoint (.pth) not yet wrapped
into a submission — against a fixed reference pool {v007, v009a, v033, v036},
side-alternating, honest non-god-mode (cg_native's obs_dict structurally
cannot leak the opponent's hand — see PlayerState.hand docstring in
src/cg_native/api.py: "None for the opponent" — so there is no hide_opp_hand
FLAG to forget here, unlike the ptcg_engine/rl_state_ids training backend).

Design choice — cg_native (gate_expanded.py lineage), NOT eval_vs_expert.py
(ptcg_engine lineage):
  scripts/gate_expanded.py already plays submissions/vNNN/main.py agents
  against each other on cg_native, locally on macOS ARM, with the EXACT
  obs_dict shape Kaggle feeds at inference. It already supports Wilson95 CI,
  side-alternation and per-agent exception counters. This file generalizes
  that proven harness to (a) loop over several named opponents in one run,
  and (b) accept a raw .pth checkpoint as "our agent" by wrapping it with the
  submission-bundled converter/model (cabt_conv.py + nn_plan_d.py, copied
  byte-for-byte into every Plan-D submission dir for Kaggle self-containment)
  so a fresh GPU checkpoint can be gated WITHOUT first hand-building a
  submissions/vNNN/ dir.

  eval_vs_expert.py's ptcg_engine path was considered (it evaluates a raw
  checkpoint directly via rl_state_ids/rl_action_ids/rl_step) but rejected as
  the base for this file: (1) it needs an opponent's `main.py` to ALSO speak
  the packed-ID rl_step action-index convention indirectly via
  cg_observation_with_view — one more moving part than cg_native, which the
  opponent's main.py already speaks natively since that's what Kaggle feeds
  it; (2) hide_opp_hand there is a boolean the caller must remember to pass
  correctly (v033's main.py docstring flags this exact class of bug — PPO
  training/eval both ran god-mode by accident); cg_native's obs_dict has no
  such footgun — the opponent's hand key is simply absent from the dict.
  ptcg_engine remains the right tool for the GPU-side training loop itself
  (env_adapter.py / train_ppo_v2.py); this file is downstream of that, for
  gating what training produced.

Reference pool (submissions/ subdir names, see OPPONENT_REGISTRY below):
    v007  -> v007_crustle       (IL/GBDT Crustle)
    v009a -> v009a_crustle      (IL/GBDT Crustle, current strongest local ref)
    v033  -> v033_plan_d_ppo    (PPO NN, no MCTS)
    v036  -> v036_archaludon_patched (heuristic + Archaludon tech)

Usage — gate an existing submission dir against all 4:
    .venv/bin/python scripts/eval_vs_pool.py \\
        --agent-dir v038_plan_d_ppo_poolfix --opponents v007,v009a,v033,v036 \\
        --games 100 --trace-games 5 --trace-dir results/gates/traces/v038_pool \\
        --out results/gates/v038_vs_pool.json --tag v038_vs_pool

Usage — gate a raw PPO checkpoint fresh off the GPU box (no submission dir
yet), reusing the v038 submission's bundled converter/model architecture:
    .venv/bin/python scripts/eval_vs_pool.py \\
        --ckpt /path/to/ppo_new_final.pth --agent-deck decks/v039_pool/deck.csv \\
        --opponents v009a --games 100 --deterministic \\
        --trace-games 5 --trace-dir results/gates/traces/new_ckpt \\
        --out results/gates/new_ckpt_vs_pool.json --tag new_ckpt_vs_v009a

Output JSON: {tag, agent, opponents: {name: {win_rate, wilson95_lo/hi, wins/
losses/draws, avg_game_length, elapsed_s, ...}}, overall_win_rate}.

--trace-dir / --trace-games write results/gates/traces/<tag>/trace_<opp>.json
— condensed per-decision records for OUR side only (board state + chosen
option), consumed by forensic_failure_modes.py (see
scratchpad/gold/plan_d/d1_train/forensic_failure_modes.py).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

ROOT = "/Users/max/kaggle-pokemon-tcg-ai-battle"
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# lightgbm + torch coexisting in one process is fragile on at least one dev
# machine (macOS ARM venv): torch's default multi-threaded intra-op pool
# segfaults on ordinary CPU tensor ops once lightgbm's own OpenMP pool has
# initialized in the same process, UNLESS lightgbm is imported first and
# torch is forced single-threaded. This gate loads v007/v009a (lightgbm
# GBDT) and v033/v036/--ckpt (torch NN) opponents/agents in the SAME
# process, so both orderings can occur depending on --agent-dir/--ckpt vs
# --opponents — importing both up front, in this order, sidesteps it
# regardless of which combination the CLI args pick. Mirror of
# eval_vs_expert.py:34-37 (confirmed: dropping this guard reproduces a
# SIGSEGV / exit 139 with --ckpt + a lightgbm opponent in the pool).
import lightgbm as _lgb_import_order_guard  # noqa: F401,E402
import torch  # noqa: E402
torch.set_num_threads(1)

# cg alias shim BEFORE loading any agent that does `from cg.api import ...`
# (every submissions/vNNN/main.py does this).
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

from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
from cg_native.api import to_observation_class, AreaType  # noqa: E402

SUBMISSIONS = os.path.join(ROOT, "submissions")
MAX_MOVES = 2000

# short name -> submissions/<dir> (Task 16 pool: "проверять правильность новой
# версии... на нескольких прошлых версиях: 7, 9, 36, 33").
OPPONENT_REGISTRY = {
    "v007": "v007_crustle",
    "v009a": "v009a_crustle",
    "v033": "v033_plan_d_ppo",
    "v036": "v036_archaludon_patched",
    "v017": "v017_alakazam_heur",
}

# Reference submission dir supplying the generic (non-version-specific)
# cabt_conv.py + nn_plan_d.py used to wrap a raw checkpoint. Any Plan-D
# submission dir works — v038 is the most recently shipped.
DEFAULT_ARCH_REF_DIR = os.path.join(SUBMISSIONS, "v038_plan_d_ppo_poolfix")

# SelectContext values (src/cg_native/api.py::SelectContext — authoritative,
# matches submissions/v017_alakazam_heur/main.py's empirically-used CTX_*
# constants). Used by both the trace recorder below and
# forensic_failure_modes.py.
CTX_MAIN = 0
CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2
CTX_SWITCH = 3       # "select the Pokemon to swap with the one in your Active Spot" (retreat target)
CTX_TO_ACTIVE = 4    # "select the Pokemon to put into your Active Spot" (active spot EMPTY -> KO promotion)
OPT_RETREAT = 12
OPT_ATTACK = 13


def wilson(wins, n, z=1.96):
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = (z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))) / den
    return (max(0.0, c - m), min(1.0, c + m))


def load_submission_module(dir_name, cache_tag=None):
    """Load submissions/<dir_name>/main.py in a sys.path-isolated fashion.

    Mirrors train_ppo_v2.py::_load_external_agent's snapshot/restore of
    sys.path: v007 and v009a both ship a `features.py` — leaving one
    submission's dir on sys.path would silently shadow the other's `import
    features` when several opponents are loaded in the same process (this is
    exactly what this multi-opponent gate does).
    """
    cache_key = f"_pool_agent_{cache_tag or dir_name}"
    if cache_key in sys.modules:
        return sys.modules[cache_key]
    sub_dir = os.path.join(SUBMISSIONS, dir_name)
    main_path = os.path.join(sub_dir, "main.py")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"submission main.py not found: {main_path}")
    sys_path_snapshot = list(sys.path)
    cwd = os.getcwd()
    try:
        sys.path.insert(0, sub_dir)
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(cache_key, main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        sys.modules[cache_key] = mod
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
        sys.path[:] = sys_path_snapshot
    if not hasattr(mod, "agent"):
        del sys.modules[cache_key]
        raise AttributeError(f"no agent() in {main_path}")
    return mod


def load_deck_file(path):
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip()]


def load_submission_deck(dir_name):
    return load_deck_file(os.path.join(SUBMISSIONS, dir_name, "deck.csv"))


# ---------------------------------------------------------------------------
# Raw-checkpoint agent wrapper. Reuses the submission-bundled converter
# (cabt_obs_to_state_packed / cabt_select_to_action_packed) + model class
# (PtcgPolicyValueNet) so a fresh GPU checkpoint can be gated without first
# hand-assembling a submissions/vNNN/ dir. Mirrors
# submissions/v033_plan_d_ppo/main.py::_get_model / _nn_action, same
# never-throws fallback-to-first-legal-options contract as a real submission
# (a crash here would silently misattribute a bug to "the checkpoint is bad"
# instead of "the gate harness broke").
# ---------------------------------------------------------------------------
class CheckpointAgent:
    def __init__(self, ckpt_path, arch_ref_dir=DEFAULT_ARCH_REF_DIR,
                 card_dim=128, hidden_dim=256, max_card_id=1300, max_attack_id=4096,
                 deterministic=True):
        import torch
        torch.set_num_threads(1)
        self._torch = torch
        import numpy as np
        self._np = np

        # Modules must be registered in sys.modules BEFORE exec_module: the
        # nn_plan_d.py classes use @dataclass, whose field-type resolution
        # (_is_type) looks up `sys.modules[cls.__module__]` at class-body
        # execution time -- if absent, dataclass() crashes with
        # "NoneType has no attribute '__dict__'".
        cc_spec = importlib.util.spec_from_file_location(
            "_pool_cabt_conv", os.path.join(arch_ref_dir, "cabt_conv.py"))
        cc = importlib.util.module_from_spec(cc_spec)
        sys.modules["_pool_cabt_conv"] = cc
        cc_spec.loader.exec_module(cc)
        nn_spec = importlib.util.spec_from_file_location(
            "_pool_nn_plan_d", os.path.join(arch_ref_dir, "nn_plan_d.py"))
        nn_mod = importlib.util.module_from_spec(nn_spec)
        sys.modules["_pool_nn_plan_d"] = nn_mod
        nn_spec.loader.exec_module(nn_mod)
        self._cabt_conv = cc

        model = nn_mod.PtcgPolicyValueNet(
            max_card_id=max_card_id, card_dim=card_dim, hidden_dim=hidden_dim,
            max_attack_id=max_attack_id, card_overflow_buckets=0,
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        self.model = model
        self.ckpt_path = ckpt_path
        self.deterministic = deterministic
        self.n_nn_decisions = 0
        self.n_fallback_decisions = 0
        self.n_exceptions = 0

    def _nn_action(self, obs_dict):
        torch = self._torch
        np = self._np
        cur = obs_dict.get("current")
        sel = obs_dict.get("select")
        opts = sel.get("option") or []
        n = len(opts)
        if n == 0 or cur is None:
            return None
        state = self._cabt_conv.cabt_obs_to_state_packed(obs_dict)
        action = self._cabt_conv.cabt_select_to_action_packed(sel, cur)
        sb = {k: torch.as_tensor(np.asarray(state[k])[None]) for k in
              ("in_play", "zones", "player_counts", "player_status", "global")}
        ab = {"options": torch.as_tensor(np.asarray(action["options"])[None]),
              "mask": torch.as_tensor(np.asarray(action["mask"])[None])}
        with torch.inference_mode():
            out = self.model.forward_tensors(
                sb["in_play"], sb["zones"], sb["player_counts"],
                sb["player_status"], sb["global"], ab["options"], ab["mask"],
            )
            logits = out["logits"].float()
        m = ab["mask"].bool()
        logits = logits.masked_fill(~m, torch.finfo(logits.dtype).min)
        min_c = int(sel.get("minCount") or 1)
        max_c = max(1, int(sel.get("maxCount") or 1))
        if min_c <= 1 and max_c == 1:
            if self.deterministic:
                return [int(torch.argmax(logits, dim=-1).item())]
            dist = torch.distributions.Categorical(logits=logits)
            return [int(dist.sample().item())]
        legal = torch.nonzero(m[0], as_tuple=False).flatten().tolist()
        if not legal:
            return None
        scored = sorted(((float(logits[0, i].item()), i) for i in legal), reverse=True)
        k = min(max(min_c, 1), max_c, len(scored))
        return [i for _, i in scored[:k]]

    def agent(self, obs_dict):
        sel = obs_dict.get("select") if obs_dict else None
        if sel is None:
            # Deck-selection prompt — never hit in this harness (battle_start
            # takes decks directly), kept only so this object satisfies the
            # same agent() contract a real submission main.py has.
            return []
        opts = sel.get("option") or []
        if len(opts) == 0:
            return []
        try:
            picked = self._nn_action(obs_dict)
            if picked:
                n = len(opts)
                min_c = int(sel.get("minCount") or 1)
                max_c = int(sel.get("maxCount") or 1)
                if all(0 <= i < n for i in picked) and min_c <= len(picked) <= max(max_c, 1):
                    self.n_nn_decisions += 1
                    return list(picked)
        except Exception as exc:  # never-throws contract
            self.n_exceptions += 1
            print(f"[eval_vs_pool] CheckpointAgent exception: {exc}", file=sys.stderr)
        self.n_fallback_decisions += 1
        min_c = int(sel.get("minCount") or 1)
        max_c = int(sel.get("maxCount") or 1)
        k = max(1, min(min_c, max_c, len(opts)))
        return list(range(k))


# ---------------------------------------------------------------------------
# Trace capture (forensic input) — condensed per-decision records for OUR
# side only. Uses to_observation_class() dataclasses (Pokemon.id/serial/
# energies, Option.cardId/serial, SelectData.context) directly rather than
# raw obs_dict area/index lookups: the engine already resolves cardId/serial
# for CARD-type options, which disambiguates duplicate-card-id cases (e.g.
# 3 copies of Mega Kangaskhan ex on the bench) that a naive area/index
# re-lookup would not.
# ---------------------------------------------------------------------------
def _pokemon_brief(p):
    if p is None:
        return None
    return {
        "id": p.id, "serial": p.serial, "hp": p.hp, "maxHp": p.maxHp,
        "n_energy": len(p.energies) if p.energies else 0,
    }


def _side_brief(pl):
    active = (pl.active or [None])[0] if pl.active else None
    return {
        "active": _pokemon_brief(active),
        "bench": [_pokemon_brief(b) for b in (pl.bench or [])],
        "poisoned": bool(pl.poisoned), "burned": bool(pl.burned), "asleep": bool(pl.asleep),
        "paralyzed": bool(pl.paralyzed), "confused": bool(pl.confused),
    }


def _resolve_target_pokemon(cur, me, opt):
    """Resolve the Pokemon an Option points at, via its area/index/playerIndex.

    NOTE: Option.cardId/Option.serial (per src/cg_native/api.py::Option
    docstrings) are populated ONLY for SKILL-type options — CARD-type options
    (used for retreat targets / KO-promotion targets / bench selection in
    general) instead carry area+index(+playerIndex), the same fields
    submissions/*/cabt_conv.py resolves manually via
    `_lookup_pokemon_id`/`_get_pokemon`. This mirrors that lookup using the
    typed dataclasses from to_observation_class() instead of raw dict access.
    """
    if opt is None or opt.area is None or opt.index is None:
        return None
    pidx = opt.playerIndex if opt.playerIndex is not None else me
    if pidx is None or pidx < 0 or pidx > 1:
        return None
    pl = cur.players[pidx]
    area = int(opt.area)
    idx = int(opt.index)
    if area == int(AreaType.ACTIVE):
        active = pl.active or []
        return active[0] if active else None
    if area == int(AreaType.BENCH):
        bench = pl.bench or []
        return bench[idx] if 0 <= idx < len(bench) else None
    return None


def _decision_record(obs, choice, step):
    cur = obs.current
    sel = obs.select
    if cur is None or sel is None:
        return None
    me = cur.yourIndex
    my = cur.players[me]
    opts = sel.option or []
    chosen = opts[choice[0]] if (choice and 0 <= choice[0] < len(opts)) else None
    target = _resolve_target_pokemon(cur, me, chosen)
    return {
        "step": step,
        "ctx": int(sel.context) if sel.context is not None else None,
        "sel_type": int(sel.type) if sel.type is not None else None,
        "opt_type": int(chosen.type) if (chosen is not None and chosen.type is not None) else None,
        "chosen_target": _pokemon_brief(target),
        "my": _side_brief(my),
    }


# ---------------------------------------------------------------------------
# Single-game loop (cg_native, non-god-mode). Pattern lifted from
# scripts/gate_expanded.py::play_game, generalized to a uniform
# `.agent(obs_dict)` interface for both sides and optional trace recording
# for OUR side.
# ---------------------------------------------------------------------------
def play_game(agent_obj, opp_mod, agent_deck, opp_deck, agent_is_left, trace=None):
    """Play one game. agent_is_left: True if our agent controls deck0/left.
    Returns (agent_outcome in {+1,-1,0}, n_moves, exc_side)."""
    if agent_is_left:
        deck0, deck1 = agent_deck, opp_deck
        agent0, agent1 = agent_obj.agent, opp_mod.agent
    else:
        deck0, deck1 = opp_deck, agent_deck
        agent0, agent1 = opp_mod.agent, agent_obj.agent

    obs_dict, start = battle_start(list(deck0), list(deck1))
    if obs_dict is None:
        return 0.0, 0, "load"
    n_moves = 0
    exc_side = None
    winner = None
    while n_moves < MAX_MOVES:
        obs = to_observation_class(obs_dict)
        if obs.current and obs.current.result != -1:
            winner = obs.current.result
            break
        your = obs.current.yourIndex if obs.current else 0
        agent_fn = agent0 if your == 0 else agent1
        is_our_turn = (your == 0 and agent_is_left) or (your == 1 and not agent_is_left)
        sel = obs_dict.get("select")
        try:
            choice = agent_fn(obs_dict)
        except Exception:
            exc_side = "agent" if is_our_turn else "opp"
            winner = 1 - your
            break
        opts = sel.get("option") or [] if sel else []
        if sel is not None:
            n_opt = len(opts)
            mn = int(sel.get("minCount") if sel.get("minCount") is not None else 1)
            mx = int(sel.get("maxCount") if sel.get("maxCount") is not None else 1)
            ok = (isinstance(choice, list) and all(isinstance(i, int) and 0 <= i < n_opt for i in choice)
                  and mn <= len(choice) <= mx and len(set(choice)) == len(choice))
            if not ok:
                exc_side = "agent" if is_our_turn else "opp"
                winner = 1 - your
                break
        if trace is not None and is_our_turn:
            rec = _decision_record(obs, choice, n_moves)
            if rec is not None:
                trace.append(rec)
        try:
            obs_dict = battle_select(choice)
        except Exception:
            exc_side = "agent" if is_our_turn else "opp"
            winner = 1 - your
            break
        n_moves += 1
    try:
        battle_finish()
    except Exception:
        pass
    if winner is None:
        return 0.0, n_moves, exc_side
    if winner == 2:
        return 0.0, n_moves, exc_side
    agent_side = 0 if agent_is_left else 1
    return (1.0 if winner == agent_side else -1.0), n_moves, exc_side


def eval_vs_opponent(agent_obj, opp_name, agent_deck, n_games, seed, trace_games=0):
    opp_dir = OPPONENT_REGISTRY.get(opp_name, opp_name)
    opp_mod = load_submission_module(opp_dir, cache_tag=opp_name)
    opp_deck = load_submission_deck(opp_dir)

    outcomes, moves_list = [], []
    exc_agent = exc_opp = 0
    traces = []
    t0 = time.perf_counter()
    for g in range(n_games):
        agent_is_left = (g % 2 == 0)
        trace = [] if g < trace_games else None
        out, nmoves, exc = play_game(agent_obj, opp_mod, agent_deck, opp_deck, agent_is_left, trace=trace)
        outcomes.append(out)
        moves_list.append(nmoves)
        if exc == "agent":
            exc_agent += 1
        elif exc == "opp":
            exc_opp += 1
        if trace is not None:
            traces.append({"game": g, "agent_is_left": agent_is_left, "outcome": out, "decisions": trace})
    elapsed = time.perf_counter() - t0

    wins = sum(1 for o in outcomes if o > 0)
    losses = sum(1 for o in outcomes if o < 0)
    draws = sum(1 for o in outcomes if o == 0)
    n = len(outcomes)
    we = wins + 0.5 * draws
    wr = we / n if n else float("nan")
    lo, hi = wilson(we, n)
    summary = {
        "opponent": opp_name, "opponent_dir": opp_dir, "n_games": n,
        "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wr, "wilson95_lo": lo, "wilson95_hi": hi,
        "decisive_win_rate": wins / max(wins + losses, 1), "decisive_n": wins + losses,
        "avg_game_length": (sum(moves_list) / max(1, len(moves_list))) if moves_list else 0.0,
        "min_game_length": min(moves_list) if moves_list else 0,
        "max_game_length": max(moves_list) if moves_list else 0,
        "exc_agent": exc_agent, "exc_opp": exc_opp,
        "elapsed_s": elapsed,
    }
    return summary, traces


def build_agent(args):
    if args.ckpt and args.agent_dir:
        raise SystemExit("pass exactly one of --ckpt / --agent-dir")
    if args.ckpt:
        agent_obj = CheckpointAgent(
            args.ckpt, arch_ref_dir=os.path.join(ROOT, args.arch_ref_dir) if not os.path.isabs(args.arch_ref_dir) else args.arch_ref_dir,
            card_dim=args.card_dim, hidden_dim=args.hidden_dim, max_card_id=args.max_card_id,
            max_attack_id=args.max_attack_id, deterministic=args.deterministic,
        )
        deck_path = args.agent_deck
        if deck_path is None:
            raise SystemExit("--ckpt requires --agent-deck (path to a deck.csv)")
        deck_path = deck_path if os.path.isabs(deck_path) else os.path.join(ROOT, deck_path)
        agent_label = f"ckpt:{os.path.basename(args.ckpt)}"
    elif args.agent_dir:
        agent_obj = load_submission_module(args.agent_dir, cache_tag="under_test")
        deck_path = args.agent_deck
        if deck_path is None:
            deck_path = os.path.join(SUBMISSIONS, args.agent_dir, "deck.csv")
        else:
            deck_path = deck_path if os.path.isabs(deck_path) else os.path.join(ROOT, deck_path)
        agent_label = f"submission:{args.agent_dir}"
    else:
        raise SystemExit("must pass --ckpt or --agent-dir")
    agent_deck = load_deck_file(deck_path)
    assert len(agent_deck) == 60, f"agent deck has {len(agent_deck)} cards (expected 60): {deck_path}"
    return agent_obj, agent_deck, deck_path, agent_label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=str, default=None, help="raw PPO checkpoint .pth (packed-ID PtcgPolicyValueNet)")
    ap.add_argument("--agent-dir", type=str, default=None, help="submissions/<dir> name to gate as-is")
    ap.add_argument("--agent-deck", type=str, default=None,
                    help="deck.csv path (required with --ckpt; optional override with --agent-dir)")
    ap.add_argument("--arch-ref-dir", type=str, default=DEFAULT_ARCH_REF_DIR,
                    help="submissions dir supplying cabt_conv.py/nn_plan_d.py for --ckpt mode")
    ap.add_argument("--card-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--max-card-id", type=int, default=1300)
    ap.add_argument("--max-attack-id", type=int, default=4096)
    ap.add_argument("--deterministic", action="store_true", default=True,
                    help="--ckpt mode only: argmax over policy logits (default True)")
    ap.add_argument("--stochastic", dest="deterministic", action="store_false",
                    help="--ckpt mode only: sample instead of argmax")
    ap.add_argument("--opponents", type=str, default="v007,v009a,v033,v036",
                    help="comma-separated short names (OPPONENT_REGISTRY) or raw submissions/<dir> names")
    ap.add_argument("--games", type=int, default=100, help="games PER opponent")
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--trace-games", type=int, default=0,
                    help="save condensed per-decision traces for the first N games of EACH opponent")
    ap.add_argument("--trace-dir", type=str, default=None,
                    help="dir to write trace_<opponent>.json files (forensic_failure_modes.py input)")
    ap.add_argument("--out", type=str, default=None, help="write result JSON here")
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    agent_obj, agent_deck, deck_path, agent_label = build_agent(args)
    print(f"[eval_vs_pool] agent={agent_label} deck={deck_path} ({len(agent_deck)} cards)")

    opp_names = [x.strip() for x in args.opponents.split(",") if x.strip()]
    results = {}
    for name in opp_names:
        print(f"[eval_vs_pool] === {name} === games={args.games}", flush=True)
        summary, traces = eval_vs_opponent(agent_obj, name, agent_deck, args.games, args.seed,
                                           trace_games=args.trace_games)
        results[name] = summary
        print(f"[result] vs {name} n={summary['n_games']} WR={summary['win_rate']:.3f} "
              f"({summary['wins']}W-{summary['losses']}L-{summary['draws']}D) "
              f"Wilson95=[{summary['wilson95_lo']:.3f},{summary['wilson95_hi']:.3f}] "
              f"avg_len={summary['avg_game_length']:.1f} "
              f"exc(agent={summary['exc_agent']},opp={summary['exc_opp']}) "
              f"elapsed={summary['elapsed_s']:.1f}s", flush=True)
        if args.trace_dir and traces:
            os.makedirs(args.trace_dir, exist_ok=True)
            tp = os.path.join(args.trace_dir, f"trace_{name}.json")
            with open(tp, "w") as f:
                json.dump(traces, f, indent=2)
            print(f"[eval_vs_pool] wrote trace {tp} ({len(traces)} games)")

    ckpt_exc = getattr(agent_obj, "n_exceptions", None)
    overall_wins = sum(r["wins"] for r in results.values())
    overall_n = sum(r["n_games"] for r in results.values())
    out_summary = {
        "tag": args.tag,
        "agent": {"ckpt": args.ckpt, "agent_dir": args.agent_dir, "deck": deck_path, "label": agent_label},
        "agent_ckpt_exceptions": ckpt_exc,
        "opponents": results,
        "overall_win_rate": overall_wins / max(1, overall_n),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(out_summary, indent=2))
    if args.out:
        out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out_summary, f, indent=2)
        print(f"[eval_vs_pool] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
