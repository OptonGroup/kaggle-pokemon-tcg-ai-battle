#!/usr/bin/env python3
"""cardemb_agent.py — inference-only agent wrapping the card-embedding BC model
(kiyotah's MyModel encoder/decoder, trained via scripts/train_card_embedding.py,
reused verbatim via scripts/cardemb_common.py).

PURPOSE (PLAN_GOLD "moment of truth"): find out whether the card-embedding model
(val top-1 accuracy 58.3%) plays STRONGER than our LightGBM Crustle champion
(v009a) in real games, when BOTH play the identical Crustle deck (v009a's
deck.csv) — isolates the "model" variable from the "deck" variable.

Model coverage (see scripts/build_card_embedding_data.py docstring): training
data only covered decisions where
  - select.context not in {IS_FIRST(41), MULLIGAN(42)}
  - select.maxCount == 1  (single-choice)
  - 2 <= len(select.option) <= 48
So at inference:
  - IN COVERAGE  -> score every option with the model (decoder action = [i] for
    each option i, batch of n_opts single-option "actions"), pick argmax.
  - OUT OF COVERAGE (deck-request, 0/1 option, multi-select, coin-flip/mulligan
    context) -> deterministic greedy fallback (adapted from
    scripts/h3_gate.py's greedy_meta_agent: attack-max-damage on MAIN context,
    YES on yes/no, max NUMBER on count-select, else first minCount..maxCount
    indices). This never throws and is always legal — matches the "never
    return an illegal move" requirement; it is NOT a smart policy for the
    ~13% of decisions outside training coverage, just crash-proof.

Usage as a submission-style agent:
    from cardemb_agent import agent, CRUSTLE_DECK
    agent(obs_dict) -> list[int]

Counters _n_model_decisions / _n_fallback_decisions / _n_exceptions are exposed
for gate instrumentation (mirrors v009a_crustle's counter pattern).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Guard against the classic macOS "libomp.dylib already initialized" abort/
# segfault when this process ALSO loads LightGBM (v009a's model, in the same
# worker for h2h gates) — both PyTorch and LightGBM link their own OpenMP
# runtime on macOS arm64, and loading both in one process aborts unless this
# is set BEFORE either library is imported. Must be set here, ahead of the
# `import torch` below, and ideally before any other library (lightgbm) has
# been imported elsewhere in the same process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import torch  # noqa: E402

torch.set_num_threads(1)  # one worker process per CPU core already; avoid intra-op thread contention/conflicts

import cardemb_common as cc  # noqa: E402

# ---------------------------------------------------------------------------
# Model + deck loading
# ---------------------------------------------------------------------------
MODEL_PATH = ROOT / "scratchpad" / "gold" / "cardemb_out_v2" / "out" / "model_best.pth"
DECK_PATH = ROOT / "submissions" / "v009a_crustle" / "deck.csv"

# Hyperparameters MUST match training (scripts/train_card_embedding.py defaults,
# confirmed against scratchpad/gold/cardemb_out_v2/out/metrics.json args and the
# actual state_dict tensor shapes: encoder_bag (22000,128), decoder_bag
# (73847,128), one encoder layer, one decoder layer).
D_MODEL = 128
NUM_HEADS = 2
D_FEEDFORWARD = 256
NUM_LAYERS_ENCODER = 1
NUM_LAYERS_DECODER = 1

_SKIP_CONTEXTS = {41, 42}  # IS_FIRST, MULLIGAN — excluded from BC training data
MAX_OPTIONS = 48  # build_card_embedding_data.py's MAX_OPTIONS cap


def _load_deck(path: Path) -> list[int]:
    with open(path) as f:
        ids = [int(x.strip()) for x in f if x.strip()]
    assert len(ids) == 60, f"{path}: expected 60 cards, got {len(ids)}"
    return ids


CRUSTLE_DECK: list[int] = _load_deck(DECK_PATH)

_model: cc.MyModel | None = None
_device = torch.device("cpu")


def _get_model() -> cc.MyModel:
    global _model
    if _model is None:
        m = cc.MyModel(
            D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER,
            dropout=0.0, emb_dropout=0.0,  # irrelevant at eval() time, kept 0 for clarity
        )
        state = torch.load(MODEL_PATH, map_location=_device, weights_only=True)
        m.load_state_dict(state)
        m.eval()
        _model = m
    return _model


def eval_nn(sv_enc: cc.SparseVector, sv_dec: cc.SparseVector, model: cc.MyModel) -> tuple[float, list[float]]:
    """Single-sample forward pass. (kiyotah_cell1.py's eval_nn; not exported by
    cardemb_common.py, so reimplemented here identically.)"""
    device = next(model.parameters()).device
    with torch.inference_mode():
        value, policy = model(
            torch.tensor(sv_enc.index, dtype=torch.int32, device=device),
            torch.tensor(sv_enc.value, dtype=torch.float32, device=device),
            torch.tensor(sv_enc.offset, dtype=torch.int32, device=device),
            torch.tensor(sv_dec.index, dtype=torch.int32, device=device),
            torch.tensor(sv_dec.value, dtype=torch.float32, device=device),
            torch.tensor(sv_dec.offset, dtype=torch.int32, device=device),
        )
    return value.tolist()[0][0], policy.tolist()[0]


# ---------------------------------------------------------------------------
# Fallback greedy pilot for out-of-coverage decisions (adapted from
# scripts/h3_gate.py::greedy_meta_agent — self-contained copy here so this
# module has no import-order dependency on h3_gate, and stays spawn-safe /
# multiprocessing-pickle-safe on its own).
# ---------------------------------------------------------------------------
_OT_YES = 1
_OT_CARD = 3
_OT_PLAY = 7
_OT_ATTACH = 8
_OT_EVOLVE = 9
_OT_ABILITY = 10
_OT_RETREAT = 12
_OT_ATTACK = 13
_OT_END = 14
_ST_YES_NO = 9
_ST_COUNT = 8
_SC_MAIN = 0
_GREEDY_PRIORITY: tuple[int, ...] = (
    _OT_ATTACK, _OT_ABILITY, _OT_EVOLVE, _OT_PLAY, _OT_ATTACH, _OT_RETREAT, _OT_CARD, _OT_END,
)

_ATTACK_DAMAGE_MAP: dict[int, int] | None = None


def _opt_get(o, key: str, default=None):
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


def _get_attack_damage_map() -> dict[int, int]:
    global _ATTACK_DAMAGE_MAP
    if _ATTACK_DAMAGE_MAP is None:
        try:
            from cg.api import all_attack  # type: ignore[import-not-found]
        except ImportError:
            from cg_native.api import all_attack  # noqa: E402
        _ATTACK_DAMAGE_MAP = {a.attackId: (a.damage or 0) for a in all_attack()}
    return _ATTACK_DAMAGE_MAP


def _fallback_agent(obs_dict: dict) -> list[int]:
    """Deterministic, always-legal fallback for decisions outside model coverage."""
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return list(CRUSTLE_DECK)
    opts = sel.get("option") or []
    n = len(opts)
    if n == 0:
        return []
    sel_type = sel.get("type")
    if sel_type == _ST_YES_NO:
        for i in range(n):
            if _opt_get(opts[i], "type") == _OT_YES:
                return [i]
        return [0]
    if sel_type == _ST_COUNT:
        best = max(range(n), key=lambda i: (_opt_get(opts[i], "number", 0) or 0))
        return [best]
    min_c = int(sel.get("minCount") or 1)
    max_c = int(sel.get("maxCount") or 1)
    context = sel.get("context")
    if context == _SC_MAIN:
        atk_idxs = [i for i in range(n) if _opt_get(opts[i], "type") == _OT_ATTACK]
        if atk_idxs:
            dmg_map = _get_attack_damage_map()
            best = max(atk_idxs, key=lambda i: (dmg_map.get(_opt_get(opts[i], "attackId"), 0) or 0))
            return [best]
        for ot in _GREEDY_PRIORITY:
            idxs = [i for i in range(n) if _opt_get(opts[i], "type") == ot]
            if idxs:
                return [idxs[0]]
    k = max(1, min_c)
    k = min(k, max_c, n)
    return list(range(k))


# ---------------------------------------------------------------------------
# Public agent
# ---------------------------------------------------------------------------
_n_model_decisions = 0
_n_fallback_decisions = 0
_n_exceptions = 0


def agent(obs_dict: dict) -> list[int]:
    global _n_model_decisions, _n_fallback_decisions, _n_exceptions

    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return list(CRUSTLE_DECK)

    opts = sel.get("option") or []
    n_opts = len(opts)
    if n_opts == 0:
        _n_fallback_decisions += 1
        return []

    max_count = int(sel.get("maxCount") or 1)
    context = sel.get("context")

    model_eligible = (
        context not in _SKIP_CONTEXTS
        and max_count == 1
        and 2 <= n_opts <= MAX_OPTIONS
    )

    if model_eligible:
        try:
            try:
                from cg.api import to_observation_class  # type: ignore[import-not-found]
            except ImportError:
                from cg_native.api import to_observation_class  # noqa: E402

            obs = to_observation_class(obs_dict)
            model = _get_model()
            actions = [[k] for k in range(n_opts)]
            sv_enc = cc.get_encoder_input(obs, CRUSTLE_DECK)
            sv_dec = cc.get_decoder_input(obs, actions)
            _, policy = eval_nn(sv_enc, sv_dec, model)
            best = max(range(n_opts), key=lambda i: policy[i])
            if 0 <= best < n_opts:
                _n_model_decisions += 1
                return [best]
        except Exception:
            _n_exceptions += 1

    _n_fallback_decisions += 1
    return _fallback_agent(obs_dict)


if __name__ == "__main__":
    # Quick standalone sanity: load model + deck, print param count.
    m = _get_model()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"model loaded OK: {n_params:,} params, deck={len(CRUSTLE_DECK)} cards")
