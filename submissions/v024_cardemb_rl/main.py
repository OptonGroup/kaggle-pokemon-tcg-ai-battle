"""
v024_cardemb_rl — RL-tuned card-embedding NN agent (kiyotah's MyModel: per-card
EmbeddingBag encoder/decoder + 1-layer Transformer), playing the Crustle deck
(identical to submissions/v009a_crustle/deck.csv).

Live-probe candidate (PLAN_GOLD): this model is more UNIVERSAL across the field
(gate-measured 0.945 WR vs 19 meta decks, vs v009a's 0.843) but LOSES the
Crustle mirror to v009a (~34-38% WR gate-measured). Hypothesis: the live ladder
meta is anti-Crustle-tech-heavy, so the more universal-but-mirror-weaker model
may out-score the sinking v009a (live mu ~733) on the actual ladder. This
submission exists to test that hypothesis with a live probe.

Model checkpoint: model.pth == scratchpad/gold/rl_full/iter2/model_iter2.pth
(RL iter2 checkpoint — stronger than the frozen BC checkpoint in internal
iter-vs-frozen H2H gates). Architecture code: cardemb_model.py (co-located,
verbatim copy of scripts/cardemb_common.py, itself a near-verbatim extraction
of the organizer's kiyotah_cell1.py sample MyModel/encoder code).

Design mirrors submissions/v009a_crustle/main.py and scripts/cardemb_agent.py:
  - __file__-optional path bootstrap: Kaggle loads main.py via exec(), which
    does NOT set __file__ in the module namespace (see project memory:
    kaggle_runtime_no_file). All path resolution below guards against that.
  - Never-throws: any load/inference exception anywhere falls back to a
    deterministic greedy pilot (attack-max-damage / yes / max-count / first-
    legal), and even THAT is wrapped in a final trivial legal fallback. A
    crash forfeits the whole match, so nothing here is allowed to raise.
  - Model is loaded ONCE (lazy singleton, first eligible decision) and reused
    for the rest of the match — NOT reloaded per call.

Counters _n_model_decisions / _n_fallback_decisions / _n_exceptions exposed
for gate/instrumentation parity with v009a_crustle and scripts/cardemb_agent.py.
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# libomp double-init guard — MUST be set before `import torch`. Prevents the
# macOS "libomp.dylib already initialized" abort when torch's OpenMP runtime
# collides with another OpenMP-linked library loaded in the same process
# (e.g. during local dev/gate runs alongside lightgbm-based agents). No-op /
# harmless on Kaggle's Linux runtime — kept for defense in depth.
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---------------------------------------------------------------------------
# Path bootstrap — same pattern as submissions/v009a_crustle/main.py.
# Kaggle's exec() does NOT set __file__; resolution order:
#   1. /kaggle_simulations/agent  (Kaggle runtime — always correct there)
#   2. __file__-based dir          (local importlib / normal script run)
#   3. os.getcwd()                 (last resort — exec() with no __file__)
# ---------------------------------------------------------------------------
_KAGGLE_AGENT_DIR = "/kaggle_simulations/agent"
if os.path.isdir(_KAGGLE_AGENT_DIR):
    _DIR = _KAGGLE_AGENT_DIR
else:
    try:
        _DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _DIR = os.getcwd()
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)


def _find_file(name: str) -> str:
    """Search for a co-located file in submission dir, CWD, and Kaggle agent dir."""
    candidates = [
        os.path.join(_DIR, name),
        name,
        os.path.join("/kaggle_simulations/agent", name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(_DIR, name)


# ---------------------------------------------------------------------------
# Instrumentation counters (parity with v009a_crustle / cardemb_agent.py)
# ---------------------------------------------------------------------------
_n_model_decisions: int = 0
_n_fallback_decisions: int = 0
_n_exceptions: int = 0

# ---------------------------------------------------------------------------
# Load deck — Crustle, identical to v009a_crustle/deck.csv. Unguarded (like
# v009a's main.py): if this fails, the submission is fundamentally broken
# regardless of any try/except, so fail loudly at load time rather than mask it.
# ---------------------------------------------------------------------------
_deck_path = _find_file("deck.csv")
with open(_deck_path, "r") as _f:
    _csv = [line.strip() for line in _f if line.strip()]
my_deck: list[int] = [int(x) for x in _csv[:60]]
assert len(my_deck) == 60, f"deck.csv has {len(my_deck)} cards (expected 60)"
CRUSTLE_DECK = my_deck  # alias, matches scripts/cardemb_agent.py naming

# ---------------------------------------------------------------------------
# Load torch + kiyotah encoders (cardemb_model.py, co-located). Guarded: a
# failure here degrades to the greedy fallback for every decision, rather
# than crashing the whole submission.
# ---------------------------------------------------------------------------
_torch_ok = True
try:
    import torch  # noqa: E402

    torch.set_num_threads(1)  # avoid intra-op thread contention; one process per game
    import cardemb_model as cm  # noqa: E402
except Exception as _imp_err:  # pragma: no cover
    _torch_ok = False
    print(f"[v024] WARNING: torch/cardemb_model import failed: {_imp_err}", file=sys.stderr)

# Hyperparameters MUST match training (scratchpad/gold/rl_full/iter2/metrics.json
# + state_dict tensor shapes: encoder_bag (22000,128), decoder_bag (N,128),
# 1 encoder layer, 1 decoder layer — same architecture as cardemb_out_v2's BC model).
D_MODEL = 128
NUM_HEADS = 2
D_FEEDFORWARD = 256
NUM_LAYERS_ENCODER = 1
NUM_LAYERS_DECODER = 1

_SKIP_CONTEXTS = {41, 42}  # SelectContext.IS_FIRST, MULLIGAN — excluded from BC/RL training data
MAX_OPTIONS = 48  # build_card_embedding_data.py's MAX_OPTIONS cap

_model = None  # lazy singleton — loaded once, first eligible decision
_model_load_failed = False


def _get_model():
    global _model, _model_load_failed
    if _model is not None or _model_load_failed:
        return _model
    try:
        m = cm.MyModel(
            D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER,
            dropout=0.0, emb_dropout=0.0,
        )
        _model_path = _find_file("model.pth")
        state = torch.load(_model_path, map_location="cpu", weights_only=True)
        m.load_state_dict(state)
        m.eval()
        _model = m
    except Exception as _load_err:  # pragma: no cover
        _model_load_failed = True
        print(f"[v024] WARNING: model load failed: {_load_err}", file=sys.stderr)
    return _model


def _eval_nn(sv_enc, sv_dec, model) -> tuple[float, list[float]]:
    """Single-sample forward pass (kiyotah's eval_nn)."""
    with torch.inference_mode():
        value, policy = model(
            torch.tensor(sv_enc.index, dtype=torch.int32),
            torch.tensor(sv_enc.value, dtype=torch.float32),
            torch.tensor(sv_enc.offset, dtype=torch.int32),
            torch.tensor(sv_dec.index, dtype=torch.int32),
            torch.tensor(sv_dec.value, dtype=torch.float32),
            torch.tensor(sv_dec.offset, dtype=torch.int32),
        )
    return value.tolist()[0][0], policy.tolist()[0]


# ---------------------------------------------------------------------------
# Fallback greedy pilot for out-of-coverage decisions AND any exception path
# (self-contained copy of scripts/cardemb_agent.py::_fallback_agent — kept
# independent here so this bundle has no runtime dependency on scripts/).
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
        from cg.api import all_attack  # type: ignore[import-not-found]
        _ATTACK_DAMAGE_MAP = {a.attackId: (a.damage or 0) for a in all_attack()}
    return _ATTACK_DAMAGE_MAP


def _fallback_agent(obs_dict: dict) -> list[int]:
    """Deterministic, always-legal fallback for decisions outside model coverage
    or on any inference exception. NOT a smart policy — crash-proof only."""
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return list(my_deck)
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
        try:
            atk_idxs = [i for i in range(n) if _opt_get(opts[i], "type") == _OT_ATTACK]
            if atk_idxs:
                dmg_map = _get_attack_damage_map()
                best = max(atk_idxs, key=lambda i: (dmg_map.get(_opt_get(opts[i], "attackId"), 0) or 0))
                return [best]
        except Exception:
            pass  # fall through to type-priority / trivial fallback below
        for ot in _GREEDY_PRIORITY:
            idxs = [i for i in range(n) if _opt_get(opts[i], "type") == ot]
            if idxs:
                return [idxs[0]]
    k = max(1, min_c)
    k = min(k, max_c, n)
    return list(range(k))


def _trivial_fallback(obs_dict: dict) -> list[int]:
    """Absolute last resort — no cg.api, no card logic, cannot itself throw
    (barring a malformed obs_dict, which is the caller's contract to avoid)."""
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return list(my_deck)
    opts = sel.get("option") or []
    n = len(opts)
    if n == 0:
        return []
    min_c = int(sel.get("minCount") or 1)
    max_c = int(sel.get("maxCount") or 1)
    k = max(1, min(min_c, max_c, n))
    return list(range(k))


# ---------------------------------------------------------------------------
# Public agent
# ---------------------------------------------------------------------------
def agent(obs_dict: dict) -> list[int]:
    global _n_model_decisions, _n_fallback_decisions, _n_exceptions

    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return list(my_deck)

    opts = sel.get("option") or []
    n_opts = len(opts)
    if n_opts == 0:
        _n_fallback_decisions += 1
        return []

    max_count = int(sel.get("maxCount") or 1)
    context = sel.get("context")

    model_eligible = (
        _torch_ok
        and context not in _SKIP_CONTEXTS
        and max_count == 1
        and 2 <= n_opts <= MAX_OPTIONS
    )

    if model_eligible:
        try:
            from cg.api import to_observation_class  # type: ignore[import-not-found]

            obs = to_observation_class(obs_dict)
            model = _get_model()
            if model is not None:
                actions = [[k] for k in range(n_opts)]
                sv_enc = cm.get_encoder_input(obs, my_deck)
                sv_dec = cm.get_decoder_input(obs, actions)
                _, policy = _eval_nn(sv_enc, sv_dec, model)
                best = max(range(n_opts), key=lambda i: policy[i])
                if 0 <= best < n_opts:
                    _n_model_decisions += 1
                    return [best]
        except Exception:
            _n_exceptions += 1

    _n_fallback_decisions += 1
    try:
        return _fallback_agent(obs_dict)
    except Exception:
        _n_exceptions += 1
        return _trivial_fallback(obs_dict)


if __name__ == "__main__":
    # Quick standalone sanity: load model + deck, print param count.
    _m = _get_model()
    if _m is not None:
        n_params = sum(p.numel() for p in _m.parameters())
        print(f"model loaded OK: {n_params:,} params, deck={len(my_deck)} cards")
    else:
        print("model load FAILED (see stderr warning above)")
