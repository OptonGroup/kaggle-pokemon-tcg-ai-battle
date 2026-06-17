"""
v009a_crustle — IL agent: Dwebble/Crustle deck + LightGBM ranker
trained on CLEAN-MIRROR data: both players Crustle, winner-side only, MAIN decisions.
Config C6: lr=0.005, num_leaves=15, min_child_samples=50, 300 trees, early-stop=OFF.
Mirror episodes: 336, MAIN winner decisions: 9080.
Offline val MAIN top-1: 0.5921 (v007 reference: 0.5388).

Architecture:
  - Load IL ranker (il_crustle.txt) at import time.
  - Deck: Dwebble/Crustle (60 cards, legal; see deck.csv).
  - On deck-selection step (obs.select is None): return the 60-card Crustle deck.
  - On all other steps: build per-option feature matrix → score with model → pick best.
  - For single-select: return [argmax score].
  - For multi-select: return top-maxCount distinct indices.
  - Never throws: on ANY exception, falls back to a trivial-but-legal selector
    (first minCount..maxCount indices). NO Crustle-specific heuristic exists; the
    fallback is purely crash-proof, not smart.

Counters: _n_model_decisions, _n_trivial_fallback tracked for gate reporting.
"""
from __future__ import annotations

import os
import sys

# ── Path bootstrap — make this directory importable for features.py ──────────
# Kaggle loads agents via exec() which does NOT set __file__, so we must not
# rely on it unconditionally.  Resolution order:
#   1. /kaggle_simulations/agent  (Kaggle runtime — always correct there)
#   2. __file__-based dir          (local / importlib path)
#   3. os.getcwd()                 (last resort fallback)
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

import numpy as np

# ── Instrumentation counters ─────────────────────────────────────────────────
_n_model_decisions: int = 0
_n_heuristic_fallback: int = 0  # kept for gate_runner compatibility (always 0)
_n_trivial_fallback: int = 0
_exception_count: int = 0


def _find_file(name: str) -> str:
    """Search for a file in submission dir, CWD, and Kaggle agent dir."""
    candidates = [
        os.path.join(_DIR, name),
        name,
        os.path.join("/kaggle_simulations/agent", name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(_DIR, name)


# ── Load deck ─────────────────────────────────────────────────────────────────

_deck_path = _find_file("deck.csv")
with open(_deck_path, "r") as _f:
    _csv = [line.strip() for line in _f if line.strip()]
my_deck: list[int] = [int(x) for x in _csv[:60]]
assert len(my_deck) == 60, f"deck.csv has {len(my_deck)} cards (expected 60)"

# ── Load LightGBM model ────────────────────────────────────────────────────────

try:
    import lightgbm as lgb
    _model_path = _find_file("il_crustle.txt")
    _ranker: lgb.Booster | None = lgb.Booster(model_file=_model_path)
except Exception as _load_err:
    _ranker = None
    print(f"[v009a] WARNING: model load failed: {_load_err}", file=sys.stderr)

# ── Feature extraction ────────────────────────────────────────────────────────
# features_v009a.py is co-located; sys.path insert above ensures it's importable.
# Using a versioned module name avoids sys.modules collision when gate_runner
# loads multiple candidates in the same process.

try:
    from features_v009a import build_feature_matrix  # noqa: E402
    _features_ok = True
except Exception as _feat_err:
    _features_ok = False
    print(f"[v009a] WARNING: features_v009a.py import failed: {_feat_err}", file=sys.stderr)


# ── IL agent core ─────────────────────────────────────────────────────────────

def _il_decide(obs_dict: dict) -> list[int]:
    """Score options with the LightGBM ranker and return the best selection."""
    if _ranker is None or not _features_ok:
        raise RuntimeError("model or features unavailable")

    sel = obs_dict.get("select") or {}
    opts = sel.get("option") or []
    n_opts = len(opts)
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)

    if n_opts == 0:
        return []

    X = build_feature_matrix(obs_dict)   # (n_opts, FEATURE_DIM)
    if X.shape[0] != n_opts:
        raise ValueError(f"feature matrix shape mismatch: {X.shape[0]} != {n_opts}")

    scores = _ranker.predict(X)          # (n_opts,)

    order = [int(i) for i in np.argsort(scores)[::-1]]
    order = [i for i in order if 0 <= i < n_opts]

    k = max(min_count, 1)
    k = min(k, max_count)
    k = min(k, n_opts)

    return order[:k]


# ── Public agent function ─────────────────────────────────────────────────────

def agent(obs_dict: dict) -> list[int]:
    """Never-throw public entry point.

    Priority:
      1. IL ranker (model decision) — the Crustle-specific lambdarank model.
      2. Trivial fallback: first minCount indices (always legal, not smart).

    There is NO Crustle-specific heuristic; the fallback is purely crash-proof.
    Any exception in the IL path logs to stderr and falls through.
    """
    global _n_model_decisions, _n_trivial_fallback, _n_heuristic_fallback, _exception_count

    # Deck selection step
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return my_deck

    n_options = len((sel.get("option") or []))
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)

    if n_options == 0:
        _n_trivial_fallback += 1
        return []

    # 1. Try IL model
    try:
        result = _il_decide(obs_dict)
        if result and all(0 <= i < n_options for i in result) and len(result) >= min_count:
            _n_model_decisions += 1
            return result
    except Exception as exc:
        _exception_count += 1
        if _exception_count <= 3:
            print(
                f"[v009a] IL exception #{_exception_count}: {exc}  "
                f"(model={_n_model_decisions} fallback={_n_trivial_fallback})",
                file=sys.stderr,
            )

    # 2. Trivial fallback — always legal
    _n_trivial_fallback += 1
    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))
