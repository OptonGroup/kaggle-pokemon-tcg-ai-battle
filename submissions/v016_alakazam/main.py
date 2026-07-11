"""
v016_alakazam — IL agent: Alakazam "Powerful Hand" combo deck + LightGBM ranker.
Single-expert BC clone of aidy #8 (teamId 16379469, subId 54064028).
Config C6: lr=0.005, num_leaves=15, min_child_samples=50, 300 trees, early-stop=OFF.
aidy Alakazam episodes: 61, MAIN winner decisions: 2129.
Offline val MAIN top-1: 0.5163 (checkpoint only — NOT a pass/fail gate).

Architecture:
  - Load IL ranker (il_alakazam.txt) at import time.
  - Deck: Alakazam combo (60 cards, legal; see deck.csv).
  - On deck-selection step (obs.select is None): return the 60-card deck.
  - On all other steps:
      1. HARD LETHAL-KO GUARD (anti-whiff, runs BEFORE the IL):
         If my active is Alakazam 743 and an ATTACK option exists whose
         Powerful-Hand damage (20 × my hand size) >= opp active remaining HP,
         take that ATTACK option immediately. Powerful Hand places 2 damage
         counters per card in hand = 20×handN effective damage; damage COUNTERS
         bypass "prevent damage done by attacks" shields and Alakazam is non-ex.
         This guard ensures we never miss a KO — the Iono v010c lesson was that
         BC of a combo deck whiffs the combo; the guard pins the win-con.
      2. IL ranker: build per-option feature matrix (features_alakazam.py) →
         score with model → pick best.
      3. Trivial fallback (never throws): first minCount indices.

Kaggle loads main.py via exec() WITHOUT __file__ → guarded _DIR resolution.
"""
from __future__ import annotations

import os
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Kaggle exec() does NOT set __file__. Resolution order:
#   1. /kaggle_simulations/agent  (Kaggle runtime)
#   2. __file__-based dir          (local / importlib)
#   3. os.getcwd()                 (last resort)
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
_n_ko_guard: int = 0
_n_trivial_fallback: int = 0
_n_heuristic_fallback: int = 0
_exception_count: int = 0

# ── Constants ────────────────────────────────────────────────────────────────
ALAKAZAM_ID = 743
OPT_ATTACK = 13
POWERFUL_HAND_DMG_PER_CARD = 20  # 2 damage counters (=20 dmg) per card in hand


def _find_file(name: str) -> str:
    candidates = [
        os.path.join(_DIR, name),
        name,
        os.path.join("/kaggle_simulations/agent", name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(_DIR, name)


# ── Load deck ────────────────────────────────────────────────────────────────
_deck_path = _find_file("deck.csv")
with open(_deck_path, "r") as _f:
    _csv = [line.strip() for line in _f if line.strip()]
my_deck: list[int] = [int(x) for x in _csv[:60]]
assert len(my_deck) == 60, f"deck.csv has {len(my_deck)} cards (expected 60)"

# ── Load LightGBM model ──────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    _model_path = _find_file("il_alakazam.txt")
    _ranker: lgb.Booster | None = lgb.Booster(model_file=_model_path)
except Exception as _load_err:
    _ranker = None
    print(f"[v016] WARNING: model load failed: {_load_err}", file=sys.stderr)

# ── Feature extraction ───────────────────────────────────────────────────────
try:
    from features_alakazam import build_feature_matrix  # noqa: E402
    _features_ok = True
except Exception as _feat_err:
    _features_ok = False
    print(f"[v016] WARNING: features_alakazam.py import failed: {_feat_err}", file=sys.stderr)


# ── Lethal-KO guard ──────────────────────────────────────────────────────────
def _ko_guard(obs_dict: dict) -> list[int] | None:
    """If my active is Alakazam 743 and a Powerful-Hand attack would KO,
    return that option index immediately. Else None.

    Powerful Hand: 2 damage counters per card in hand = 20×hand_size effective
    damage. Damage counters bypass "prevent damage" shields. This guard pins
    the win-con so BC never whiffs the lethal.
    """
    try:
        sel = obs_dict.get("select") or {}
        opts = sel.get("option") or []
        if not opts:
            return None
        cur = obs_dict.get("current") or {}
        me = int(cur.get("yourIndex") or 0)
        players = cur.get("players") or [{}, {}]
        while len(players) < 2:
            players = players + [{}]
        my_pl = players[me]
        opp_pl = players[1 - me]
        my_active = (my_pl.get("active") or [None])[0]
        opp_active = (opp_pl.get("active") or [None])[0]
        if my_active is None or opp_active is None:
            return None
        if int(my_active.get("id") or 0) != ALAKAZAM_ID:
            return None
        hand_size = int(my_pl.get("handCount") or 0)
        opp_hp = int(opp_active.get("hp") or 0)
        if hand_size <= 0 or opp_hp <= 0:
            return None
        est_dmg = POWERFUL_HAND_DMG_PER_CARD * hand_size
        if est_dmg < opp_hp:
            return None
        # find an ATTACK option (Alakazam's only attack is Powerful Hand)
        for i, opt in enumerate(opts):
            if int(opt.get("type") or 0) == OPT_ATTACK:
                return [i]
        return None
    except Exception:
        return None


# ── IL decision ──────────────────────────────────────────────────────────────
def _il_decide(obs_dict: dict) -> list[int]:
    if _ranker is None or not _features_ok:
        raise RuntimeError("model or features unavailable")
    sel = obs_dict.get("select") or {}
    opts = sel.get("option") or []
    n_opts = len(opts)
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)
    if n_opts == 0:
        return []
    X = build_feature_matrix(obs_dict)
    if X.shape[0] != n_opts:
        raise ValueError(f"feature matrix shape mismatch: {X.shape[0]} != {n_opts}")
    scores = _ranker.predict(X)
    order = [int(i) for i in np.argsort(scores)[::-1]]
    order = [i for i in order if 0 <= i < n_opts]
    k = max(min_count, 1)
    k = min(k, max_count, n_opts)
    return order[:k]


# ── Public agent ─────────────────────────────────────────────────────────────
def agent(obs_dict: dict) -> list[int]:
    """Never-throw public entry point.

    Priority:
      1. Lethal-KO guard (Alakazam Powerful-Hand lethal → take it).
      2. IL ranker.
      3. Trivial fallback (first minCount indices).
    """
    global _n_model_decisions, _n_ko_guard, _n_trivial_fallback
    global _n_heuristic_fallback, _exception_count

    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return my_deck

    n_options = len((sel.get("option") or []))
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)

    if n_options == 0:
        _n_trivial_fallback += 1
        return []

    # 1. Lethal-KO guard
    try:
        guard = _ko_guard(obs_dict)
        if guard is not None and all(0 <= i < n_options for i in guard):
            _n_ko_guard += 1
            return guard
    except Exception:
        pass

    # 2. IL model
    try:
        result = _il_decide(obs_dict)
        if result and all(0 <= i < n_options for i in result) and len(result) >= min_count:
            _n_model_decisions += 1
            return result
    except Exception as exc:
        _exception_count += 1
        if _exception_count <= 3:
            print(
                f"[v016] IL exception #{_exception_count}: {exc}  "
                f"(model={_n_model_decisions} ko={_n_ko_guard} "
                f"fallback={_n_trivial_fallback})",
                file=sys.stderr,
            )

    # 3. Trivial fallback — always legal
    _n_trivial_fallback += 1
    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))