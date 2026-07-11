"""
Pilot IL agent — сгенерировано src/exit/pilot.py:export_to_submission().
LightGBM lambdarank policy prior, обученный на выигранных партиях (quality > volume).
"""
from __future__ import annotations

import os
import sys

# ── Path bootstrap — __file__ guard для Kaggle exec() ────────────────────────
# Kaggle загружает main.py через exec() без __file__. Порядок резолюции:
#   1. /kaggle_simulations/agent  (Kaggle runtime — всегда верно там)
#   2. __file__-based dir          (local / importlib путь)
#   3. os.getcwd()                 (последний резерв)
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

# ── Счётчики для диагностики ─────────────────────────────────────────────────
_n_model_decisions: int = 0
_n_trivial_fallback: int = 0
_n_heuristic_fallback: int = 0  # совместимость с gate_runner
_exception_count: int = 0


def _find_file(name: str) -> str:
    """Ищет файл в директории сабмита, CWD и в /kaggle_simulations/agent."""
    candidates = [
        os.path.join(_DIR, name),
        name,
        os.path.join("/kaggle_simulations/agent", name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(_DIR, name)


# ── Дек ──────────────────────────────────────────────────────────────────────
_deck_path = _find_file("deck.csv")
with open(_deck_path, "r") as _f:
    _csv_lines = [ln.strip() for ln in _f if ln.strip()]
my_deck: list[int] = [int(x) for x in _csv_lines[:60]]
assert len(my_deck) == 60, f"deck.csv: {len(my_deck)} карт (ожидалось 60)"

# ── LightGBM модель ───────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    _model_path = _find_file("model.txt")
    _ranker: lgb.Booster | None = lgb.Booster(model_file=_model_path)
except Exception as _load_err:
    _ranker = None
    print(f"[pilot] WARNING: не удалось загрузить модель: {_load_err}", file=sys.stderr)

# ── Фичи ─────────────────────────────────────────────────────────────────────
# features_iono.py лежит co-located; sys.path уже содержит _DIR.
try:
    from features_iono import build_feature_matrix
    _features_ok = True
except Exception as _feat_err:
    _features_ok = False
    print(f"[pilot] WARNING: импорт features_iono.py не удался: {_feat_err}", file=sys.stderr)


# ── Core scorer ──────────────────────────────────────────────────────────────

def _il_decide(obs_dict: dict) -> list[int]:
    """Ранжирует опции LightGBM и возвращает лучший выбор."""
    if _ranker is None or not _features_ok:
        raise RuntimeError("модель или фичи недоступны")

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

    scores = _ranker.predict(X.astype(np.float32))
    order = [int(i) for i in np.argsort(scores)[::-1]]
    order = [i for i in order if 0 <= i < n_opts]

    k = max(min_count, 1)
    k = min(k, max_count, n_opts)
    return order[:k]


# ── Публичная точка входа ─────────────────────────────────────────────────────

def agent(obs_dict: dict) -> list[int]:
    """Never-throw entry point.

    1. Deck-selection (select is None) → my_deck.
    2. IL model → best-ranked option(s).
    3. Trivial fallback (первые k индексов) — если модель упала.
    """
    global _n_model_decisions, _n_trivial_fallback, _n_heuristic_fallback, _exception_count

    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return my_deck

    n_options = len((sel.get("option") or []))
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)

    if n_options == 0:
        _n_trivial_fallback += 1
        return []

    # 1. IL model
    try:
        result = _il_decide(obs_dict)
        if result and all(0 <= i < n_options for i in result) and len(result) >= min_count:
            _n_model_decisions += 1
            return result
    except Exception as exc:
        _exception_count += 1
        if _exception_count <= 3:
            print(
                f"[pilot] exception #{_exception_count}: {exc}  "
                f"(model={_n_model_decisions} fallback={_n_trivial_fallback})",
                file=sys.stderr,
            )

    # 2. Trivial fallback — always legal
    _n_trivial_fallback += 1
    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))
