#!/usr/bin/env python3
"""
linear_value.py — Stable linear-probe value function for MCTS/ExIt leaf eval.

The NN value-head (card-embedding MyModel encoder -> tanh scalar) is seed-fragile:
2/3 training seeds pass (AUC ~0.68), 1/3 fails (AUC ~0.60). The cheap 11-feature
linear probe (LogisticRegression on board features) is DETERMINISTIC and reproduces
AUC ~0.68 on held-out episodes, so it is a safer leaf evaluator for ExIt.

This module:
  - TRAINS (in __main__): extracts decision states from data/episodes/all_teams
    replays via value_probe.collect_states, fits LogisticRegression on the 11
    board features, measures group-aware ROC-AUC, and saves the fitted model to
    models/linear_value_probe.joblib.
  - SERVES (as a library): exposes LinearValuePredictor with predict_value(obs)
    -> float in [-1,1] (logistic proba p mapped to 2p-1, matching the NN value
    head's tanh output range). Works on BOTH the dict obs (replay JSON shape)
    and the cg_native Observation dataclass (leaf-eval shape) via duck typing.

Output contract: predict_value returns a scalar in [-1,1], +1 = good for the
player-to-move (obs.current.yourIndex), -1 = bad. This is the SAME convention as
the NN value head (eval_nn returns value in [-1,1] from the yourIndex POV; the
MCTS code flips sign when state.yourIndex != your_index).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = REPO / "models" / "linear_value_probe.joblib"
# Enriched LightGBM value model (preferred over the linear probe when present).
GBDT_ARTIFACT_PATH = REPO / "models" / "value_gbdt.joblib"

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


# ---------------------------------------------------------------------------
# Feature extraction (identical semantics to value_probe.extract_features_and_label,
# but works on BOTH dict obs and cg_native dataclass obs via duck typing).
# ---------------------------------------------------------------------------
def _get(obj, name, default=None):
    """dict.get OR attribute access."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _active_pokemon(pl):
    act = _get(pl, "active") or []
    if act and act[0] is not None:
        return act[0]
    return None


def _active_hp(pl):
    a = _active_pokemon(pl)
    if a is not None:
        v = _get(a, "hp", 0)
        return v or 0
    return 0


def _active_maxhp(pl):
    a = _active_pokemon(pl)
    if a is not None:
        v = _get(a, "maxHp", 1)
        return v or 1
    return 1


def _total_energy(pl):
    tot = 0
    a = _active_pokemon(pl)
    if a is not None:
        e = _get(a, "energies") or []
        tot += len(e)
    bench = _get(pl, "bench") or []
    for b in bench:
        if b is not None:
            e = _get(b, "energies") or []
            tot += len(e)
    return tot


def _n_evolved(pl):
    n = 0
    a = _active_pokemon(pl)
    if a is not None:
        pre = _get(a, "preEvolution") or []
        n += len(pre)
    bench = _get(pl, "bench") or []
    for b in bench:
        if b is not None:
            pre = _get(b, "preEvolution") or []
            n += len(pre)
    return n


def _can_attack(pl):
    a = _active_pokemon(pl)
    if a is not None:
        e = _get(a, "energies") or []
        return 1.0 if len(e) > 0 else 0.0
    return 0.0


def _len(x):
    return len(x) if x is not None else 0


def extract_features(obs) -> list[float] | None:
    """Return the 11-feature board vector from the player-to-move's POV, or None
    if the state is pre-game / has no active pokemon / not a real decision state.

    `obs` may be:
      - a dict with obs["current"]["players"][...]  (replay JSON shape), OR
      - a cg_native.api.Observation dataclass with obs.current.players[...]  (leaf shape)
    """
    cur = _get(obs, "current")
    if cur is None:
        return None
    yi = _get(cur, "yourIndex", 0) or 0
    turn = _get(cur, "turn", 0) or 0
    first_player = _get(cur, "firstPlayer", -1)
    if first_player is None:
        first_player = -1
    ps = _get(cur, "players") or []
    if len(ps) < 2:
        return None
    me = ps[yi]
    opp = ps[1 - yi]

    my_hp = _active_hp(me)
    opp_hp = _active_hp(opp)
    my_maxhp = _active_maxhp(me)
    opp_maxhp = _active_maxhp(opp)

    feats = [
        (my_hp - opp_hp) / 200.0,
        (my_hp / max(my_maxhp, 1)) - (opp_hp / max(opp_maxhp, 1)),
        _len(_get(me, "prize")) - _len(_get(opp, "prize")),
        _len(_get(me, "bench")) - _len(_get(opp, "bench")),
        (_get(me, "handCount", 0) or 0) - (_get(opp, "handCount", 0) or 0),
        turn / 20.0,
        _total_energy(me) - _total_energy(opp),
        _n_evolved(me) - _n_evolved(opp),
        _len(_get(me, "discard")) - _len(_get(opp, "discard")),
        1.0 if (first_player == yi) else 0.0,
        _can_attack(me) - _can_attack(opp),
    ]
    return feats


# ---------------------------------------------------------------------------
# ENRICHED feature extractor (~35 board features) for the LightGBM value model.
# Builds on the 11 original features by adding absolute board quantities,
# damage counters, bench HP / attacker counts, stage-2 count, deck remaining,
# active energy, and the active's energy-to-attack shortfall from select.
# Works on BOTH dict obs and cg_native dataclass obs (same duck-typing helpers).
# ---------------------------------------------------------------------------
def _bench(pl):
    b = _get(pl, "bench")
    return b if b else []


def _active_energy_count(pl):
    a = _active_pokemon(pl)
    if a is None:
        return 0
    e = _get(a, "energies") or []
    return len(e)


def _bench_energy_count(pl):
    tot = 0
    for b in _bench(pl):
        if b is None:
            continue
        e = _get(b, "energies") or []
        tot += len(e)
    return tot


def _bench_hp_sum(pl):
    s = 0
    for b in _bench(pl):
        if b is None:
            continue
        hp = _get(b, "hp", 0) or 0
        s += hp
    return s


def _bench_attackers(pl):
    """Benched pokemon that can attack (proxy: has >0 energy attached)."""
    n = 0
    for b in _bench(pl):
        if b is None:
            continue
        e = _get(b, "energies") or []
        if len(e) > 0:
            n += 1
    return n


def _n_stage2(pl):
    """Pokemon whose preEvolution chain length >= 2 (Stage-2 evolved)."""
    n = 0
    a = _active_pokemon(pl)
    if a is not None:
        pre = _get(a, "preEvolution") or []
        if len(pre) >= 2:
            n += 1
    for b in _bench(pl):
        if b is None:
            continue
        pre = _get(b, "preEvolution") or []
        if len(pre) >= 2:
            n += 1
    return n


def _active_damage(pl):
    """Damage counters on active = maxHp - hp (0 if none)."""
    a = _active_pokemon(pl)
    if a is None:
        return 0
    hp = _get(a, "hp", 0) or 0
    maxhp = _get(a, "maxHp", 1) or 1
    return max(0, maxhp - hp)


def _attack_energy_shortfall(obs, cur):
    """Energy still needed for the active to attack, from the player-to-move's
    POV. Uses select.remainEnergyCost when the pending decision is an attack
    (or any context that exposes it); 0 means ready. Returns 0 if unavailable
    (pre-game, non-attack context). Only defined for the player-to-move."""
    sel = _get(obs, "select")
    if sel is None:
        return 0
    rec = _get(sel, "remainEnergyCost", None)
    if rec is None:
        return 0
    try:
        return float(rec or 0)
    except (TypeError, ValueError):
        return 0.0


def extract_features_rich(obs) -> list[float] | None:
    """Return the enriched ~35-feature board vector from the player-to-move's
    POV, or None if the state is pre-game / has no active pokemon.

    Same obs shapes as extract_features (dict OR cg_native dataclass). Falls
    back to 0 for any missing field, so it never crashes on partial obs.
    """
    cur = _get(obs, "current")
    if cur is None:
        return None
    yi = _get(cur, "yourIndex", 0) or 0
    turn = _get(cur, "turn", 0) or 0
    first_player = _get(cur, "firstPlayer", -1)
    if first_player is None:
        first_player = -1
    ps = _get(cur, "players") or []
    if len(ps) < 2:
        return None
    me = ps[yi]
    opp = ps[1 - yi]

    my_hp = _active_hp(me)
    opp_hp = _active_hp(opp)
    my_maxhp = _active_maxhp(me)
    opp_maxhp = _active_maxhp(opp)
    my_hp_frac = my_hp / max(my_maxhp, 1)
    opp_hp_frac = opp_hp / max(opp_maxhp, 1)

    my_active_e = _active_energy_count(me)
    opp_active_e = _active_energy_count(opp)
    my_total_e = _total_energy(me)
    opp_total_e = _total_energy(opp)

    my_bench = _len(_get(me, "bench"))
    opp_bench = _len(_get(opp, "bench"))
    my_bench_hp = _bench_hp_sum(me)
    opp_bench_hp = _bench_hp_sum(opp)
    my_bench_atk = _bench_attackers(me)
    opp_bench_atk = _bench_attackers(opp)

    my_evolved = _n_evolved(me)
    opp_evolved = _n_evolved(opp)
    my_stage2 = _n_stage2(me)
    opp_stage2 = _n_stage2(opp)

    my_prize = _len(_get(me, "prize"))
    opp_prize = _len(_get(opp, "prize"))
    my_discard = _len(_get(me, "discard"))
    opp_discard = _len(_get(opp, "discard"))
    my_deck = _get(me, "deckCount", 0) or 0
    opp_deck = _get(opp, "deckCount", 0) or 0
    my_hand = _get(me, "handCount", 0) or 0
    opp_hand = _get(opp, "handCount", 0) or 0

    my_can_atk = _can_attack(me)
    opp_can_atk = _can_attack(opp)
    my_dmg = _active_damage(me)
    opp_dmg = _active_damage(opp)
    atk_shortfall = _attack_energy_shortfall(obs, cur)

    feats = [
        # --- active HP (absolute + frac + diff) ---
        my_hp / 200.0,
        opp_hp / 200.0,
        my_hp_frac,
        opp_hp_frac,
        (my_hp - opp_hp) / 200.0,
        my_hp_frac - opp_hp_frac,
        # --- damage counters on active ---
        my_dmg / 200.0,
        opp_dmg / 200.0,
        # --- energy (active + total + diff) ---
        my_active_e,
        opp_active_e,
        my_total_e,
        opp_total_e,
        my_total_e - opp_total_e,
        # --- bench (count + hp + attackers + diff) ---
        my_bench,
        opp_bench,
        my_bench - opp_bench,
        my_bench_hp / 200.0,
        opp_bench_hp / 200.0,
        my_bench_atk,
        opp_bench_atk,
        # --- evolution (stage1+ evolved, stage2, diff) ---
        my_evolved,
        opp_evolved,
        my_evolved - opp_evolved,
        my_stage2,
        opp_stage2,
        # --- prizes (absolute + diff; more prizes left = behind) ---
        my_prize,
        opp_prize,
        my_prize - opp_prize,
        # --- discard / deck / hand ---
        my_discard,
        opp_discard,
        my_discard - opp_discard,
        my_deck / 60.0,
        opp_deck / 60.0,
        my_hand,
        opp_hand,
        my_hand - opp_hand,
        # --- turn (linear + squared) ---
        turn / 20.0,
        (turn * turn) / 400.0,
        # --- first-player flag ---
        1.0 if (first_player == yi) else 0.0,
        # --- can-attack proxy (absolute + diff) ---
        my_can_atk,
        opp_can_atk,
        my_can_atk - opp_can_atk,
        # --- attack energy shortfall (player-to-move only) ---
        atk_shortfall,
    ]
    return feats


# Feature name list (for inspection / feature importance reporting).
RICH_FEATURE_NAMES = [
    "my_active_hp", "opp_active_hp", "my_active_hp_frac", "opp_active_hp_frac",
    "active_hp_diff", "active_hp_frac_diff",
    "my_active_damage", "opp_active_damage",
    "my_active_energy", "opp_active_energy",
    "my_total_energy", "opp_total_energy", "energy_diff",
    "my_bench_count", "opp_bench_count", "bench_diff",
    "my_bench_hp", "opp_bench_hp", "my_bench_attackers", "opp_bench_attackers",
    "my_evolved", "opp_evolved", "evolved_diff",
    "my_stage2", "opp_stage2",
    "my_prize", "opp_prize", "prize_rem_diff",
    "my_discard", "opp_discard", "discard_diff",
    "my_deck", "opp_deck",
    "my_hand", "opp_hand", "hand_diff",
    "turn", "turn_sq",
    "is_first",
    "my_can_attack", "opp_can_attack", "can_attack_diff",
    "attack_energy_shortfall",
]


# ---------------------------------------------------------------------------
# Predictor: loads the joblib artifact, exposes predict_value(obs) -> float in [-1,1]
# ---------------------------------------------------------------------------
class LinearValuePredictor:
    """Leaf-eval value predictor.

    Prefers the enriched LightGBM (models/value_gbdt.joblib) when present, with
    fallback to the linear probe (models/linear_value_probe.joblib). Both expose
    sklearn's predict_proba([...]) -> P(win). The LightGBM uses extract_features_rich
    (~35 features); the linear probe uses the original 11-feature extract_features.

    predict_value(obs) -> float in [-1,1] via 2*p-1 (same contract as the NN value
    head's tanh range). Works on dict obs and cg_native dataclass obs.
    """

    def __init__(self, model=None, artifact_path: str | os.PathLike = ARTIFACT_PATH,
                 gbdt_path: str | os.PathLike = GBDT_ARTIFACT_PATH):
        self.n_calls = 0  # leaf-eval fire counter (smoke instrumentation)
        self.use_gbdt = False
        if model is not None:
            # Caller injected a model directly — assume it pairs with the rich
            # feature extractor (LightGBM) for backward-compatible tests.
            self.model = model
            self.use_gbdt = True
        else:
            import joblib
            # Prefer LightGBM; fall back to linear probe.
            if os.path.exists(str(gbdt_path)):
                self.model = joblib.load(str(gbdt_path))
                self.use_gbdt = True
            else:
                self.model = joblib.load(str(artifact_path))
                self.use_gbdt = False

    def predict_proba(self, obs) -> float:
        """P(win | obs) in [0,1] from the player-to-move's POV."""
        if self.use_gbdt:
            feats = extract_features_rich(obs)
        else:
            feats = extract_features(obs)
        if feats is None:
            return 0.5  # uninformative prior for pre-game / non-decision states
        import warnings
        import numpy as np
        # LGBMClassifier / LogisticRegression both implement predict_proba.
        arr = np.asarray(feats, dtype=float).reshape(1, -1)
        # LGBM auto-stores Column_N feature names at fit time and warns when
        # predict is called on a nameless array; the warning is cosmetic (the
        # values are positional). Suppress it to keep leaf-eval logs clean.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(self.model.predict_proba(arr)[0, 1])
        # Guard against numerical leakage outside [0,1].
        if p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0
        return p

    def predict_value(self, obs) -> float:
        """Value in [-1,1] from the player-to-move's POV (+1 = winning).
        Maps logistic proba p to 2p-1, matching the NN value head's tanh range."""
        self.n_calls += 1
        p = self.predict_proba(obs)
        return 2.0 * p - 1.0


# Module-level singleton (lazy) so mcts_rl_lib can call predict_value without
# plumbing an object through every worker if it prefers.
_SINGLETON = None


def get_predictor() -> LinearValuePredictor:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = LinearValuePredictor()
    return _SINGLETON


def predict_value(obs) -> float:
    """Convenience: predict_value via the module-level singleton."""
    return get_predictor().predict_value(obs)


# ---------------------------------------------------------------------------
# TRAIN (run as: .venv/bin/python scripts/linear_value.py)
# ---------------------------------------------------------------------------
def train_and_save(max_games: int = 600, seed: int = 0,
                   artifact_path: str | os.PathLike = ARTIFACT_PATH):
    import glob
    import hashlib
    import random

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    import joblib

    # Reuse value_probe.collect_states for the EXACT same state set the probe used.
    import sys
    scripts_dir = str(REPO / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import value_probe as vp

    replay_paths = sorted(glob.glob(
        str(REPO / "data" / "episodes" / "all_teams" / "*-replay.json")))
    print(f"[linear_value] found {len(replay_paths)} replay files")
    rows, _vhead = vp.collect_states(replay_paths, max_games=max_games, seed=seed)
    if not rows:
        raise RuntimeError("no decision states collected from replays")
    X = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=int)
    groups = np.array([vp._stable_group_id(r[2]) for r in rows], dtype=int)
    print(f"[linear_value] {len(rows)} decision states; pos fraction={y.mean():.3f}")

    # Group-aware held-out split for an honest AUC
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    tr, te = next(gss.split(X, y, groups=groups))
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X[tr], y[tr])
    proba_te = clf.predict_proba(X[te])[:, 1]
    holdout_auc = roc_auc_score(y[te], proba_te)
    print(f"[linear_value] held-out AUC (group-aware) = {holdout_auc:.4f}  (n_test={len(te)})")

    # 5-fold group-aware CV AUC for a stabler number
    aucs = []
    gss2 = GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
    for tr2, te2 in gss2.split(X, y, groups=groups):
        c2 = LogisticRegression(max_iter=2000, C=1.0)
        c2.fit(X[tr2], y[tr2])
        aucs.append(roc_auc_score(y[te2], c2.predict_proba(X[te2])[:, 1]))
    cv_auc = float(np.mean(aucs))
    print(f"[linear_value] 5-fold group-aware AUC = {cv_auc:.4f} +/- {np.std(aucs):.4f}")

    # Fit final model on ALL data and save
    final = LogisticRegression(max_iter=2000, C=1.0)
    final.fit(X, y)
    os.makedirs(os.path.dirname(str(artifact_path)), exist_ok=True)
    joblib.dump(final, str(artifact_path))
    print(f"[linear_value] saved artifact -> {artifact_path}")

    # Sanity: the saved model reproduces holdout AUC on the same split
    saved = joblib.load(str(artifact_path))
    preds_saved = saved.predict_proba(X[te])[:, 1]
    # note: final is trained on ALL data so its holdout numbers are optimistic;
    # report the honest clf holdout AUC as THE number.
    return {
        "holdout_auc": float(holdout_auc),
        "cv_auc": cv_auc,
        "n_states": len(rows),
        "pos_fraction": float(y.mean()),
        "artifact_path": str(artifact_path),
    }


def train_gbdt(max_games: int = 1000, seed: int = 0,
               gbdt_path: str | os.PathLike = GBDT_ARTIFACT_PATH):
    """Train a LightGBM binary value classifier on the ENRICHED feature set,
    with group-aware (by episode) evaluation. Also refits a linear
    LogisticRegression on the same enriched features for comparison.

    Saves the LightGBM to models/value_gbdt.joblib. Reports:
      - linear (11-feat) AUC  (from value_probe collect_states)
      - linear-rich AUC       (LogisticRegression on the 35 enriched feats)
      - LightGBM AUC          (5-fold group-aware + single holdout)
    """
    import glob
    import json
    import random

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    import joblib

    import sys
    scripts_dir = str(REPO / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import value_probe as vp

    replay_paths = sorted(glob.glob(
        str(REPO / "data" / "episodes" / "all_teams" / "*-replay.json")))
    print(f"[gbdt] found {len(replay_paths)} replay files")
    rows, _vhead = vp.collect_states(replay_paths, max_games=max_games, seed=seed)
    if not rows:
        raise RuntimeError("no decision states collected from replays")

    # Re-extract the RICH features from the SAME (obs, won) states collect_states
    # saw. rows[i] = (linear11_feats, won, ep_id) but we kept the obs via a
    # second pass: collect_states also returns vhead_inputs with obs dicts.
    # Re-run the same path selection deterministically to align obs with rows.
    rng = random.Random(seed)
    paths = list(replay_paths)
    rng.shuffle(paths)
    paths = paths[:max_games]

    # Rebuild (obs, won, ep_id) pairs mirroring collect_states' iteration order
    # so each row's rich features come from the exact obs that produced the row.
    obs_rows = []  # parallel to rows: (obs_dict, won, ep_id)
    for p in paths:
        try:
            data = json.load(open(p))
        except Exception:
            continue
        rewards = data.get("rewards", [])
        agents = data.get("info", {}).get("Agents", [])
        steps = data.get("steps", [])
        n_players = len(agents)
        won_by_player = {}
        for i in range(min(n_players, len(rewards))):
            r = rewards[i]
            won_by_player[i] = (r is not None and r > 0)
        if not any(won_by_player.values()):
            continue
        ep_id = data.get("id") or data.get("info", {}).get("EpisodeId", p)
        for i in range(len(steps) - 1):
            for pi in range(min(n_players, 2)):
                obs = steps[i][pi].get("observation", {})
                sel = obs.get("select")
                cur = obs.get("current")
                if sel is None or cur is None:
                    continue
                ctx = sel.get("context", -1)
                if ctx in (41, 42):
                    continue
                if cur.get("turn", 0) < 1:
                    continue
                nxt = steps[i + 1]
                if pi >= len(nxt):
                    continue
                action = nxt[pi].get("action", [])
                if not action or len(action) == 60:
                    continue
                won = won_by_player.get(pi, False)
                obs_rows.append((obs, int(won), ep_id))

    if len(obs_rows) != len(rows):
        # Fallback: if alignment drifts (e.g. a row filtered by extract returning
        # None), rebuild rich features only where extract_features_rich is not
        # None, mirroring the probe's own filter.
        print(f"[gbdt] WARN: obs_rows={len(obs_rows)} != rows={len(rows)}; "
              f"filtering via extract_features_rich non-None")
        rich_feats = []
        ys = []
        eps = []
        for (obs, won, ep_id) in obs_rows:
            f = extract_features_rich(obs)
            if f is None:
                continue
            rich_feats.append(f)
            ys.append(won)
            eps.append(ep_id)
    else:
        rich_feats = []
        ys = []
        eps = []
        for (obs, won, ep_id), row in zip(obs_rows, rows):
            f = extract_features_rich(obs)
            if f is None:
                # Fall back to the row's linear feats padded? No — skip to match
                # the probe filter exactly. (Should be rare.)
                continue
            rich_feats.append(f)
            ys.append(won)
            eps.append(ep_id)

    X = np.asarray(rich_feats, dtype=float)
    y = np.asarray(ys, dtype=int)
    groups = np.array([vp._stable_group_id(e) for e in eps], dtype=int)
    # Replace any non-finite (NaN/inf) with 0 so LGBM never chokes.
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        print(f"[gbdt] replacing {int(nan_mask.sum())} non-finite entries with 0")
        X = np.where(nan_mask, 0.0, X)
    print(f"[gbdt] {len(rich_feats)} enriched decision states "
          f"({X.shape[1]} features); pos fraction={y.mean():.3f}")

    # --- linear (11-feat) baseline AUC, from the probe's own rows ---
    X_lin = np.array([r[0] for r in rows], dtype=float)
    y_lin = np.array([r[1] for r in rows], dtype=int)
    g_lin = np.array([vp._stable_group_id(r[2]) for r in rows], dtype=int)
    lin_aucs = []
    gss_lin = GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
    for tr, te in gss_lin.split(X_lin, y_lin, groups=g_lin):
        c = LogisticRegression(max_iter=2000, C=1.0)
        c.fit(X_lin[tr], y_lin[tr])
        lin_aucs.append(roc_auc_score(y_lin[te], c.predict_proba(X_lin[te])[:, 1]))
    print(f"[gbdt] linear-11  5-fold group-aware AUC = "
          f"{np.mean(lin_aucs):.4f} +/- {np.std(lin_aucs):.4f}")

    # --- linear on enriched features ---
    lin_rich_aucs = []
    gss_r = GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
    for tr, te in gss_r.split(X, y, groups=groups):
        c = LogisticRegression(max_iter=4000, C=1.0)
        c.fit(X[tr], y[tr])
        lin_rich_aucs.append(roc_auc_score(y[te], c.predict_proba(X[te])[:, 1]))
    print(f"[gbdt] linear-rich 5-fold group-aware AUC = "
          f"{np.mean(lin_rich_aucs):.4f} +/- {np.std(lin_rich_aucs):.4f}")

    # --- LightGBM (enriched) with group-aware early stopping ---
    # Single group-aware holdout split for honest reporting + early stopping.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    tr, te = next(gss.split(X, y, groups=groups))
    # Carve a small group-aware validation set out of the training half for
    # early stopping (so the holdout `te` stays untouched).
    gss_es = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    tr_tr, tr_va = next(gss_es.split(X[tr], y[tr], groups=groups[tr]))

    best_params = dict(
        n_estimators=2000,
        num_leaves=31,
        learning_rate=0.03,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        objective="binary",
        metric="auc",
        random_state=0,
        n_jobs=1,
        verbose=-1,
    )
    print(f"[gbdt] params: {best_params}")
    gbdt = LGBMClassifier(**best_params)
    gbdt.fit(
        X[tr][tr_tr], y[tr][tr_tr],
        eval_set=[(X[tr][tr_va], y[tr][tr_va])],
        eval_names=["val"],
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    holdout_proba = gbdt.predict_proba(X[te])[:, 1]
    holdout_auc = roc_auc_score(y[te], holdout_proba)
    print(f"[gbdt] LightGBM holdout AUC (group-aware) = {holdout_auc:.4f}  "
          f"(n_test={len(te)}, best_iter={gbdt.best_iteration_})")

    # 5-fold group-aware CV AUC for a stabler number.
    cv_aucs = []
    gss2 = GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
    for tr2, te2 in gss2.split(X, y, groups=groups):
        g_es = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
        a_tr, a_va = next(g_es.split(X[tr2], y[tr2], groups=groups[tr2]))
        m = LGBMClassifier(**best_params)
        m.fit(
            X[tr2][a_tr], y[tr2][a_tr],
            eval_set=[(X[tr2][a_va], y[tr2][a_va])],
            eval_names=["val"],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        cv_aucs.append(roc_auc_score(y[te2], m.predict_proba(X[te2])[:, 1]))
    cv_auc = float(np.mean(cv_aucs))
    print(f"[gbdt] LightGBM 5-fold group-aware AUC = "
          f"{cv_auc:.4f} +/- {np.std(cv_aucs):.4f}")

    # Feature importance (gain) for sanity.
    try:
        importances = gbdt.feature_importances_
        order = np.argsort(importances)[::-1]
        print("[gbdt] top-10 features (gain):")
        for k in order[:10]:
            name = RICH_FEATURE_NAMES[k] if k < len(RICH_FEATURE_NAMES) else f"f{k}"
            print(f"    {name:24s} {int(importances[k])}")
    except Exception as e:
        print(f"[gbdt] feature importance unavailable: {e}")

    # Refit final LightGBM on ALL data (no early stopping; use best_iter from
    # the holdout fit capped to a sane ceiling).
    n_best = int(gbdt.best_iteration_ or 500)
    final_params = dict(best_params)
    final_params["n_estimators"] = n_best
    final = LGBMClassifier(**final_params)
    final.fit(X, y)
    os.makedirs(os.path.dirname(str(gbdt_path)), exist_ok=True)
    joblib.dump(final, str(gbdt_path))
    print(f"[gbdt] saved LightGBM -> {gbdt_path} (n_estimators={n_best})")

    return {
        "linear11_cv_auc": float(np.mean(lin_aucs)),
        "linear_rich_cv_auc": float(np.mean(lin_rich_aucs)),
        "gbdt_holdout_auc": float(holdout_auc),
        "gbdt_cv_auc": cv_auc,
        "gbdt_cv_std": float(np.std(cv_aucs)),
        "n_states": len(rich_feats),
        "n_features": int(X.shape[1]),
        "n_best_estimators": n_best,
        "pos_fraction": float(y.mean()),
        "artifact_path": str(gbdt_path),
    }


def main():
    res = train_and_save()
    print("\n=== LINEAR VALUE PROBE TRAINED ===")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()