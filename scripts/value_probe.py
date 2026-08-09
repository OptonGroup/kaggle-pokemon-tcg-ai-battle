#!/usr/bin/env python3
"""
value_probe.py — Linear-probe vs v024 value-head AUC sanity check.

Diagnoses the Kaggle forum 713608 claim for OUR data:
  - Extract decision states from a sample of replays in data/episodes/all_teams.
  - Label = did the player-to-move eventually win (from replay rewards)?
  - Linear probe: ~10 cheap board features -> LogisticRegression -> ROC-AUC.
  - v024 value head: load submissions/v024_cardemb_rl/model.pth, run the
    MyModel encoder value output on the same states, compute ROC-AUC of
    value vs won.

No game engine (libcg) required — pure JSON replay parsing + torch CPU.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _stable_group_id(episode_id, n_folds=10**9):
    """Deterministic integer group id for an episode_id, independent of
    PYTHONHASHSEED. Replaces the old `hash(str(episode_id))` which was
    randomized per process and made linear_probe_auc vary run-to-run."""
    h = hashlib.md5(str(episode_id).encode("utf-8")).hexdigest()
    return int(h, 16) % n_folds

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

# ---------- replay parsing ----------

def extract_features_and_label(obs: dict, won: bool):
    """Return (feature_vector, won) or None if state is pre-game (no active)."""
    cur = obs.get("current")
    if not cur:
        return None
    yi = cur.get("yourIndex", 0)
    turn = cur.get("turn", 0)
    first_player = cur.get("firstPlayer", -1)
    ps = cur.get("players", [])
    if len(ps) < 2:
        return None
    me = ps[yi]
    opp = ps[1 - yi]

    def active_hp(pl):
        act = pl.get("active") or []
        if act and isinstance(act[0], dict):
            return act[0].get("hp") or 0
        return 0

    def active_maxhp(pl):
        act = pl.get("active") or []
        if act and isinstance(act[0], dict):
            return act[0].get("maxHp") or 1
        return 1

    def total_energy(pl):
        tot = 0
        act = pl.get("active") or []
        if act and isinstance(act[0], dict):
            tot += len(act[0].get("energies") or [])
        for b in (pl.get("bench") or []):
            if isinstance(b, dict):
                tot += len(b.get("energies") or [])
        return tot

    def n_evolved(pl):
        """Count pokemon that have a preEvolution (i.e. evolved)."""
        n = 0
        act = pl.get("active") or []
        if act and isinstance(act[0], dict):
            n += len(act[0].get("preEvolution") or [])
        for b in (pl.get("bench") or []):
            if isinstance(b, dict):
                n += len(b.get("preEvolution") or [])
        return n

    def can_attack(pl):
        """Active pokemon has at least one attack option? Approx via energy presence
        on active. We don't have attack costs here without the card DB; use a
        cheap proxy: active has >0 energy attached. (Boolean feature.)"""
        act = pl.get("active") or []
        if act and isinstance(act[0], dict):
            return 1.0 if len(act[0].get("energies") or []) > 0 else 0.0
        return 0.0

    my_hp = active_hp(me)
    opp_hp = active_hp(opp)
    my_maxhp = active_maxhp(me)
    opp_maxhp = active_maxhp(opp)

    # ~10 cheap board features (all normalized modestly)
    feats = [
        (my_hp - opp_hp) / 200.0,                              # active hp diff
        (my_hp / max(my_maxhp, 1)) - (opp_hp / max(opp_maxhp, 1)),  # active hp frac diff
        len(me.get("prize") or []) - len(opp.get("prize") or []),  # prize remaining diff (more prizes left = behind)
        len(me.get("bench") or []) - len(opp.get("bench") or []),  # bench count diff
        me.get("handCount", 0) - opp.get("handCount", 0),     # hand count diff
        turn / 20.0,                                           # turn number
        total_energy(me) - total_energy(opp),                 # total energy attached diff
        n_evolved(me) - n_evolved(opp),                       # evolved pokemon count diff
        len(me.get("discard") or []) - len(opp.get("discard") or []),  # discard count diff
        1.0 if (first_player == yi) else 0.0,                 # is-first flag
        can_attack(me) - can_attack(opp),                      # active-can-attack proxy diff
    ]
    return feats, won


def collect_states(replay_paths, max_games=600, seed=0):
    rng = random.Random(seed)
    paths = list(replay_paths)
    rng.shuffle(paths)
    paths = paths[:max_games]
    rows = []  # (feats, won, episode_id)
    vhead_inputs = []  # (obs_dict, won, episode_id, player_index) for value-head eval
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
            continue  # no winner recorded (draw/incomplete)
        ep_id = data.get("id") or data.get("info", {}).get("EpisodeId", p)
        for i in range(len(steps) - 1):
            for pi in range(min(n_players, 2)):
                obs = steps[i][pi].get("observation", {})
                sel = obs.get("select")
                cur = obs.get("current")
                if sel is None or cur is None:
                    continue
                ctx = sel.get("context", -1)
                if ctx in (41, 42):  # skip IS_FIRST / MULLIGAN
                    continue
                if cur.get("turn", 0) < 1:
                    continue
                # action lives in next step
                nxt = steps[i + 1]
                if pi >= len(nxt):
                    continue
                action = nxt[pi].get("action", [])
                if not action:
                    continue
                if len(action) == 60:
                    continue
                won = won_by_player.get(pi, False)
                r = extract_features_and_label(obs, won)
                if r is None:
                    continue
                feats, won = r
                rows.append((feats, int(won), ep_id))
                vhead_inputs.append((obs, int(won), ep_id, pi))
    return rows, vhead_inputs


# ---------- v024 value head eval ----------

def load_v024_model():
    """Load the v024 card-embedding MyModel + its Crustle deck (baseline)."""
    sys.path.insert(0, str(REPO / "submissions" / "v024_cardemb_rl"))
    import torch
    import cardemb_model as cm
    model = cm.MyModel(128, 2, 256, 1, 1, dropout=0.0, emb_dropout=0.0)
    state = torch.load(
        str(REPO / "submissions" / "v024_cardemb_rl" / "model.pth"),
        map_location="cpu", weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    deck_csv = (REPO / "submissions" / "v024_cardemb_rl" / "deck.csv").read_text().splitlines()
    deck = [int(x) for x in deck_csv if x.strip()][:60]
    return model, deck


def _resolve_enc_dec_fns(model):
    """Return (get_encoder_input, get_decoder_input) from the model's own
    module if available (works for both cardemb_common.MyModel and
    cardemb_model.MyModel — they define identical kiyotah helpers)."""
    mod = sys.modules.get(type(model).__module__)
    ge = getattr(mod, "get_encoder_input", None)
    gd = getattr(mod, "get_decoder_input", None)
    if ge is None or gd is None:
        # Fall back to cardemb_common (scripts/ dir).
        sys.path.insert(0, str(REPO / "scripts"))
        import cardemb_common as cc  # noqa: E402
        ge, gd = cc.get_encoder_input, cc.get_decoder_input
    return ge, gd


def eval_vhead(vhead_inputs, model=None, deck=None, max_n=4000, seed=0):
    """Run a card-embedding MyModel value head on `vhead_inputs` states.
    Returns (values, labels) numpy arrays. If model/deck are None, loads the
    v024 baseline model + Crustle deck."""
    import torch
    if model is None or deck is None:
        model, deck = load_v024_model()
    model.eval()
    get_encoder_input, get_decoder_input = _resolve_enc_dec_fns(model)

    rng = random.Random(seed)
    idxs = list(range(len(vhead_inputs)))
    rng.shuffle(idxs)
    idxs = idxs[:max_n]

    values = []
    labels = []
    n_ok = 0
    n_skip = 0
    for k in idxs:
        obs_dict, won, ep_id, pi = vhead_inputs[k]
        try:
            # Build a minimal observation object with the attributes the encoder
            # expects (obs.current.yourIndex, .turn, .firstPlayer, .players[],
            # obs.current.stadium, obs.select.option). Dict-namespace shim.
            obs = _DictObs(obs_dict)
            sv_enc = get_encoder_input(obs, deck)
            # decoder: build a dummy single-option decoder so the forward pass
            # has valid shapes; we only need the value output.
            sel = obs_dict.get("select") or {}
            opts = sel.get("option") or []
            n_opts = max(1, min(len(opts), 48))
            actions = [[k] for k in range(n_opts)]
            sv_dec = get_decoder_input(obs, actions)
            with torch.inference_mode():
                v, _ = model(
                    torch.tensor(sv_enc.index, dtype=torch.int32),
                    torch.tensor(sv_enc.value, dtype=torch.float32),
                    torch.tensor(sv_enc.offset, dtype=torch.int32),
                    torch.tensor(sv_dec.index, dtype=torch.int32),
                    torch.tensor(sv_dec.value, dtype=torch.float32),
                    torch.tensor(sv_dec.offset, dtype=torch.int32),
                )
            values.append(float(v.tolist()[0][0]))
            labels.append(won)
            n_ok += 1
        except Exception:
            n_skip += 1
    print(f"[vhead] evaluated {n_ok} states, skipped {n_skip}")
    return np.array(values), np.array(labels)


def compute_aucs(model=None, deck=None, replay_paths=None,
                 max_games=600, seed=0, verbose=True):
    """Run the full value-head AUC probe and return a dict with:
      available (bool), linear_probe_auc (float|None), value_head_auc (float|None),
      n_decision_states, n_value_states, pos_fraction.
    If model/deck are None, loads the v024 baseline. If replay_paths is None,
    globs data/episodes/all_teams/*-replay.json. If no replays are found,
    returns available=False (does NOT raise) so callers can skip the gate."""
    if replay_paths is None:
        replay_paths = sorted(glob.glob(str(REPO / "data" / "episodes" / "all_teams" / "*-replay.json")))
    if not replay_paths:
        if verbose:
            print("[auc-gate] no replay files found under data/episodes/all_teams/; skipping AUC gate.")
        return {"available": False, "linear_probe_auc": None, "value_head_auc": None,
                "n_decision_states": 0, "n_value_states": 0, "pos_fraction": None}
    if verbose:
        print(f"[auc-gate] found {len(replay_paths)} replay files")
    rows, vhead_inputs = collect_states(replay_paths, max_games=max_games, seed=seed)
    if not rows:
        if verbose:
            print("[auc-gate] no decision states collected; skipping AUC gate.")
        return {"available": False, "linear_probe_auc": None, "value_head_auc": None,
                "n_decision_states": 0, "n_value_states": 0, "pos_fraction": None}

    X = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=int)
    groups = np.array([_stable_group_id(r[2]) for r in rows], dtype=int)
    pos_frac = float(y.mean())
    if verbose:
        print(f"[auc-gate] collected {len(rows)} decision states; pos fraction={pos_frac:.3f}")

    # Linear probe: group-aware 5-fold CV (by episode, no leakage).
    aucs = []
    gss2 = GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
    for tr2, te2 in gss2.split(X, y, groups=groups):
        c2 = LogisticRegression(max_iter=2000, C=1.0)
        c2.fit(X[tr2], y[tr2])
        aucs.append(roc_auc_score(y[te2], c2.predict_proba(X[te2])[:, 1]))
    linear_probe_auc = float(np.mean(aucs))

    # Value head AUC.
    vals, vlabels = eval_vhead(vhead_inputs, model=model, deck=deck, max_n=4000, seed=seed)
    value_head_auc = None
    if len(vals) > 10:
        value_head_auc = float(roc_auc_score(vlabels, vals))

    if verbose:
        print(f"[auc-gate] linear_probe_auc={linear_probe_auc:.4f}  value_head_auc={value_head_auc}")
    return {
        "available": True,
        "linear_probe_auc": linear_probe_auc,
        "value_head_auc": value_head_auc,
        "n_decision_states": len(rows),
        "n_value_states": int(len(vals)),
        "pos_fraction": pos_frac,
    }


class _DictObs:
    """Shim that exposes attribute access over a dict, for cm.get_encoder_input /
    get_decoder_input which use obs.current.x and obs.select.option[i].type etc.
    Nested dicts/lists are wrapped on demand."""
    def __init__(self, d):
        self._d = d

    def __getattr__(self, name):
        if name == "_d":
            raise AttributeError()
        v = self._d.get(name)
        if isinstance(v, dict):
            return _DictObs(v)
        if isinstance(v, list):
            return [_DictObs(x) if isinstance(x, dict) else x for x in v]
        return v

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return iter(self._d)


def main():
    paths = sorted(glob.glob(str(REPO / "data" / "episodes" / "all_teams" / "*-replay.json")))
    print(f"found {len(paths)} replay files")
    rows, vhead_inputs = collect_states(paths, max_games=600, seed=0)
    X = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=int)
    groups = np.array([_stable_group_id(r[2]) for r in rows], dtype=int)
    print(f"collected {len(rows)} decision states; pos fraction={y.mean():.3f}")
    feat_names = [
        "active_hp_diff", "active_hp_frac_diff", "prize_rem_diff", "bench_diff",
        "hand_diff", "turn", "energy_diff", "evolved_diff", "discard_diff",
        "is_first", "can_attack_diff",
    ]

    # Linear probe: group-aware split (by episode) so no leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    tr, te = next(gss.split(X, y, groups=groups))
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])[:, 1]
    auc_probe = roc_auc_score(y[te], proba)
    # also cross-val AUC over full set (group-aware) for a stabler number
    aucs = []
    gss2 = GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
    for tr2, te2 in gss2.split(X, y, groups=groups):
        c2 = LogisticRegression(max_iter=2000, C=1.0)
        c2.fit(X[tr2], y[tr2])
        aucs.append(roc_auc_score(y[te2], c2.predict_proba(X[te2])[:, 1]))
    print(f"\n=== LINEAR PROBE (11 board features, LogisticRegression) ===")
    print(f"  single holdout AUC = {auc_probe:.4f}  (n_test={len(te)})")
    print(f"  5-fold group-aware AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"  coefficients:")
    for name, coef in sorted(zip(feat_names, clf.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"    {name:22s} {coef:+.3f}")

    # v024 value head (baseline; pass model=None to load v024)
    print(f"\n=== v024 VALUE HEAD (model.pth = rl_full/iter2) ===")
    vals, vlabels = eval_vhead(vhead_inputs, model=None, deck=None, max_n=4000, seed=0)
    if len(vals) > 10:
        auc_vhead = roc_auc_score(vlabels, vals)
        print(f"  n = {len(vals)}  pos fraction = {vlabels.mean():.3f}")
        print(f"  value mean={vals.mean():.4f} std={vals.std():.4f} min={vals.min():.4f} max={vals.max():.4f}")
        print(f"  ROC-AUC(value vs won) = {auc_vhead:.4f}")
        print(f"  Pearson r(value, won) = {np.corrcoef(vals, vlabels)[0,1]:.4f}")
        print(f"  value | won: mean={vals[vlabels==1].mean():.4f} std={vals[vlabels==1].std():.4f}")
        print(f"  value | lost: mean={vals[vlabels==0].mean():.4f} std={vals[vlabels==0].std():.4f}")
    else:
        print("  (not enough evaluated states)")

    print("\n=== VERDICT ===")
    if len(vals) > 10:
        print(f"  linear probe AUC ~ {np.mean(aucs):.3f}; v024 value-head AUC = {auc_vhead:.3f}")
        if auc_vhead <= np.mean(aucs) + 0.02:
            print("  -> value head is DEAD or below trivial linear probe.")
        else:
            print("  -> value head beats linear probe (signal present).")


if __name__ == "__main__":
    main()