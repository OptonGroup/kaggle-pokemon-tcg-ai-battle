#!/usr/bin/env python3
"""
train_grimmsnarl_il.py — Train v018 IL model for Grimmsnarl ex toolbox deck.

Single-expert BC: filter to KAZUKI0123 ONLY (player_name == "kazuki0123") using
the Grimmsnarl deck signature (contains 648), winner-side, MAIN decisions.

Model: Config C6 (same as v009a/v011/v016):
  lr=0.005, num_leaves=15, min_child_samples=50, 300 trees, early-stop=OFF,
  lambdarank, one query-group per decision, y=1 chosen/0 others, winner weight 2.0

Output: models/il_grimmsnarl.txt
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

from replays.parse import parse_replay, walk_replays
from il.features_grimmsnarl import build_feature_matrix, FEATURE_DIM, FEATURE_NAMES

# ─── Constants ────────────────────────────────────────────────────────────────

ALL_TEAMS_DIR = REPO / "data" / "episodes" / "all_teams"
TOP_DIR = REPO / "data" / "episodes" / "top_players"
ROOT_EPISODES_DIR = REPO / "data" / "episodes"
OUT_MODEL = REPO / "models" / "il_grimmsnarl.txt"

EXPERT_NAME = "kazuki0123"
GRIMMSNARL_SIG = {648}

LR = 0.005
NUM_LEAVES = 15
MIN_CHILD_SAMPLES = 50
NUM_BOOST_ROUND = 300
VAL_FRAC = 0.15
WINNER_WEIGHT = 2.0
SKIP_CONTEXTS = {41, 42}


def episode_to_val_flag(episode_id) -> bool:
    h = int(hashlib.md5(str(episode_id).encode()).hexdigest(), 16)
    return (h % 100) < int(VAL_FRAC * 100)


def get_all_unique_replays() -> list[Path]:
    all_dirs = [ALL_TEAMS_DIR, TOP_DIR, ROOT_EPISODES_DIR]
    raw = walk_replays(all_dirs)
    seen: set[str] = set()
    unique: list[Path] = []
    for p in raw:
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p)
    return unique


def extract_decks(path: Path) -> dict[int, list[int]]:
    with open(path) as f:
        data = json.load(f)
    steps = data.get("steps", [])
    if len(steps) < 2:
        return {}
    decks = {}
    for pi in range(min(2, len(steps[1]))):
        action = steps[1][pi].get("action", [])
        if len(action) == 60 and all(isinstance(x, int) for x in action):
            decks[pi] = action
    return decks


def identify_kazuki_episodes(paths: list[Path]) -> dict[Path, set[int]]:
    """Return path -> set of player indices where kazuki0123 plays the Grimmsnarl deck."""
    out: dict[Path, set[int]] = {}
    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
            agents = data.get("info", {}).get("Agents", [])
            names = [a.get("Name") for a in agents]
            decks = extract_decks(p)
            for pi, deck in decks.items():
                if GRIMMSNARL_SIG.issubset(set(deck)) and pi < len(names) and names[pi] == EXPERT_NAME:
                    out.setdefault(p, set()).add(pi)
        except Exception:
            pass
    return out


def build_dataset(paths_pis: dict[Path, set[int]]) -> dict:
    train_X, train_y, train_g, train_w, train_ctx = [], [], [], [], []
    val_X, val_y, val_g, val_w, val_ctx = [], [], [], [], []

    total_pairs = main_train = main_val = 0
    skipped_empty = parse_errs = 0

    for path, pis in paths_pis.items():
        try:
            with open(path) as f:
                data = json.load(f)
            ep_id = data.get("id") or data.get("info", {}).get("EpisodeId", 0)
            pairs = parse_replay(path)
        except Exception:
            parse_errs += 1
            continue

        for pair in pairs:
            pi = pair["player_index"]
            if pi not in pis:
                continue
            ctx = pair["select_context"]
            if ctx in SKIP_CONTEXTS or ctx != 0:
                continue
            if not pair["won"]:
                continue
            n_opts = pair["n_options"]
            if n_opts == 0:
                skipped_empty += 1
                continue
            obs = pair["obs_dict"]
            chosen = set(pair["chosen_option_indices"])
            try:
                X_group = build_feature_matrix(obs)
            except Exception:
                skipped_empty += 1
                continue
            if X_group.shape[0] == 0 or X_group.shape[0] != n_opts:
                skipped_empty += 1
                continue
            y_group = np.zeros(n_opts, dtype=np.int8)
            for ci in chosen:
                if ci < n_opts:
                    y_group[ci] = 1
            w = WINNER_WEIGHT
            total_pairs += 1
            is_val = episode_to_val_flag(ep_id)
            if is_val:
                val_X.append(X_group); val_y.append(y_group)
                val_g.append(n_opts); val_w.append(np.full(n_opts, w, dtype=np.float32))
                val_ctx.append(ctx); main_val += 1
            else:
                train_X.append(X_group); train_y.append(y_group)
                train_g.append(n_opts); train_w.append(np.full(n_opts, w, dtype=np.float32))
                train_ctx.append(ctx); main_train += 1

    return {
        "train_X": train_X, "train_y": train_y, "train_g": train_g,
        "train_w": train_w, "train_ctx": train_ctx,
        "val_X": val_X, "val_y": val_y, "val_g": val_g,
        "val_w": val_w, "val_ctx": val_ctx,
        "stats": {"total_pairs": total_pairs, "main_train": main_train,
                  "main_val": main_val, "skipped_empty": skipped_empty,
                  "parse_errs": parse_errs},
    }


def train_model(ds: dict, out_path: Path) -> float:
    import lightgbm as lgb

    X_tr = np.vstack(ds["train_X"]) if ds["train_X"] else np.zeros((0, FEATURE_DIM), dtype=np.float32)
    y_tr = np.concatenate(ds["train_y"]) if ds["train_y"] else np.zeros(0, dtype=np.int8)
    g_tr = np.array(ds["train_g"], dtype=np.int32)
    w_tr = np.concatenate(ds["train_w"]) if ds["train_w"] else np.zeros(0, dtype=np.float32)
    X_va = np.vstack(ds["val_X"]) if ds["val_X"] else np.zeros((0, FEATURE_DIM), dtype=np.float32)
    y_va = np.concatenate(ds["val_y"]) if ds["val_y"] else np.zeros(0, dtype=np.int8)
    g_va = np.array(ds["val_g"], dtype=np.int32)
    c_va = np.array(ds["val_ctx"], dtype=np.int32)

    print(f"\n  Dataset: train={len(g_tr)} groups / {len(X_tr)} rows  val={len(g_va)} groups / {len(X_va)} rows")

    dtrain = lgb.Dataset(X_tr.astype(np.float32), label=y_tr.astype(np.int32),
                         group=g_tr, weight=w_tr.astype(np.float32),
                         feature_name=FEATURE_NAMES, free_raw_data=False)
    dval = lgb.Dataset(X_va.astype(np.float32), label=y_va.astype(np.int32),
                       group=g_va, feature_name=FEATURE_NAMES,
                       reference=dtrain, free_raw_data=False)

    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 3],
        "num_leaves": NUM_LEAVES, "learning_rate": LR,
        "min_child_samples": MIN_CHILD_SAMPLES, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1,
        "seed": 42, "n_jobs": -1,
    }
    booster = lgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND,
                        valid_sets=[dval], valid_names=["val"],
                        callbacks=[lgb.log_evaluation(period=50)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_path))
    print(f"  Model saved -> {out_path}  (trees: {booster.num_trees()})")

    importances = booster.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    print(f"\n  Top-15 feature importances (gain):")
    for fname, imp in feat_imp[:15]:
        print(f"    {fname:42s}: {imp:.1f}")

    if len(X_va) == 0:
        print("  WARNING: empty val set")
        return 0.0, 0.0

    scores_va = booster.predict(X_va.astype(np.float32))
    correct_main = n_main = 0
    correct_overall = n_overall = 0
    pos = 0
    for gsz, gctx in zip(g_va, c_va):
        sg = scores_va[pos:pos + gsz]; yg = y_va[pos:pos + gsz]; pos += gsz
        if len(sg) == 0:
            continue
        top_idx = int(np.argmax(sg))
        n_overall += 1
        if yg[top_idx] == 1:
            correct_overall += 1
        if gctx == 0:
            n_main += 1
            if yg[top_idx] == 1:
                correct_main += 1
    correct_opt0 = n_opt0 = 0
    pos = 0
    for gsz, gctx in zip(g_va, c_va):
        yg = y_va[pos:pos + gsz]; pos += gsz
        if gctx != 0 or len(yg) == 0:
            continue
        n_opt0 += 1
        if yg[0] == 1:
            correct_opt0 += 1
    main_top1 = correct_main / n_main if n_main else 0.0
    overall_top1 = correct_overall / n_overall if n_overall else 0.0
    opt0_top1 = correct_opt0 / n_opt0 if n_opt0 else 0.0
    print(f"\n  Offline val MAIN top-1:    model={main_top1:.4f}  option-0={opt0_top1:.4f}  (n_main={n_main})")
    print(f"  Offline val OVERALL top-1: model={overall_top1:.4f}  (n_overall={n_overall})")
    return main_top1, overall_top1


def main():
    print("=" * 70)
    print("  v018 IL Training — Grimmsnarl ex toolbox (single-expert: kazuki0123)")
    print(f"  Config C6: lr={LR}, num_leaves={NUM_LEAVES}, min_child={MIN_CHILD_SAMPLES}, {NUM_BOOST_ROUND} trees")
    print("=" * 70)

    print("\n[1/4] Scanning replay files...")
    all_paths = get_all_unique_replays()
    print(f"  Total unique replays: {len(all_paths)}")

    print("\n[2/4] Identifying kazuki0123 (Grimmsnarl-sig) episodes...")
    kazuki_pis = identify_kazuki_episodes(all_paths)
    n_games = sum(len(pis) for pis in kazuki_pis.values())
    print(f"  kazuki0123 Grimmsnarl episodes: {len(kazuki_pis)}  (player-game slots: {n_games})")

    print("\n[3/4] Building dataset (winner-MAIN only)...")
    ds = build_dataset(kazuki_pis)
    s = ds["stats"]
    total_main = s["main_train"] + s["main_val"]
    print(f"  Winner-MAIN train: {s['main_train']}")
    print(f"  Winner-MAIN val:   {s['main_val']}")
    print(f"  TOTAL MAIN:        {total_main}")
    print(f"  Skipped (empty):   {s['skipped_empty']}")
    print(f"  Parse errors:      {s['parse_errs']}")
    if total_main < 100:
        print("ERROR: Too few training decisions — aborting.")
        sys.exit(1)

    print("\n[4/4] Training model...")
    main_top1, overall_top1 = train_model(ds, OUT_MODEL)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print(f"  kazuki0123 Grimmsnarl episodes: {len(kazuki_pis)}")
    print(f"  Winner-MAIN decisions total:    {total_main}")
    print(f"  Offline val MAIN top-1:         {main_top1:.4f}")
    print(f"  Offline val OVERALL top-1:      {overall_top1:.4f}")
    print(f"  Model: {OUT_MODEL}")
    print("=" * 70)


if __name__ == "__main__":
    main()
