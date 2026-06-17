#!/usr/bin/env python3
"""
train_v009_il.py — Train v009 IL candidates with CLEAN-MIRROR data selection.

Task:
  Candidate A = clean-mirror ONLY (both players Crustle, winner-side, MAIN only).
  Candidate B = v007's top_players UNION new clean-mirror episodes (dedup).

Both use Config C6: lr=0.005, num_leaves=15, min_child_samples=50, 300 trees, early-stop DISABLED.

Run:
  .venv/bin/python scripts/train_v009_il.py

Outputs:
  models/il_crustle_v009a.txt  (Candidate A)
  models/il_crustle_v009b.txt  (Candidate B)
"""
from __future__ import annotations

import json
import sys
import hashlib
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

from replays.parse import parse_replay, walk_replays
from il.features import build_feature_matrix, FEATURE_DIM, FEATURE_NAMES

# ─── Constants ────────────────────────────────────────────────────────────────

TOP_DIR = REPO / "data" / "episodes" / "top_players"
ALL_TEAMS_DIR = REPO / "data" / "episodes" / "all_teams"
ROOT_EPISODES_DIR = REPO / "data" / "episodes"

OUT_A = REPO / "models" / "il_crustle_v009a.txt"
OUT_B = REPO / "models" / "il_crustle_v009b.txt"

# Config C6 (from task spec)
LR = 0.005
NUM_LEAVES = 15
MIN_CHILD_SAMPLES = 50
NUM_BOOST_ROUND = 300  # fixed, early-stop DISABLED

VAL_FRAC = 0.15
WINNER_WEIGHT = 2.0
LOSER_WEIGHT = 1.0  # only used for Candidate B (which includes non-mirror Crustle games)
SKIP_CONTEXTS = {41, 42}  # IS_FIRST, MULLIGAN

CRUSTLE_IDS = {344, 345}


def episode_to_val_flag(episode_id) -> bool:
    """Stable train/val split by episode id hash."""
    h = int(hashlib.md5(str(episode_id).encode()).hexdigest(), 16)
    return (h % 100) < int(VAL_FRAC * 100)


def extract_decks(path: Path) -> dict[int, list[int]]:
    """Extract player decks from a replay file (step[1] action = 60-card deck)."""
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


def is_crustle_deck(deck: list[int]) -> bool:
    return bool(CRUSTLE_IDS.intersection(deck))


def get_all_unique_replays() -> list[Path]:
    """Scan all episode dirs, dedupe by filename, return unique paths."""
    all_dirs = [TOP_DIR, ALL_TEAMS_DIR, ROOT_EPISODES_DIR]
    raw = walk_replays(all_dirs)
    seen: set[str] = set()
    unique: list[Path] = []
    for p in raw:
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p)
    return unique


def identify_mirror_episodes(paths: list[Path]) -> tuple[list[Path], dict[Path, set[int]]]:
    """
    Returns:
      mirror_paths: replays where BOTH players use a Crustle deck
      crustle_pis_by_path: path -> set of player indices that are Crustle
    """
    mirror_paths = []
    crustle_pis_by_path: dict[Path, set[int]] = {}
    parse_errors = 0
    for p in paths:
        try:
            decks = extract_decks(p)
            if len(decks) < 2:
                continue
            pis_with_crustle = {pi for pi, deck in decks.items() if is_crustle_deck(deck)}
            if len(pis_with_crustle) == 2:
                # TRUE MIRROR: both players have Crustle
                mirror_paths.append(p)
                crustle_pis_by_path[p] = pis_with_crustle
        except Exception as e:
            parse_errors += 1
    if parse_errors:
        print(f"  WARN: {parse_errors} parse errors during deck extraction")
    return mirror_paths, crustle_pis_by_path


def build_dataset(
    paths: list[Path],
    crustle_pis_by_path: dict[Path, set[int]],
    winner_only: bool = False,
    main_only: bool = False,
):
    """
    Build training dataset from replay paths.

    Args:
        paths: replay file paths to process
        crustle_pis_by_path: path -> set of Crustle player indices (filter to these)
        winner_only: if True, skip loser decisions entirely (for Candidate A)
        main_only: if True, only include MAIN (select_context==0) decisions

    Returns:
        dicts with train_X, train_y, train_g, train_w, train_ctx,
                    val_X, val_y, val_g, val_ctx, stats
    """
    train_X, train_y, train_g, train_w, train_ctx = [], [], [], [], []
    val_X, val_y, val_g, val_w, val_ctx = [], [], [], [], []

    total_pairs = 0
    main_train = 0
    main_val = 0
    winner_pairs = 0
    loser_pairs = 0
    skipped_ctx = 0
    skipped_empty = 0
    parse_errs = 0

    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
            ep_id = data.get("id") or data.get("info", {}).get("EpisodeId", 0)
            pairs = parse_replay(path)
        except Exception as e:
            parse_errs += 1
            continue

        # Which player indices to keep from this replay
        allowed_pis = crustle_pis_by_path.get(path)

        for pair in pairs:
            pi = pair["player_index"]
            if allowed_pis is not None and pi not in allowed_pis:
                continue

            ctx = pair["select_context"]
            if ctx in SKIP_CONTEXTS:
                skipped_ctx += 1
                continue

            if main_only and ctx != 0:
                continue

            n_opts = pair["n_options"]
            if n_opts == 0:
                skipped_empty += 1
                continue

            is_winner = pair["won"]

            if winner_only and not is_winner:
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

            w = WINNER_WEIGHT if is_winner else LOSER_WEIGHT

            total_pairs += 1
            if is_winner:
                winner_pairs += 1
            else:
                loser_pairs += 1

            is_val = episode_to_val_flag(ep_id)
            if is_val:
                val_X.append(X_group)
                val_y.append(y_group)
                val_g.append(n_opts)
                val_w.append(np.full(n_opts, w, dtype=np.float32))
                val_ctx.append(ctx)
                if ctx == 0:
                    main_val += 1
            else:
                train_X.append(X_group)
                train_y.append(y_group)
                train_g.append(n_opts)
                train_w.append(np.full(n_opts, w, dtype=np.float32))
                train_ctx.append(ctx)
                if ctx == 0:
                    main_train += 1

    return {
        "train_X": train_X, "train_y": train_y, "train_g": train_g,
        "train_w": train_w, "train_ctx": train_ctx,
        "val_X": val_X, "val_y": val_y, "val_g": val_g,
        "val_w": val_w, "val_ctx": val_ctx,
        "stats": {
            "total_pairs": total_pairs,
            "winner_pairs": winner_pairs,
            "loser_pairs": loser_pairs,
            "main_train": main_train,
            "main_val": main_val,
            "skipped_ctx": skipped_ctx,
            "skipped_empty": skipped_empty,
            "parse_errs": parse_errs,
        }
    }


def train_model(ds: dict, out_path: Path, name: str) -> float:
    """
    Train a LightGBM lambdarank model on the dataset dict.
    Returns offline MAIN top-1 accuracy on val set.
    """
    import lightgbm as lgb

    train_X = ds["train_X"]
    train_y = ds["train_y"]
    train_g = ds["train_g"]
    train_w = ds["train_w"]
    train_ctx = ds["train_ctx"]
    val_X = ds["val_X"]
    val_y = ds["val_y"]
    val_g = ds["val_g"]
    val_ctx = ds["val_ctx"]

    X_tr = np.vstack(train_X) if train_X else np.zeros((0, FEATURE_DIM), dtype=np.float32)
    y_tr = np.concatenate(train_y) if train_y else np.zeros(0, dtype=np.int8)
    g_tr = np.array(train_g, dtype=np.int32)
    w_tr = np.concatenate(train_w) if train_w else np.zeros(0, dtype=np.float32)

    X_va = np.vstack(val_X) if val_X else np.zeros((0, FEATURE_DIM), dtype=np.float32)
    y_va = np.concatenate(val_y) if val_y else np.zeros(0, dtype=np.int8)
    g_va = np.array(val_g, dtype=np.int32)
    c_va = np.array(val_ctx, dtype=np.int32)

    print(f"\n  [{name}] Dataset: train={len(g_tr)} groups / {len(X_tr)} rows  val={len(g_va)} groups / {len(X_va)} rows")

    dtrain = lgb.Dataset(
        X_tr.astype(np.float32), label=y_tr.astype(np.int32),
        group=g_tr.astype(np.int32), weight=w_tr.astype(np.float32),
        feature_name=FEATURE_NAMES, free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_va.astype(np.float32), label=y_va.astype(np.int32),
        group=g_va.astype(np.int32),
        feature_name=FEATURE_NAMES, reference=dtrain, free_raw_data=False,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3],
        "num_leaves": NUM_LEAVES,          # C6: 15 (low capacity)
        "learning_rate": LR,               # C6: 0.005
        "min_child_samples": MIN_CHILD_SAMPLES,  # C6: 50
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }

    # Early stopping DISABLED (C6)
    callbacks = [lgb.log_evaluation(period=50)]

    print(f"  [{name}] Training C6: lr={LR}, num_leaves={NUM_LEAVES}, min_child_samples={MIN_CHILD_SAMPLES}, {NUM_BOOST_ROUND} trees, early-stop=OFF ...")
    booster = lgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval], valid_names=["val"],
        callbacks=callbacks,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_path))
    print(f"  [{name}] Model saved → {out_path}  (trees: {booster.num_trees()})")

    # Offline evaluation
    if len(X_va) == 0:
        print(f"  [{name}] WARNING: empty val set — cannot evaluate offline accuracy")
        return 0.0

    scores_va = booster.predict(X_va.astype(np.float32))

    correct_main = n_main = 0
    pos = 0
    for gsz, gctx in zip(g_va, c_va):
        sg = scores_va[pos:pos + gsz]
        yg = y_va[pos:pos + gsz]
        pos += gsz
        if gctx != 0 or len(sg) == 0:
            continue
        top_idx = int(np.argmax(sg))
        n_main += 1
        if yg[top_idx] == 1:
            correct_main += 1

    # option-0 baseline
    correct_opt0 = n_opt0 = 0
    pos = 0
    for gsz, gctx in zip(g_va, c_va):
        yg = y_va[pos:pos + gsz]
        pos += gsz
        if gctx != 0 or len(yg) == 0:
            continue
        n_opt0 += 1
        if yg[0] == 1:
            correct_opt0 += 1

    main_top1 = correct_main / n_main if n_main else 0.0
    opt0_top1 = correct_opt0 / n_opt0 if n_opt0 else 0.0

    print(f"  [{name}] Offline val MAIN top-1: model={main_top1:.4f}  option-0={opt0_top1:.4f}  (n_main={n_main})")
    return main_top1


def main():
    print("=" * 70)
    print("  v009 IL Training — Clean-Mirror Data Selection")
    print(f"  Config C6: lr={LR}, num_leaves={NUM_LEAVES}, min_child_samples={MIN_CHILD_SAMPLES}, {NUM_BOOST_ROUND} trees, early-stop=OFF")
    print("=" * 70)

    # Step 1: find all unique replays
    print("\n[1/5] Scanning replay files...")
    all_paths = get_all_unique_replays()
    print(f"  Total unique replays: {len(all_paths)}")

    # Step 2: identify TRUE MIRROR episodes (both players Crustle)
    print("\n[2/5] Identifying clean-mirror episodes...")
    mirror_paths, mirror_crustle_pis = identify_mirror_episodes(all_paths)
    print(f"  Mirror episodes (BOTH Crustle): {len(mirror_paths)}")
    print(f"  Non-mirror episodes:            {len(all_paths) - len(mirror_paths)}")

    # Step 3: Candidate A — mirror only, winner-side only, MAIN only
    print("\n[3/5] Building Candidate A dataset (mirror + winner + MAIN)...")
    ds_a = build_dataset(
        paths=mirror_paths,
        crustle_pis_by_path=mirror_crustle_pis,
        winner_only=True,
        main_only=True,
    )
    s = ds_a["stats"]
    main_a = s["main_train"] + s["main_val"]
    print(f"  Total decisions kept:   {s['total_pairs']}")
    print(f"  Winner decisions:       {s['winner_pairs']}")
    print(f"  MAIN train:             {s['main_train']}")
    print(f"  MAIN val:               {s['main_val']}")
    print(f"  TOTAL MAIN (A):         {main_a}")
    print(f"  Skipped (ctx):          {s['skipped_ctx']}")
    print(f"  Skipped (empty/err):    {s['skipped_empty']}")
    print(f"  Parse errors:           {s['parse_errs']}")

    # Step 4: Candidate B — top_players UNION new mirror episodes (dedup)
    print("\n[4/5] Building Candidate B dataset (top_players UNION mirror)...")
    top_paths = walk_replays([TOP_DIR])
    top_names = {p.name for p in top_paths}
    extra_mirror = [p for p in mirror_paths if p.name not in top_names]
    union_paths = list(top_paths) + extra_mirror
    print(f"  top_players: {len(top_paths)}, extra_mirror: {len(extra_mirror)}, union: {len(union_paths)}")

    # For B, need Crustle player indices for all paths
    # top_players: use extract_decks to find Crustle pis
    print("  Identifying Crustle slots in top_players...")
    top_crustle_pis: dict[Path, set[int]] = {}
    for p in top_paths:
        try:
            decks = extract_decks(p)
            pis = {pi for pi, deck in decks.items() if is_crustle_deck(deck)}
            if pis:
                top_crustle_pis[p] = pis
        except:
            pass
    print(f"  top_players with Crustle: {len(top_crustle_pis)}/{len(top_paths)}")

    # Merge
    b_crustle_pis: dict[Path, set[int]] = {}
    for p, pis in top_crustle_pis.items():
        b_crustle_pis[p] = pis
    for p, pis in mirror_crustle_pis.items():
        if p not in b_crustle_pis:
            b_crustle_pis[p] = pis

    # B: winner-side only + MAIN only (same filter as A, but larger data)
    ds_b = build_dataset(
        paths=union_paths,
        crustle_pis_by_path=b_crustle_pis,
        winner_only=True,
        main_only=True,
    )
    s = ds_b["stats"]
    main_b = s["main_train"] + s["main_val"]
    print(f"  Total decisions kept:   {s['total_pairs']}")
    print(f"  Winner decisions:       {s['winner_pairs']}")
    print(f"  MAIN train:             {s['main_train']}")
    print(f"  MAIN val:               {s['main_val']}")
    print(f"  TOTAL MAIN (B):         {main_b}")

    # Step 5: train both
    print("\n[5/5] Training...")
    print("\n--- Candidate A (mirror-only) ---")
    top1_a = train_model(ds_a, OUT_A, "A")

    print("\n--- Candidate B (top_players + mirror union) ---")
    top1_b = train_model(ds_b, OUT_B, "B")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print(f"  Mirror episodes (BOTH Crustle):  {len(mirror_paths)}")
    print(f"  Mirror MAIN winner decisions (A): {main_a}")
    print(f"  v007 reference:                  ~4500 MAIN decisions (top_players, all-Crustle-player)")
    print(f"  Union MAIN winner decisions (B): {main_b}")
    print(f"")
    print(f"  Candidate A offline MAIN top-1:  {top1_a:.4f}  (v007 reference: 0.5388)")
    print(f"  Candidate B offline MAIN top-1:  {top1_b:.4f}  (v007 reference: 0.5388)")
    print(f"")
    print(f"  Model A: {OUT_A}")
    print(f"  Model B: {OUT_B}")
    print("=" * 70)


if __name__ == "__main__":
    main()
