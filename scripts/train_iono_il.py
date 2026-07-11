#!/usr/bin/env python3
"""
train_iono_il.py — Filter Iono Electric Swarm (episode,player) slots from top-player replays,
build an Iono-only IL dataset, train LightGBM lambdarank, save models/il_iono_a.txt or il_iono_b.txt.

Run:
  .venv/bin/python scripts/train_iono_il.py --scope onechan1
  .venv/bin/python scripts/train_iono_il.py --scope top3

Steps:
  1. Extract decks from all top-player replays.
  2. Assign archetype signature (top-3 Pokémon card IDs).
  3. Identify the Iono archetype slot (Voltorb=265, Tadbulb=268, Bellibolt ex=269, Wattrel=270, Kilowattrel=271).
  4. Filter parse_replay() pairs to only those (episodeId, player_idx) slots AND matching player scope.
  5. Build X/y/groups/weights (same logic as src/il/dataset.py), split 85/15 by episode.
  6. Train LightGBM lambdarank (300 trees, no early stopping).
  7. Report: MAIN decision count, offline top-1 vs option-0 baseline, save model.
"""
from __future__ import annotations

import json
import sys
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

from replays.parse import parse_replay, walk_replays
from il.features_iono import build_feature_matrix, FEATURE_DIM, FEATURE_NAMES

# ─── Constants ────────────────────────────────────────────────────────────────

import argparse

TOP_DIR = REPO / "data" / "episodes" / "top_players"
ALL_TEAMS_DIR = REPO / "data" / "episodes" / "all_teams"
ROOT_EPISODES_DIR = REPO / "data" / "episodes"
EN_CARD_CSV = REPO / "data" / "EN_Card_Data.csv"
OUT_MODEL_A = REPO / "models" / "il_iono_a.txt"
OUT_MODEL_B = REPO / "models" / "il_iono_b.txt"
VAL_FRAC = 0.15
WINNER_WEIGHT = 2.0
LOSER_WEIGHT = 1.0
SKIP_CONTEXTS = {41, 42}  # IS_FIRST, MULLIGAN

IONO_POKEMON_IDS = {265, 268, 269, 270, 271}  # Voltorb, Tadbulb, Bellibolt ex, Wattrel, Kilowattrel

# Player scopes
SCOPE_ONECHAN1 = {"onechan1"}
SCOPE_TOP3 = {"onechan1", "Kyo_s_s", "YT"}

# ─── CLI argument parsing ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Iono Electric Swarm IL ranker")
    p.add_argument("--lr", type=float, default=0.005, help="learning rate")
    p.add_argument("--num_leaves", type=int, default=15)
    p.add_argument("--min_child_samples", type=int, default=50)
    p.add_argument("--num_boost_round", type=int, default=300)
    p.add_argument("--scope", choices=["onechan1", "top3"], default="top3",
                   help="Player scope: onechan1=rank1 only, top3=onechan1+Kyo_s_s+YT")
    p.add_argument("--out", type=str, default=None, help="output model path (overrides default)")
    return p.parse_args()

# ─── Card info loading ────────────────────────────────────────────────────────

def load_card_info() -> dict[int, dict]:
    info: dict[int, dict] = {}
    with open(EN_CARD_CSV) as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
        for line in f:
            parts = line.strip().split(",")
            if not parts or not parts[0].strip():
                continue
            try:
                cid = int(parts[0].strip())
            except ValueError:
                continue
            row = {h: (parts[i].strip() if i < len(parts) else "") for i, h in enumerate(header)}
            info[cid] = row
    return info


def deck_signature(deck: list[int], card_info: dict[int, dict]) -> str:
    """Top-3 Pokémon card IDs by frequency → stable archetype signature."""
    counts = Counter(deck)
    POKEMON_STAGES = {"Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"}
    pokemon_ids = [
        (cid, cnt) for cid, cnt in counts.most_common()
        if card_info.get(cid, {}).get("Stage (Pokémon)/Type (Energy and Trainer)", "") in POKEMON_STAGES
    ]
    if not pokemon_ids:
        non_energy = [
            (cid, cnt) for cid, cnt in counts.most_common()
            if card_info.get(cid, {}).get("Stage (Pokémon)/Type (Energy and Trainer)", "") not in
               {"Basic Energy", "Special Energy"}
        ]
        top3 = sorted(non_energy[:3], key=lambda x: (-x[1], x[0]))
        return str([cid for cid, _ in top3])
    top3 = sorted(pokemon_ids, key=lambda x: (-x[1], x[0]))[:3]
    return str([cid for cid, _ in top3])


def extract_deck(path: Path) -> dict[int, list[int]]:
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


def is_iono_deck(deck: list[int]) -> bool:
    return bool(IONO_POKEMON_IDS.intersection(set(deck)))


def _episode_to_val_flag(episode_id: int) -> bool:
    h = int(hashlib.md5(str(episode_id).encode()).hexdigest(), 16)
    return (h % 100) < int(VAL_FRAC * 100)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Determine scope filter and output model
    if args.scope == "onechan1":
        scope_players = SCOPE_ONECHAN1
        out_model = Path(args.out) if args.out else OUT_MODEL_A
    else:  # top3
        scope_players = SCOPE_TOP3
        out_model = Path(args.out) if args.out else OUT_MODEL_B

    print("=" * 70)
    print("  IONO ELECTRIC SWARM IL TRAINING")
    print(f"  scope={args.scope}  players={sorted(scope_players)}")
    print(f"  lr={args.lr}  num_leaves={args.num_leaves}  min_child_samples={args.min_child_samples}  "
          f"early_stopping=DISABLED  num_boost_round={args.num_boost_round}")
    print("=" * 70)

    card_info = load_card_info()
    print(f"\nLoaded {len(card_info)} card entries.")

    # Scan all episode directories; dedupe by filename
    all_dirs = [TOP_DIR, ALL_TEAMS_DIR, ROOT_EPISODES_DIR]
    replay_paths_raw = walk_replays(all_dirs)
    seen_names: set[str] = set()
    unique_paths: list[Path] = []
    for p in replay_paths_raw:
        if p.name not in seen_names:
            seen_names.add(p.name)
            unique_paths.append(p)
    replay_paths = unique_paths
    print(f"Found {len(replay_paths)} unique replay files across {[str(d) for d in all_dirs]}")

    # ── Step 1: identify Iono (episode, player_idx) slots ─────────────────────
    print("\n[1/5] Identifying Iono slots...")
    iono_slots: set[tuple] = set()   # (episode_id, player_idx)
    iono_paths: list[Path] = []
    parse_errors = 0
    total_slots = 0

    for path in replay_paths:
        try:
            with open(path) as f:
                data = json.load(f)
            ep_id = data.get("id") or data.get("info", {}).get("EpisodeId", 0)
            decks = extract_deck(path)
            has_iono = False
            for pi, deck in decks.items():
                total_slots += 1
                if is_iono_deck(deck):
                    iono_slots.add((ep_id, pi))
                    has_iono = True
            if has_iono:
                iono_paths.append(path)
        except Exception as e:
            parse_errors += 1
            print(f"  WARN {path.name}: {e}")

    print(f"  Total player-game slots scanned: {total_slots}")
    print(f"  Iono slots identified:           {len(iono_slots)}")
    print(f"  Replays with >= 1 Iono player:   {len(iono_paths)}")
    print(f"  Parse errors: {parse_errors}")

    if not iono_slots:
        print("ERROR: No Iono slots found!")
        sys.exit(1)

    # ── Step 2: parse decisions for Iono slots only ────────────────────────────
    print(f"\n[2/5] Parsing decisions for Iono-only slots (scope={args.scope}: {sorted(scope_players)})...")

    train_X, train_y, train_g, train_w, train_ctx = [], [], [], [], []
    val_X, val_y, val_g, val_w, val_ctx = [], [], [], [], []

    total_pairs = 0
    main_train = 0
    main_val = 0
    winner_pairs = 0
    loser_pairs = 0
    skipped_empty = 0
    skipped_ctx = 0
    skipped_scope = 0
    parse_errs2 = 0
    multi_select = 0

    for path in iono_paths:
        try:
            with open(path) as f:
                data = json.load(f)
            ep_id = data.get("id") or data.get("info", {}).get("EpisodeId", 0)
            pairs = parse_replay(path)
        except Exception as e:
            parse_errs2 += 1
            print(f"  WARN {path.name}: {e}")
            continue

        for pair in pairs:
            pi = pair["player_index"]
            if (ep_id, pi) not in iono_slots:
                continue  # other player, not Iono

            # Scope filter: only include decisions from players in scope
            player_name = pair.get("player_name", "")
            if player_name not in scope_players:
                skipped_scope += 1
                continue

            ctx = pair["select_context"]
            if ctx in SKIP_CONTEXTS:
                skipped_ctx += 1
                continue

            n_opts = pair["n_options"]
            if n_opts == 0:
                skipped_empty += 1
                continue

            # Winner-side filter: only include pairs where is_winner==True
            is_winner = pair["won"]
            if not is_winner:
                loser_pairs += 1
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

            w = WINNER_WEIGHT  # always winner (loser skipped above)
            if len(chosen) > 1:
                multi_select += 1

            total_pairs += 1
            winner_pairs += 1

            is_val = _episode_to_val_flag(ep_id)
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

    print(f"  Total Iono decisions parsed:     {total_pairs}")
    print(f"  Winner decisions included:       {winner_pairs}")
    print(f"  Loser decisions skipped:         {loser_pairs}")
    print(f"  Skipped (scope filter):          {skipped_scope}")
    print(f"  Skipped (ctx/empty/err):         {skipped_ctx + skipped_empty}")
    print(f"  Multi-select:                    {multi_select}")
    print(f"\n  Train groups: {len(train_g)}  (MAIN: {main_train})")
    print(f"  Val groups:   {len(val_g)}  (MAIN: {main_val})")
    total_main = main_train + main_val
    print(f"  TOTAL MAIN decisions:            {total_main}")

    if total_pairs < 50:
        print("ERROR: Too few decisions to train on!")
        sys.exit(1)

    # ── Step 3: stack arrays ───────────────────────────────────────────────────
    X_tr = np.vstack(train_X) if train_X else np.zeros((0, FEATURE_DIM), dtype=np.float32)
    y_tr = np.concatenate(train_y) if train_y else np.zeros(0, dtype=np.int8)
    g_tr = np.array(train_g, dtype=np.int32)
    w_tr = np.concatenate(train_w) if train_w else np.zeros(0, dtype=np.float32)
    c_tr = np.array(train_ctx, dtype=np.int32)

    X_va = np.vstack(val_X) if val_X else np.zeros((0, FEATURE_DIM), dtype=np.float32)
    y_va = np.concatenate(val_y) if val_y else np.zeros(0, dtype=np.int8)
    g_va = np.array(val_g, dtype=np.int32)
    # w_va = np.concatenate(val_w) if val_w else np.zeros(0, dtype=np.float32)
    c_va = np.array(val_ctx, dtype=np.int32)

    print(f"\n[3/5] Dataset sizes:")
    print(f"  Train: {len(g_tr)} groups, {len(X_tr)} rows")
    print(f"  Val:   {len(g_va)} groups, {len(X_va)} rows")

    # ── Step 4: train LightGBM lambdarank (no early stopping) ─────────────────
    print("\n[4/5] Training LightGBM lambdarank (Iono-only, no early stopping)...")

    try:
        import lightgbm as lgb
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
        import lightgbm as lgb

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
        "num_leaves": args.num_leaves,
        "learning_rate": args.lr,
        "min_child_samples": args.min_child_samples,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }

    # Always use fixed num_boost_round, no early stopping
    callbacks = [lgb.log_evaluation(period=50)]

    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval], valid_names=["val"],
        callbacks=callbacks,
    )

    out_model.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_model))
    print(f"\n  Model saved → {out_model}  (trees: {booster.num_trees()})")

    # ── Step 5: offline evaluation ─────────────────────────────────────────────
    print("\n[5/5] Offline evaluation (val set)...")

    scores_va = booster.predict(X_va.astype(np.float32))

    def top_k_acc(scores, y, groups, ctx, k=1):
        correct_overall = correct_main = n_total = n_main = 0
        pos = 0
        for gsz, gctx in zip(groups, ctx):
            sg = scores[pos:pos + gsz]
            yg = y[pos:pos + gsz]
            pos += gsz
            if len(sg) == 0:
                continue
            top_idx = np.argsort(sg)[::-1][:k]
            hit = bool(np.any(yg[top_idx] == 1))
            n_total += 1
            correct_overall += int(hit)
            if gctx == 0:
                n_main += 1
                correct_main += int(hit)
        return {
            f"top{k}_overall": correct_overall / n_total if n_total else 0.0,
            f"top{k}_main": correct_main / n_main if n_main else 0.0,
            "n_total": n_total, "n_main": n_main,
        }

    def option0_baseline(y, groups, ctx):
        correct_overall = correct_main = n_total = n_main = 0
        pos = 0
        for gsz, gctx in zip(groups, ctx):
            yg = y[pos:pos + gsz]
            pos += gsz
            if len(yg) == 0:
                continue
            hit = bool(yg[0] == 1)
            n_total += 1
            correct_overall += int(hit)
            if gctx == 0:
                n_main += 1
                correct_main += int(hit)
        return {
            "top1_overall": correct_overall / n_total if n_total else 0.0,
            "top1_main": correct_main / n_main if n_main else 0.0,
        }

    r1 = top_k_acc(scores_va, y_va, g_va, c_va, k=1)
    r3 = top_k_acc(scores_va, y_va, g_va, c_va, k=3)
    b0 = option0_baseline(y_va, g_va, c_va)

    sizes = g_va[g_va > 0]
    sizes_main = g_va[(g_va > 0) & (c_va == 0)]
    br_overall = 1.0 / sizes.mean() if len(sizes) else 0.0
    br_main = 1.0 / sizes_main.mean() if len(sizes_main) else 0.0

    print(f"\n  Val groups total: {r1['n_total']}  (MAIN: {r1['n_main']})")
    print(f"\n  Top-1 accuracy:")
    print(f"    Model overall:       {r1['top1_overall']:.4f}")
    print(f"    Model MAIN:          {r1['top1_main']:.4f}")
    print(f"    Baseline random:     {br_overall:.4f} (overall) / {br_main:.4f} (MAIN)")
    print(f"    Baseline option-0:   {b0['top1_overall']:.4f} (overall) / {b0['top1_main']:.4f} (MAIN)")
    print(f"\n  Top-3 accuracy:")
    print(f"    Model overall:       {r3['top3_overall']:.4f}")
    print(f"    Model MAIN:          {r3['top3_main']:.4f}")

    # Feature importance
    print("\n  Feature importance (gain, top 10):")
    imp = booster.feature_importance(importance_type="gain")
    feat_names = booster.feature_name()
    top_idx = np.argsort(imp)[::-1][:10]
    total_imp = imp.sum() + 1e-9
    for rank, i in enumerate(top_idx, 1):
        print(f"    {rank:2d}. {feat_names[i]:35s}  {imp[i]:.1f} ({100*imp[i]/total_imp:.1f}%)")

    print(f"\n  TOTAL MAIN decisions (train+val): {total_main}")
    print(f"\n  Model: {out_model}")
    print("  Done.")


if __name__ == "__main__":
    main()
