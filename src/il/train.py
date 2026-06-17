"""
train.py — Train LightGBM ranker on IL dataset and report offline accuracy (P4-3).

Model choice: LightGBM **lambdarank** (objective=lambdarank, group=per-decision).
  - Each decision is one "query group"; options within it are ranked.
  - lambdarank directly optimizes NDCG which is closely related to top-k accuracy.
  - Alternative considered: binary crossentropy per option (pointwise). Lambdarank is
    better because it explicitly learns to rank the chosen option above others within
    each decision, avoiding the extreme class imbalance issue (1 positive / ~6 negatives
    per group — CBE sees this as imbalanced, lambdarank frames it as within-group ranking).

Offline eval:
  - top-1 accuracy: fraction of decisions where argmax(score) IN chosen_option_indices
  - top-3 accuracy: fraction where any of top-3 scored options IN chosen_option_indices
  - Baselines: random pick (1/avg_n_opts), always pick option 0

Saves model to models/il_ranker.txt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent.parent
_MODEL_PATH = _REPO_ROOT / "models" / "il_ranker.txt"
_DATA_DIR = _REPO_ROOT / "data" / "il"

# ── LightGBM import ────────────────────────────────────────────────────────────

try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not found, attempting pip install...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
    import lightgbm as lgb


from src.il.dataset import load_dataset, build_dataset, save_dataset
from src.il.features import FEATURE_NAMES


def eval_top_k(
    scores: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    ctx: np.ndarray,
    k: int = 1,
    main_ctx: int = 0,
) -> dict:
    """
    Compute top-k accuracy overall and for MAIN context (ctx=0).

    Args:
        scores: (N,) float array of model scores.
        y: (N,) int8 labels (1=chosen, 0=not chosen).
        groups: (G,) int array of group sizes.
        ctx: (G,) int array of select_context per group.
        k: top-k to evaluate.
        main_ctx: context value for MAIN.

    Returns:
        dict with keys: top{k}_overall, top{k}_main, n_groups, n_groups_main.
    """
    correct_overall = 0
    correct_main = 0
    n_total = 0
    n_main = 0

    pos = 0
    for g, (gsz, gctx) in enumerate(zip(groups, ctx)):
        score_g = scores[pos : pos + gsz]
        y_g = y[pos : pos + gsz]
        pos += gsz

        if len(score_g) == 0:
            continue

        # Get top-k option indices by score
        top_indices = np.argsort(score_g)[::-1][:k]
        # Check if any top-k index has label=1
        hit = bool(np.any(y_g[top_indices] == 1))

        n_total += 1
        correct_overall += int(hit)
        if gctx == main_ctx:
            n_main += 1
            correct_main += int(hit)

    acc_overall = correct_overall / n_total if n_total else 0.0
    acc_main = correct_main / n_main if n_main else 0.0
    return {
        f"top{k}_overall": acc_overall,
        f"top{k}_main": acc_main,
        "n_groups": n_total,
        "n_groups_main": n_main,
    }


def baseline_option0(y: np.ndarray, groups: np.ndarray, ctx: np.ndarray, main_ctx: int = 0) -> dict:
    """Baseline: always pick option 0."""
    correct_overall = 0
    correct_main = 0
    n_total = 0
    n_main = 0

    pos = 0
    for gsz, gctx in zip(groups, ctx):
        y_g = y[pos : pos + gsz]
        pos += gsz
        if len(y_g) == 0:
            continue
        hit = bool(y_g[0] == 1)
        n_total += 1
        correct_overall += int(hit)
        if gctx == main_ctx:
            n_main += 1
            correct_main += int(hit)

    return {
        "top1_overall": correct_overall / n_total if n_total else 0.0,
        "top1_main": correct_main / n_main if n_main else 0.0,
        "n_groups": n_total,
        "n_groups_main": n_main,
    }


def baseline_random(groups: np.ndarray, ctx: np.ndarray, main_ctx: int = 0) -> dict:
    """Baseline: random pick = 1/avg_n_options expected accuracy."""
    sizes = groups[groups > 0]
    avg_n = sizes.mean() if len(sizes) else 1.0
    sizes_main = groups[(groups > 0) & (ctx == main_ctx)]
    avg_n_main = sizes_main.mean() if len(sizes_main) else 1.0
    return {
        "top1_overall": 1.0 / avg_n,
        "top1_main": 1.0 / avg_n_main,
        "n_groups": len(sizes),
        "n_groups_main": len(sizes_main),
    }


def train_ranker(
    ds: dict,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    seed: int = 42,
) -> lgb.Booster:
    """
    Train a LightGBM lambdarank model.

    Returns the trained booster.
    """
    X_tr = ds["X_train"].astype(np.float32)
    y_tr = ds["y_train"].astype(np.int32)
    g_tr = ds["groups_train"].astype(np.int32)
    w_tr = ds["weights_train"].astype(np.float32)

    X_va = ds["X_val"].astype(np.float32)
    y_va = ds["y_val"].astype(np.int32)
    g_va = ds["groups_val"].astype(np.int32)

    # LightGBM lambdarank dataset
    dtrain = lgb.Dataset(
        X_tr, label=y_tr, group=g_tr, weight=w_tr,
        feature_name=FEATURE_NAMES,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_va, label=y_va, group=g_va,
        feature_name=FEATURE_NAMES,
        reference=dtrain,
        free_raw_data=False,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3],
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": seed,
        "n_jobs": -1,
    }

    print(f"Training LightGBM lambdarank: n_estimators={n_estimators}, lr={learning_rate}, leaves={num_leaves}")
    print(f"  Train: {len(g_tr)} groups, {len(X_tr)} rows")
    print(f"  Val:   {len(g_va)} groups, {len(X_va)} rows")

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50, verbose=True),
    ]

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )

    return booster


def main():
    parser = argparse.ArgumentParser(description="Train IL ranker + report offline accuracy")
    parser.add_argument("--rebuild-data", action="store_true", help="Rebuild dataset from replays even if cached")
    parser.add_argument("--replay-dirs", nargs="*", default=["data/episodes/top_players", "data/episodes"])
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Load or build dataset ──────────────────────────────────────────────────
    train_path = _DATA_DIR / "train.npz"
    val_path = _DATA_DIR / "val.npz"

    if not args.rebuild_data and train_path.exists() and val_path.exists():
        print("Loading cached dataset from data/il/...")
        ds = load_dataset(_DATA_DIR)
        print(f"  Train: {len(ds['groups_train'])} groups, {len(ds['X_train'])} rows")
        print(f"  Val:   {len(ds['groups_val'])} groups, {len(ds['X_val'])} rows")
    else:
        dirs = []
        for d in args.replay_dirs:
            p = Path(d)
            if not p.is_absolute():
                p = _REPO_ROOT / p
            dirs.append(p)
        print(f"Building dataset from {dirs}...")
        ds = build_dataset(dirs, val_frac=args.val_frac)
        stats = ds["stats"]
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Train groups: {stats['train_groups']}, rows: {stats['train_rows']}")
        print(f"  Val groups:   {stats['val_groups']}, rows: {stats['val_rows']}")
        print(f"  Multi-select: {stats['multi_select']}, Feature dim: {stats['feature_dim']}")
        save_dataset(ds, _DATA_DIR)

    # ── Train ─────────────────────────────────────────────────────────────────
    booster = train_ranker(
        ds,
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
        num_leaves=args.num_leaves,
        seed=args.seed,
    )

    # Save model
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(_MODEL_PATH))
    print(f"\nModel saved to {_MODEL_PATH}")

    # ── Offline evaluation ────────────────────────────────────────────────────
    print("\n=== OFFLINE EVALUATION (val set) ===")

    y_va = ds["y_val"].astype(np.int32)
    g_va = ds["groups_val"].astype(np.int32)
    c_va = ds["ctx_val"].astype(np.int32)
    X_va = ds["X_val"].astype(np.float32)

    scores_va = booster.predict(X_va)

    # Model accuracy
    r1 = eval_top_k(scores_va, y_va, g_va, c_va, k=1)
    r3 = eval_top_k(scores_va, y_va, g_va, c_va, k=3)

    # Baselines
    b0 = baseline_option0(y_va, g_va, c_va)
    br = baseline_random(g_va, c_va)

    print(f"\n  Val groups: {r1['n_groups']} total, {r1['n_groups_main']} MAIN")
    print(f"\n  Top-1 accuracy:")
    print(f"    Model overall: {r1['top1_overall']:.4f}")
    print(f"    Model MAIN:    {r1['top1_main']:.4f}")
    print(f"    Baseline random overall: {br['top1_overall']:.4f}")
    print(f"    Baseline random MAIN:    {br['top1_main']:.4f}")
    print(f"    Baseline option-0 overall: {b0['top1_overall']:.4f}")
    print(f"    Baseline option-0 MAIN:    {b0['top1_main']:.4f}")
    print(f"\n  Top-3 accuracy:")
    print(f"    Model overall: {r3['top3_overall']:.4f}")
    print(f"    Model MAIN:    {r3['top3_main']:.4f}")

    # ── Feature importance ────────────────────────────────────────────────────
    print("\n=== FEATURE IMPORTANCE (gain, top 12) ===")
    imp = booster.feature_importance(importance_type="gain")
    feat_names = booster.feature_name()
    top_idx = np.argsort(imp)[::-1][:12]
    total_imp = imp.sum() + 1e-9
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feat_names[i]:35s}  gain={imp[i]:.1f}  ({100*imp[i]/total_imp:.1f}%)")

    # ── Also eval on TRAIN for overfit check ─────────────────────────────────
    print("\n=== TRAIN SET (overfit check) ===")
    y_tr = ds["y_train"].astype(np.int32)
    g_tr = ds["groups_train"].astype(np.int32)
    c_tr = ds["ctx_train"].astype(np.int32)
    scores_tr = booster.predict(ds["X_train"].astype(np.float32))
    r1_tr = eval_top_k(scores_tr, y_tr, g_tr, c_tr, k=1)
    print(f"  Top-1 train overall: {r1_tr['top1_overall']:.4f}  MAIN: {r1_tr['top1_main']:.4f}")

    # ── Lucario archetype fraction in top-player games ────────────────────────
    print("\n=== LUCARIO ARCHETYPE IN TOP-PLAYER GAMES ===")
    _count_lucario_games()


def _count_lucario_games():
    """
    Count how many of the 189 top-player games have any card id in 673-678
    (Lucario archetype: Makuhita, Hariyama, Lunatone, Solrock, Riolu, Mega Lucario ex).
    """
    from src.replays.parse import walk_replays, revealed_cards

    top_dir = _REPO_ROOT / "data" / "episodes" / "top_players"
    paths = walk_replays([top_dir])
    seen = set()
    unique = []
    for p in paths:
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p)

    lucario_ids = set(range(673, 679))  # 673-678 inclusive
    lucario_games = 0
    total = 0

    for path in unique:
        try:
            rev = revealed_cards(path)
        except Exception:
            continue
        total += 1
        has_lucario = any(
            lucario_ids.intersection(set(counter.keys()))
            for counter in rev.values()
        )
        if has_lucario:
            lucario_games += 1

    pct = 100 * lucario_games / total if total else 0
    print(f"  Top-player games: {total}")
    print(f"  Games with Lucario archetype (ids 673-678): {lucario_games} ({pct:.1f}%)")
    print(f"  Games WITHOUT Lucario: {total - lucario_games} ({100-pct:.1f}%)")


if __name__ == "__main__":
    main()
