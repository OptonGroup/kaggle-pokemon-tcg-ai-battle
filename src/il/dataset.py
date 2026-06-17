"""
dataset.py — Build IL training dataset from all replay files (P4-2).

Walks all parsed pairs, builds per-option feature matrix (features.py),
assigns label y=1 for chosen options, creates per-decision query groups,
weights by winner status and player skill (using μ prefix if available),
and splits by episode id into train/val (~85/15).

Saves to data/il/:
  train.npz   — X_train, y_train, groups_train, weights_train, ctx_train
  val.npz     — X_val, y_val, groups_val, weights_val, ctx_val

Usage:
  python -m src.il.dataset [--replay-dirs ...] [--val-frac 0.15]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from src.replays.parse import parse_replay, walk_replays
from src.il.features import build_feature_matrix, FEATURE_DIM

# ── Skill (μ) extraction heuristics ────────────────────────────────────────────
# We don't have a leaderboard CSV, but player names sometimes encode rough skill.
# Fallback: weight winners ×2, losers ×1.
WINNER_WEIGHT = 2.0
LOSER_WEIGHT = 1.0

_REPO_ROOT = Path(__file__).parent.parent.parent
_OUT_DIR = _REPO_ROOT / "data" / "il"


def _episode_to_val_flag(episode_id: int, val_frac: float = 0.15) -> bool:
    """
    Deterministic train/val split by episode id.
    Use hash(episode_id) % 100 < val_frac*100.
    """
    h = int(hashlib.md5(str(episode_id).encode()).hexdigest(), 16)
    return (h % 100) < int(val_frac * 100)


def build_dataset(
    replay_dirs: list[Path],
    val_frac: float = 0.15,
    skip_contexts: set[int] | None = None,
) -> dict:
    """
    Build full dataset from all replay files.

    Returns:
      {
        'X_train': ndarray (N_train, FEATURE_DIM),
        'y_train': ndarray (N_train,) int8,
        'groups_train': ndarray (G_train,) int32 — sizes of each query group,
        'weights_train': ndarray (N_train,) float32,
        'ctx_train': ndarray (G_train,) int32 — select_context per group,
        'X_val': ..., 'y_val': ..., 'groups_val': ..., 'weights_val': ..., 'ctx_val': ...,
        'stats': dict,
      }
    """
    if skip_contexts is None:
        skip_contexts = {41, 42}  # IS_FIRST, MULLIGAN

    replay_paths = walk_replays(replay_dirs)
    # Deduplicate by name
    seen = set()
    unique_paths = []
    for p in replay_paths:
        if p.name not in seen:
            seen.add(p.name)
            unique_paths.append(p)
    replay_paths = unique_paths

    print(f"Found {len(replay_paths)} unique replay files.")

    # Accumulators
    train_X, train_y, train_g, train_w, train_ctx = [], [], [], [], []
    val_X, val_y, val_g, val_w, val_ctx = [], [], [], [], []

    total_pairs = 0
    skipped_empty = 0
    skipped_ctx = 0
    parse_errors = 0
    multi_select = 0

    for path in replay_paths:
        try:
            pairs = parse_replay(path)
        except Exception as e:
            parse_errors += 1
            print(f"  WARN: parse error {path.name}: {e}")
            continue

        for pair in pairs:
            ctx = pair["select_context"]
            if ctx in skip_contexts:
                skipped_ctx += 1
                continue

            ep_id = pair["episodeId"]
            is_val = _episode_to_val_flag(ep_id, val_frac)
            is_winner = pair["won"]

            obs = pair["obs_dict"]
            chosen = set(pair["chosen_option_indices"])
            n_opts = pair["n_options"]

            if n_opts == 0:
                skipped_empty += 1
                continue

            try:
                X_group = build_feature_matrix(obs)
            except Exception as e:
                skipped_empty += 1
                continue

            if X_group.shape[0] == 0:
                skipped_empty += 1
                continue

            if X_group.shape[0] != n_opts:
                # Shape mismatch, skip
                skipped_empty += 1
                continue

            y_group = np.zeros(n_opts, dtype=np.int8)
            for ci in chosen:
                if ci < n_opts:
                    y_group[ci] = 1

            # Per-group weight (same for all options in group)
            w = WINNER_WEIGHT if is_winner else LOSER_WEIGHT

            # Track multi-select decisions
            if len(chosen) > 1:
                multi_select += 1

            total_pairs += 1

            if is_val:
                val_X.append(X_group)
                val_y.append(y_group)
                val_g.append(n_opts)
                val_w.append(np.full(n_opts, w, dtype=np.float32))
                val_ctx.append(ctx)
            else:
                train_X.append(X_group)
                train_y.append(y_group)
                train_g.append(n_opts)
                train_w.append(np.full(n_opts, w, dtype=np.float32))
                train_ctx.append(ctx)

    # Stack arrays
    X_tr = np.vstack(train_X) if train_X else np.zeros((0, FEATURE_DIM), dtype=np.float32)
    y_tr = np.concatenate(train_y) if train_y else np.zeros(0, dtype=np.int8)
    g_tr = np.array(train_g, dtype=np.int32)
    w_tr = np.concatenate(train_w) if train_w else np.zeros(0, dtype=np.float32)
    c_tr = np.array(train_ctx, dtype=np.int32)

    X_va = np.vstack(val_X) if val_X else np.zeros((0, FEATURE_DIM), dtype=np.float32)
    y_va = np.concatenate(val_y) if val_y else np.zeros(0, dtype=np.int8)
    g_va = np.array(val_g, dtype=np.int32)
    w_va = np.concatenate(val_w) if val_w else np.zeros(0, dtype=np.float32)
    c_va = np.array(val_ctx, dtype=np.int32)

    stats = {
        "total_pairs": total_pairs,
        "train_groups": len(g_tr),
        "val_groups": len(g_va),
        "train_rows": len(X_tr),
        "val_rows": len(X_va),
        "skipped_empty": skipped_empty,
        "skipped_ctx": skipped_ctx,
        "parse_errors": parse_errors,
        "multi_select": multi_select,
        "feature_dim": FEATURE_DIM,
    }

    return {
        "X_train": X_tr, "y_train": y_tr, "groups_train": g_tr,
        "weights_train": w_tr, "ctx_train": c_tr,
        "X_val": X_va, "y_val": y_va, "groups_val": g_va,
        "weights_val": w_va, "ctx_val": c_va,
        "stats": stats,
    }


def save_dataset(ds: dict, out_dir: Path = _OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "train.npz",
        X=ds["X_train"], y=ds["y_train"],
        groups=ds["groups_train"], weights=ds["weights_train"], ctx=ds["ctx_train"],
    )
    np.savez_compressed(
        out_dir / "val.npz",
        X=ds["X_val"], y=ds["y_val"],
        groups=ds["groups_val"], weights=ds["weights_val"], ctx=ds["ctx_val"],
    )
    print(f"Saved train.npz and val.npz to {out_dir}/")


def load_dataset(out_dir: Path = _OUT_DIR) -> dict:
    tr = np.load(out_dir / "train.npz")
    va = np.load(out_dir / "val.npz")
    return {
        "X_train": tr["X"], "y_train": tr["y"],
        "groups_train": tr["groups"], "weights_train": tr["weights"], "ctx_train": tr["ctx"],
        "X_val": va["X"], "y_val": va["y"],
        "groups_val": va["groups"], "weights_val": va["weights"], "ctx_val": va["ctx"],
    }


def main():
    parser = argparse.ArgumentParser(description="Build IL dataset from replay files")
    parser.add_argument(
        "--replay-dirs", nargs="*",
        default=["data/episodes/top_players", "data/episodes"],
        help="Directories containing *-replay.json files",
    )
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--out-dir", default=str(_OUT_DIR))
    args = parser.parse_args()

    dirs = []
    for d in args.replay_dirs:
        p = Path(d)
        if not p.is_absolute():
            p = _REPO_ROOT / p
        dirs.append(p)

    print(f"Building dataset from: {dirs}")
    ds = build_dataset(dirs, val_frac=args.val_frac)

    stats = ds["stats"]
    print("\n=== Dataset Stats ===")
    print(f"  Total pairs processed:  {stats['total_pairs']}")
    print(f"  Train groups:           {stats['train_groups']}")
    print(f"  Val groups:             {stats['val_groups']}")
    print(f"  Train rows (options):   {stats['train_rows']}")
    print(f"  Val rows (options):     {stats['val_rows']}")
    print(f"  Feature dim:            {stats['feature_dim']}")
    print(f"  Multi-select decisions: {stats['multi_select']}")
    print(f"  Skipped (empty/err):    {stats['skipped_empty']}")
    print(f"  Skipped (ctx):          {stats['skipped_ctx']}")
    print(f"  Parse errors:           {stats['parse_errors']}")

    save_dataset(ds, Path(args.out_dir))


if __name__ == "__main__":
    main()
