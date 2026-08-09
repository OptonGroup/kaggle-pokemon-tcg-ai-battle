#!/usr/bin/env python3
"""diag_v060_per_expert.py — per-expert val top-1 diagnostic for v060 Track B models.

Loads data/il_grimmsnarl_v2/pairs.jsonl (same filtering as
scripts/train_grimmsnarl_v2_il.py: player in {jiatu.l, bono}, won==True,
select_context == MAIN(0)), rebuilds the SAME deterministic
hash(episodeId)-based val split, and re-derives per-group features once.
Then scores that fixed val set under three already-trained boosters
(models/il_grimmsnarl_v2.txt = joint, models/il_grimmsnarl_v2_jiatuonly.txt,
models/il_grimmsnarl_v2_bonoonly.txt), reporting top-1 accuracy broken out
by which expert (player_name) produced each val group.

Output: prints a table + writes results/gates/v060_per_expert_diag.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

from il.features_grimmsnarl_v2 import build_feature_matrix, FEATURE_DIM

PAIRS_PATH = REPO / "data" / "il_grimmsnarl_v2" / "pairs.jsonl"
MODELS = {
    "joint": REPO / "models" / "il_grimmsnarl_v2.txt",
    "jiatuonly": REPO / "models" / "il_grimmsnarl_v2_jiatuonly.txt",
    "bonoonly": REPO / "models" / "il_grimmsnarl_v2_bonoonly.txt",
}
EXPERT_NAMES = {"jiatu.l", "bono"}
VAL_FRAC = 0.15
SKIP_CONTEXTS = {41, 42}
MAIN_CONTEXT = 0


def episode_to_val_flag(episode_id) -> bool:
    h = int(hashlib.md5(str(episode_id).encode()).hexdigest(), 16)
    return (h % 100) < int(VAL_FRAC * 100)


def main():
    val_X, val_y, val_g, val_player = [], [], [], []
    n_lines = 0
    with open(PAIRS_PATH) as f:
        for line in f:
            n_lines += 1
            try:
                pair = json.loads(line)
            except Exception:
                continue
            pname = pair.get("player_name")
            if pname not in EXPERT_NAMES:
                continue
            if not pair.get("won"):
                continue
            ctx = pair["select_context"]
            if ctx in SKIP_CONTEXTS or ctx != MAIN_CONTEXT:
                continue
            if not episode_to_val_flag(pair["episodeId"]):
                continue  # train row, skip
            n_opts = pair["n_options"]
            if n_opts == 0:
                continue
            obs = pair["obs_dict"]
            chosen = set(pair["chosen_option_indices"])
            try:
                X_group = build_feature_matrix(obs)
            except Exception:
                continue
            if X_group.shape[0] == 0 or X_group.shape[0] != n_opts:
                continue
            y_group = np.zeros(n_opts, dtype=np.int8)
            for ci in chosen:
                if ci < n_opts:
                    y_group[ci] = 1
            if y_group.sum() == 0:
                continue
            val_X.append(X_group)
            val_y.append(y_group)
            val_g.append(n_opts)
            val_player.append(pname)

    print(f"n_lines={n_lines}  val_groups={len(val_g)}")
    by_player_n = {}
    for p in val_player:
        by_player_n[p] = by_player_n.get(p, 0) + 1
    print(f"val groups by player: {by_player_n}")

    X_va = np.vstack(val_X).astype(np.float32)
    assert X_va.shape[1] == FEATURE_DIM

    import lightgbm as lgb

    results = {}
    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"  [skip] {model_name}: {model_path} not found")
            continue
        booster = lgb.Booster(model_file=str(model_path))
        scores_va = booster.predict(X_va)

        pos = 0
        per_player_correct = {}
        per_player_n = {}
        overall_correct = overall_n = 0
        for gsz, yg_arr, pname in zip(val_g, val_y, val_player):
            sg = scores_va[pos:pos + gsz]
            pos += gsz
            top_idx = int(np.argmax(sg))
            correct = int(yg_arr[top_idx] == 1)
            overall_n += 1
            overall_correct += correct
            per_player_n[pname] = per_player_n.get(pname, 0) + 1
            per_player_correct[pname] = per_player_correct.get(pname, 0) + correct

        top1_overall = overall_correct / overall_n if overall_n else 0.0
        top1_by_player = {
            p: (per_player_correct[p] / per_player_n[p] if per_player_n[p] else 0.0)
            for p in per_player_n
        }
        results[model_name] = {
            "top1_overall": top1_overall,
            "n_overall": overall_n,
            "top1_by_player": top1_by_player,
            "n_by_player": per_player_n,
        }
        print(f"\n== model={model_name} ==")
        print(f"  overall top-1: {top1_overall:.4f}  (n={overall_n})")
        for p, acc in top1_by_player.items():
            print(f"  {p:10s} top-1: {acc:.4f}  (n={per_player_n[p]})")

    out_path = REPO / "results" / "gates" / "v060_per_expert_diag.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"val_groups_by_player": by_player_n, "results": results}, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
