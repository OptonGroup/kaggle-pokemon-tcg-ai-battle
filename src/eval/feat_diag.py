#!/usr/bin/env python3
"""Quick LOCAL diagnostic for Iono features (no lightgbm/libomp needed).

Runs build_feature_matrix on real onechan1 winner-side MAIN obs and checks:
  - exceptions / shape mismatches
  - degenerate (all-zero) rows
  - per-slot %nonzero & mean, focused on the new Iono slots (35,36,37,57,58,59,60)
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from replays.parse import parse_replay, walk_replays
from il.features_iono import build_feature_matrix, FEATURE_DIM

IONO_IDS = {265, 268, 269, 270, 271}
DIRS = [REPO / "data" / "episodes" / "top_players",
        REPO / "data" / "episodes" / "all_teams",
        REPO / "data" / "episodes"]

paths = []
seen = set()
for p in walk_replays(DIRS):
    if p.name not in seen:
        seen.add(p.name)
        paths.append(p)
print(f"replays: {len(paths)}")

n_exc = n_shape = n_ok = n_zero_rows = 0
main_count = 0
slot_nonzero = np.zeros(FEATURE_DIM)
slot_sum = np.zeros(FEATURE_DIM)
n_rows = 0
NEW_SLOTS = [35, 36, 37, 38, 39, 40, 41, 42, 62, 63, 64, 65]

for path in paths:
    try:
        pairs = parse_replay(path)
    except Exception:
        continue
    for pr in pairs:
        if pr.get("player_name") != "onechan1" or not pr["won"]:
            continue
        if pr["select_context"] != 0:  # MAIN only
            continue
        main_count += 1
        obs = pr["obs_dict"]
        n_opts = pr["n_options"]
        try:
            X = build_feature_matrix(obs)
        except Exception as e:
            n_exc += 1
            if n_exc <= 3:
                print(f"  EXC {path.name}: {type(e).__name__}: {e}")
            continue
        if X.shape[0] != n_opts or X.shape[1] != FEATURE_DIM:
            n_shape += 1
            if n_shape <= 3:
                print(f"  SHAPE {path.name}: got {X.shape}, n_opts={n_opts}")
            continue
        n_ok += 1
        n_zero_rows += int(np.sum(np.all(X == 0, axis=1)))
        slot_nonzero += (X != 0).sum(axis=0)
        slot_sum += X.sum(axis=0)
        n_rows += X.shape[0]

print(f"\nonechan1 winner MAIN decisions: {main_count}")
print(f"  feature build OK: {n_ok}   exceptions: {n_exc}   shape-mismatch: {n_shape}")
print(f"  total option-rows: {n_rows}   all-zero rows: {n_zero_rows}")
if n_rows:
    from il.features_iono import FEATURE_NAMES
    print(f"\n  Iono slots (%nonzero across all option-rows / mean):")
    for s in NEW_SLOTS:
        print(f"    slot {s:2d} {FEATURE_NAMES[s]:32s}: {100*slot_nonzero[s]/n_rows:5.1f}% nonzero   mean={slot_sum[s]/n_rows:.3f}")
