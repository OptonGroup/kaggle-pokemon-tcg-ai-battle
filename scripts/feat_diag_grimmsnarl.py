#!/usr/bin/env python3
"""feat_diag_grimmsnarl.py — verify Grimmsnarl-specific feature slots fire on kazuki0123 obs."""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from replays.parse import parse_replay, walk_replays
from il.features_grimmsnarl import build_feature_matrix, FEATURE_DIM, FEATURE_NAMES

EXPERT_NAME = "kazuki0123"
GRIMMSNARL_SIG = {648}
DIRS = [REPO / "data" / "episodes" / "all_teams",
        REPO / "data" / "episodes" / "top_players",
        REPO / "data" / "episodes"]

paths = []
seen = set()
for p in walk_replays(DIRS):
    if p.name not in seen:
        seen.add(p.name); paths.append(p)
print(f"replays: {len(paths)}")

import json
kazuki_paths = []
for p in paths:
    try:
        with open(p) as f:
            d = json.load(f)
        agents = d.get("info", {}).get("Agents", [])
        names = [a.get("Name") for a in agents]
        steps = d.get("steps", [])
        if len(steps) < 2:
            continue
        for pi in range(min(2, len(steps[1]))):
            act = steps[1][pi].get("action", [])
            if len(act) == 60 and all(isinstance(x, int) for x in act):
                if GRIMMSNARL_SIG.issubset(set(act)) and pi < len(names) and names[pi] == EXPERT_NAME:
                    kazuki_paths.append((p, pi))
                    break
    except Exception:
        pass
print(f"kazuki0123 Grimmsnarl episodes: {len(kazuki_paths)}")

n_exc = n_ok = n_zero_rows = 0
main_count = 0
slot_nonzero = np.zeros(FEATURE_DIM)
slot_sum = np.zeros(FEATURE_DIM)
n_rows = 0
# Grimmsnarl-specific slots: local 35-42, global 61-71
NEW_SLOTS = list(range(35, 43)) + list(range(61, 72))

for path, pi in kazuki_paths:
    try:
        pairs = parse_replay(path)
    except Exception:
        continue
    for pr in pairs:
        if pr["player_index"] != pi or not pr["won"] or pr["select_context"] != 0:
            continue
        main_count += 1
        obs = pr["obs_dict"]; n_opts = pr["n_options"]
        try:
            X = build_feature_matrix(obs)
        except Exception as e:
            n_exc += 1
            if n_exc <= 3: print(f"  EXC {path.name}: {type(e).__name__}: {e}")
            continue
        if X.shape[0] != n_opts or X.shape[1] != FEATURE_DIM:
            print(f"  SHAPE {path.name}: {X.shape} vs n_opts={n_opts}")
            continue
        n_ok += 1
        n_zero_rows += int(np.sum(np.all(X == 0, axis=1)))
        slot_nonzero += (X != 0).sum(axis=0)
        slot_sum += X.sum(axis=0)
        n_rows += X.shape[0]

print(f"\nkazuki0123 winner MAIN decisions: {main_count}")
print(f"  feature build OK: {n_ok}  exceptions: {n_exc}  total rows: {n_rows}  all-zero rows: {n_zero_rows}")
if n_rows:
    print(f"\n  Grimmsnarl-specific slots (%nonzero / mean):")
    for s in NEW_SLOTS:
        print(f"    slot {s:2d} {FEATURE_NAMES[s]:42s}: {100*slot_nonzero[s]/n_rows:5.1f}%  mean={slot_sum[s]/n_rows:.3f}")
    print(f"\n  Top-12 nonzero slots overall:")
    for s in np.argsort(slot_nonzero)[::-1][:12]:
        print(f"    slot {s:2d} {FEATURE_NAMES[s]:42s}: {100*slot_nonzero[s]/n_rows:5.1f}%")
