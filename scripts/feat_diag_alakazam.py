#!/usr/bin/env python3
"""feat_diag_alakazam.py — verify Alakazam-specific feature slots fire on aidy obs."""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from replays.parse import parse_replay, walk_replays
from il.features_alakazam import build_feature_matrix, FEATURE_DIM, FEATURE_NAMES

EXPERT_NAME = "aidy"
ALAKAZAM_SIG = {741, 742, 743}
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
# identify aidy alakazam episodes
aidy_paths = []
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
                if ALAKAZAM_SIG.issubset(set(act)) and pi < len(names) and names[pi] == EXPERT_NAME:
                    aidy_paths.append((p, pi))
                    break
    except Exception:
        pass
print(f"aidy Alakazam episodes: {len(aidy_paths)}")

n_exc = n_ok = n_zero_rows = 0
main_count = 0
slot_nonzero = np.zeros(FEATURE_DIM)
slot_sum = np.zeros(FEATURE_DIM)
n_rows = 0
# Alakazam-specific slots (local 33-42 + global 61-65)
NEW_SLOTS = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 61, 62, 63, 64, 65]

for path, pi in aidy_paths:
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

print(f"\naidy winner MAIN decisions: {main_count}")
print(f"  feature build OK: {n_ok}  exceptions: {n_exc}  total rows: {n_rows}  all-zero rows: {n_zero_rows}")
if n_rows:
    print(f"\n  Alakazam-specific slots (%nonzero / mean):")
    for s in NEW_SLOTS:
        print(f"    slot {s:2d} {FEATURE_NAMES[s]:42s}: {100*slot_nonzero[s]/n_rows:5.1f}%  mean={slot_sum[s]/n_rows:.3f}")
    print(f"\n  Top-8 nonzero slots overall:")
    for s in np.argsort(slot_nonzero)[::-1][:8]:
        print(f"    slot {s:2d} {FEATURE_NAMES[s]:42s}: {100*slot_nonzero[s]/n_rows:5.1f}%")