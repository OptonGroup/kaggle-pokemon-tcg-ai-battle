#!/usr/bin/env python3
"""
test_feature_parity.py — Assert that src/il/features.py and
submissions/v007_crustle/features.py produce bit-identical feature matrices
for real replay decisions.

Usage:
    .venv/bin/python scripts/test_feature_parity.py

Exits 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent

# ── Import TRAINING module ────────────────────────────────────────────────────
sys.path.insert(0, str(REPO / "src"))
import il.features as train_feat

# ── Import SERVING module via importlib to avoid name collision ──────────────
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "serve_feat",
    str(REPO / "submissions" / "v007_crustle" / "features.py"),
)
serve_feat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve_feat)

# ── 1. Check FEATURE_DIM ─────────────────────────────────────────────────────
print(f"  train FEATURE_DIM = {train_feat.FEATURE_DIM}")
print(f"  serve FEATURE_DIM = {serve_feat.FEATURE_DIM}")
assert train_feat.FEATURE_DIM == serve_feat.FEATURE_DIM, (
    f"FEATURE_DIM mismatch: {train_feat.FEATURE_DIM} != {serve_feat.FEATURE_DIM}"
)
print("  [OK] FEATURE_DIM match")

# ── 2. Check FEATURE_NAMES ────────────────────────────────────────────────────
assert train_feat.FEATURE_NAMES == serve_feat.FEATURE_NAMES, (
    "FEATURE_NAMES mismatch:\n"
    f"  train: {train_feat.FEATURE_NAMES}\n"
    f"  serve: {serve_feat.FEATURE_NAMES}"
)
print(f"  [OK] FEATURE_NAMES match ({len(train_feat.FEATURE_NAMES)} names)")

# ── 3. Find a real replay and extract Crustle decisions ───────────────────────
TOP_DIR = REPO / "data" / "episodes" / "top_players"
replay_paths = sorted(TOP_DIR.glob("*-replay.json"))
if not replay_paths:
    print("FAIL: no replay files found in", TOP_DIR)
    sys.exit(1)

sys.path.insert(0, str(REPO / "src"))
from replays.parse import parse_replay

CRUSTLE_IDS = {344, 345}

decisions_tested = 0
errors = []

for path in replay_paths:
    try:
        with open(path) as f:
            data = json.load(f)

        # Check if this replay has a Crustle player
        steps = data.get("steps", [])
        if len(steps) < 2:
            continue
        has_crustle = False
        for pi in range(min(2, len(steps[1]))):
            action = steps[1][pi].get("action", [])
            if len(action) == 60 and any(x in CRUSTLE_IDS for x in action):
                has_crustle = True
                break
        if not has_crustle:
            continue

        pairs = parse_replay(path)
        for pair in pairs:
            obs = pair["obs_dict"]
            if obs.get("select") is None:
                continue
            opts = (obs.get("select") or {}).get("option") or []
            if len(opts) == 0:
                continue

            X_train = train_feat.build_feature_matrix(obs)
            X_serve = serve_feat.build_feature_matrix(obs)

            if X_train.shape != X_serve.shape:
                errors.append(
                    f"Shape mismatch in {path.name}: "
                    f"train={X_train.shape} serve={X_serve.shape}"
                )
                continue

            if not np.allclose(X_train, X_serve, rtol=0, atol=0):
                diff_idx = np.where(~np.isclose(X_train, X_serve, rtol=0, atol=0))
                errors.append(
                    f"Value mismatch in {path.name}: "
                    f"first diff at indices {diff_idx[0][:3]}, {diff_idx[1][:3]}"
                )
                continue

            decisions_tested += 1

    except Exception as e:
        # Skip unparseable replays
        continue

    if decisions_tested >= 100:
        break

if errors:
    print(f"\nFAIL: {len(errors)} error(s):")
    for e in errors[:5]:
        print(f"  {e}")
    sys.exit(1)

if decisions_tested == 0:
    print("FAIL: no Crustle decisions found to test")
    sys.exit(1)

print(f"  [OK] {decisions_tested} decisions tested — train/serve outputs are element-wise identical")
print("\nPASS")
sys.exit(0)
