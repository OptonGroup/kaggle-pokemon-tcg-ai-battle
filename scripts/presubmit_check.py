#!/usr/bin/env python3
"""
presubmit_check.py — Simulate Kaggle's exec()-without-__file__ runtime.

Kaggle loads main.py via exec(), NOT importlib, which means __file__ is NOT
set in the module's namespace. This check catches NameError on __file__ usage
that would silently break in the Kaggle environment.

Usage:
    .venv/bin/python scripts/presubmit_check.py [submission_dir]

Default submission_dir: submissions/v007_crustle

Exits 0 on OK, 1 on FAIL.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent

DEFAULT_DIR = str(REPO / "submissions" / "v007_crustle")

submission_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
submission_dir = str(Path(submission_dir).resolve())

print(f"  Checking: {submission_dir}")

main_py = os.path.join(submission_dir, "main.py")
if not os.path.exists(main_py):
    print(f"FAIL: main.py not found at {main_py}")
    sys.exit(1)

# Change into submission dir — just like Kaggle's agent working directory
os.chdir(submission_dir)

with open("main.py", "r") as f:
    src = f.read()

ns: dict = {"__name__": "__main__"}
# Deliberately do NOT set __file__ in ns — this is the Kaggle exec() scenario

try:
    exec(compile(src, "main.py", "exec"), ns)
except Exception as exc:
    print(f"FAIL: exec of main.py raised: {exc}")
    sys.exit(1)

# Grab agent callable
agent = ns.get("agent")
if agent is None:
    print("FAIL: no 'agent' callable found in main.py namespace after exec")
    sys.exit(1)
if not callable(agent):
    print(f"FAIL: 'agent' is not callable (type={type(agent)})")
    sys.exit(1)

# Call agent with select=None (deck selection step)
try:
    result = agent({"select": None})
except Exception as exc:
    print(f"FAIL: agent({{'select': None}}) raised: {exc}")
    sys.exit(1)

if not isinstance(result, list):
    print(f"FAIL: agent returned {type(result)}, expected list")
    sys.exit(1)

if len(result) != 60:
    print(f"FAIL: agent returned {len(result)} card IDs, expected 60")
    sys.exit(1)

if not all(isinstance(x, int) for x in result):
    print(f"FAIL: deck contains non-int values: {result[:5]}...")
    sys.exit(1)

print(f"  [OK] exec() succeeded — no NameError on __file__")
print(f"  [OK] agent callable found")
print(f"  [OK] agent({{select: None}}) returned {len(result)} ints (deck)")
print(f"  [OK] First 5 card IDs: {result[:5]}")
print("\nOK")
sys.exit(0)
