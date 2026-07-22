#!/usr/bin/env python3
"""Extract the per-step decision trace from raw Kaggle agent-log JSONs.

Raw format (downloaded via kaggle episodes API): a JSON list, one entry per
engine step; each entry is a list of dicts with keys duration/stdout/stderr.
Our agents print the [dec]/[alt]/[galt]/[dbg] decision trace to stderr.

Output: one .txt per episode, each line prefixed with `stepNNN `, containing
every non-empty stderr/stdout line. Empty output file == the agent printed
nothing (e.g. not our submission's log).

Usage:
    python3 scripts/extract_agent_logs.py <download_dir>
        # reads  <download_dir>/agent_logs_raw/*.json
        # writes <download_dir>/agent_logs_extracted/*.txt
"""
import glob
import json
import os
import sys


def extract(raw_path: str) -> str:
    with open(raw_path) as f:
        d = json.load(f)
    lines = []
    for i, step in enumerate(d):
        for entry in step:
            for stream in ("stderr", "stdout"):
                txt = entry.get(stream) or ""
                if txt.strip():
                    for ln in txt.rstrip("\n").split("\n"):
                        lines.append(f"step{i:03d} {ln}")
    return "\n".join(lines) + ("\n" if lines else "")


def main():
    dl = sys.argv[1]
    raw_dir = os.path.join(dl, "agent_logs_raw")
    out_dir = os.path.join(dl, "agent_logs_extracted")
    os.makedirs(out_dir, exist_ok=True)
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
    n_empty = 0
    for rf in raw_files:
        txt = extract(rf)
        if not txt:
            n_empty += 1
        base = os.path.basename(rf).rsplit(".", 1)[0] + ".txt"
        with open(os.path.join(out_dir, base), "w") as f:
            f.write(txt)
    print(f"extracted {len(raw_files)} logs -> {out_dir} ({n_empty} empty)")


if __name__ == "__main__":
    main()
