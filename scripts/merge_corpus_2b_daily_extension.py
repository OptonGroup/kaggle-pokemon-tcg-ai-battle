#!/usr/bin/env python3
"""merge_corpus_2b_daily_extension.py — [v082 A] merge the base 2B corpus
(out/corpus_2b/decisions.jsonl + obs_shards + split/split.json) with the daily-zip extension
built by build_corpus_2b_daily_extension.py, WITHOUT modifying anything under the frozen
out/corpus_2b/ tree (that tree is the baseline other work depends on — see task instructions:
don't touch it, don't touch the frozen test split).

Output (all under --out-dir, default out/corpus_2b/versions/v082_bc2b/daily_merged):
  decisions.jsonl   — base decisions.jsonl rows + extension rows (extension episode_ids are
                       guaranteed disjoint from both the base corpus and the test split by
                       construction, see build_corpus_2b_daily_extension.py's dedup gates —
                       re-checked here defensively, hard-fails on any collision).
  obs_shards/        — SYMLINKS to each per-episode obs_shard file in both source trees (no
                        copy — these are decision-only obs, not the full multi-MB replay, but
                        still no reason to duplicate bytes on disk).
  split/split.json   — test_episode_ids UNCHANGED (byte-identical to the base split file, this
                        is the frozen held-out set); train_episode_ids = base train ids + every
                        NEW qualifying episode_id from the extension (new episodes only ever
                        enter TRAIN, per task instructions — the test split is never grown).

Usage:
    .venv/bin/python scripts/merge_corpus_2b_daily_extension.py \\
        [--base out/corpus_2b] [--extension out/corpus_2b/versions/v082_bc2b/daily_extension] \\
        [--out-dir out/corpus_2b/versions/v082_bc2b/daily_merged]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(REPO / "out" / "corpus_2b"))
    ap.add_argument("--extension", default=str(REPO / "out" / "corpus_2b" / "versions" / "v082_bc2b" / "daily_extension"))
    ap.add_argument("--out-dir", default=str(REPO / "out" / "corpus_2b" / "versions" / "v082_bc2b" / "daily_merged"))
    args = ap.parse_args()

    base = Path(args.base)
    ext = Path(args.extension)
    out_dir = Path(args.out_dir)
    out_shards = out_dir / "obs_shards"
    out_split_dir = out_dir / "split"
    out_shards.mkdir(parents=True, exist_ok=True)
    out_split_dir.mkdir(parents=True, exist_ok=True)

    base_decisions = base / "decisions.jsonl"
    ext_decisions = ext / "decisions_extra.jsonl"
    base_split = json.loads((base / "split" / "split.json").read_text())
    test_ids = set(int(x) for x in base_split["test_episode_ids"])
    train_ids = set(int(x) for x in base_split["train_episode_ids"])

    n_base = 0
    n_ext = 0
    new_train_ids: set[int] = set()
    out_decisions_path = out_dir / "decisions.jsonl"
    with open(out_decisions_path, "w") as out_f:
        with open(base_decisions) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out_f.write(line + "\n")
                n_base += 1
        if ext_decisions.is_file():
            with open(ext_decisions) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    eid = int(d["episode_id"])
                    # Defensive re-check (builder already dedups, but merge must not trust that
                    # blindly -- a collision here would silently corrupt the frozen test split).
                    if eid in test_ids:
                        sys.exit(f"FATAL: extension episode {eid} collides with the FROZEN test "
                                  f"split -- refusing to merge (builder dedup should have caught "
                                  f"this, this is a hard invariant, not a soft warning)")
                    out_f.write(line + "\n")
                    n_ext += 1
                    new_train_ids.add(eid)

    # obs_shards: symlink every per-episode file from both trees (episode_ids are disjoint by
    # construction -- base corpus episodes vs NEW daily-zip episodes never overlap).
    n_links = 0
    for src_dir in (base / "obs_shards", ext / "obs_shards"):
        if not src_dir.is_dir():
            continue
        for p in src_dir.glob("*.jsonl"):
            dest = out_shards / p.name
            if dest.exists() or dest.is_symlink():
                sys.exit(f"FATAL: obs_shard filename collision at merge: {p.name} "
                          f"(base vs extension episode_id overlap -- should be impossible)")
            dest.symlink_to(p.resolve())
            n_links += 1

    merged_split = {
        "test_episode_ids": sorted(test_ids),  # UNCHANGED -- frozen
        "train_episode_ids": sorted(train_ids | new_train_ids),
        "rule": base_split.get("rule", "") + " + v082[A] daily-zip extension appended to TRAIN only "
                "(test_episode_ids byte-identical to out/corpus_2b/split/split.json)",
    }
    (out_split_dir / "split.json").write_text(json.dumps(merged_split, indent=2))

    report = {
        "n_base_decisions": n_base,
        "n_extension_decisions": n_ext,
        "n_merged_decisions": n_base + n_ext,
        "n_base_train_episodes": len(train_ids),
        "n_new_train_episodes_from_extension": len(new_train_ids),
        "n_merged_train_episodes": len(train_ids | new_train_ids),
        "n_test_episodes_unchanged": len(test_ids),
        "n_obs_shard_symlinks": n_links,
        "out_dir": str(out_dir),
    }
    (out_dir / "merge_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
