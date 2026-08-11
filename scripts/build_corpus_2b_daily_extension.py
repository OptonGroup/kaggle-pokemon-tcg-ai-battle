#!/usr/bin/env python3
"""build_corpus_2b_daily_extension.py — [v082 A] mine EXTRA expert decision-level (obs, action)
pairs from the bulk Kaggle daily episode-dump zips (data/kaggle_daily_zips/*.zip) to extend the
2B BC train corpus (out/corpus_2b/decisions.jsonl / obs_shards) WITHOUT touching the frozen test
split.

Filter (per task instructions): a (episode, side) qualifies as a new expert source iff ALL of:
  1. team is NOT us (name != "Maxim Simakov" — see build_expert_corpus_2b.py analysis scripts
     for the exact string, project convention).
  2. team's CURRENT public-leaderboard score >= --min-score (freshest local *leaderboard*.csv
     snapshot under out/**, docs/gold/, scratchpad/gold/ — same source scan_daily_zip_field.py
     uses, reused here via load_leaderboard_by_name(); "current" not "at play time", see that
     module's rating caveat docstring — best available proxy, absence = excluded, not assumed-ok).
  3. the side's ACTUAL submitted 60-card deck (steps[1][idx].action via
     src/replays/parse.py::true_decks_from_data — ground truth, NOT the stale/mislabelled
     data/deck_scan/top10_decks.json, see MEMORY project_crustle_label trap / task instructions)
     is within --min-overlap/60 multiset overlap of our reference list (submissions/v080_bc2b/
     deck.csv, the Luca-toolscrapper list the existing 2B corpus is built on).
  4. episode_id is NOT already in the existing corpus (out/corpus_2b/decisions.jsonl) and NOT in
     the frozen test split (out/corpus_2b/split/split.json::test_episode_ids) — dedup is a hard
     requirement, not a nice-to-have: a test episode leaking into train would invalidate every
     comparison this task's B/C/A experiments rely on.

Decision-pair extraction reuses src/replays/parse.py::parse_replay EXACTLY as
build_expert_corpus_2b.py does (canonical off-by-one splice, see that module's docstring) — the
only difference is the source file lives inside a zip, so the raw entry bytes are spilled to a
NamedTemporaryFile first (parse_replay takes a path, not a dict) and removed right after.

step_idx is NOT computed here (unlike build_expert_corpus_2b.py's ЛОВУШКА #2 two-pointer scan)
— verified against featurize_corpus_2b.py::process_episode: decision_uid keys off obs_ref (the
row's own position in the obs_shard file), not step_idx; step_idx only ever lands in META_COLUMNS
for audit/display, never in FEATURE_ORDER. Skipping it avoids an O(steps) scan per qualifying
episode with zero effect on any downstream training or metric. Rows get step_idx=null.

\\uXXXX-escape trap (MEMORY kaggle_episode_dataset_gotchas): only matters for raw-text/grep
pre-filtering (scan_daily_dataset_for_experts.py's ASCII-grep fast path). This script always
goes through json.loads on the full entry, which decodes \\uXXXX escapes automatically — team
names are compared post-decode by construction, no special-casing needed.

Output (mirrors build_expert_corpus_2b.py's schema, plus a "source" tag):
  <out-dir>/decisions_extra.jsonl   — one row per decision (same fields as the base corpus'
                                       decisions.jsonl + "source": "daily_zip_A")
  <out-dir>/obs_shards/<episode_id>.jsonl
  <out-dir>/stats.json              — scan funnel (n episodes seen/skipped by reason, n
                                       qualifying sides, n decisions, per-team breakdown)

Usage:
    .venv/bin/python scripts/build_corpus_2b_daily_extension.py \\
        [--zips data/kaggle_daily_zips/*.zip] [--min-score 950] [--min-overlap 58] \\
        [--out-dir out/corpus_2b/versions/v082_bc2b/daily_extension] [--limit-per-zip N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from replays.parse import parse_replay, true_decks_from_data, _context_name  # noqa: E402
from classify_opponent_archetype import classify  # noqa: E402
from cluster_archetypes import load_card_meta  # noqa: E402
from scan_daily_zip_field import load_leaderboard_by_name  # noqa: E402 (reuse, don't reinvent)

OUR_TEAM = "Maxim Simakov"
CRUSTLE_ID = 345
DEFAULT_ZIP_DIR = REPO / "data" / "kaggle_daily_zips"
DEFAULT_REF_DECK = REPO / "submissions" / "v080_bc2b" / "deck.csv"
DEFAULT_EXISTING_DECISIONS = REPO / "out" / "corpus_2b" / "decisions.jsonl"
DEFAULT_SPLIT = REPO / "out" / "corpus_2b" / "split" / "split.json"
DEFAULT_OUT_DIR = REPO / "out" / "corpus_2b" / "versions" / "v082_bc2b" / "daily_extension"

# Known variant deck_keys (see build_expert_corpus_2b.py::_DECK_KEYS) — checked by EXACT sorted
# multiset first (cheap, precise); falls back to "other:<md5>" for anything that passes the
# --min-overlap gate but isn't byte-identical to a known list (e.g. a 1-2 card swap).
_DECK_KEYS = {
    "7 7 7 7 7 7 7 7 7 7 104 104 112 112 112 112 646 646 646 646 647 647 647 648 648 648 "
    "860 860 1079 1079 1079 1080 1086 1086 1086 1086 1097 1097 1097 1122 1137 1152 1152 1152 "
    "1152 1182 1182 1219 1219 1219 1219 1227 1227 1227 1227 1231 1259 1259 1259 1259": "luca_toolscrapper",
    "7 7 7 7 7 7 7 7 7 7 104 104 112 112 112 112 646 646 646 646 647 647 647 648 648 648 "
    "860 860 1079 1079 1079 1080 1086 1086 1086 1086 1097 1097 1097 1152 1152 1152 1152 1161 "
    "1161 1182 1182 1219 1219 1219 1219 1227 1227 1227 1227 1231 1259 1259 1259 1259": "peel_handfan",
}


def _deck_key(deck_ids: list[int]) -> str:
    s = " ".join(str(x) for x in sorted(deck_ids))
    if s in _DECK_KEYS:
        return _DECK_KEYS[s]
    import hashlib
    return "other:" + hashlib.md5(s.encode()).hexdigest()[:8]


def multiset_overlap(a: list[int], b: list[int]) -> int:
    return sum((Counter(a) & Counter(b)).values())


def load_ref_deck(path: Path) -> list[int]:
    ids = [int(x) for x in path.read_text().split()]
    assert len(ids) == 60, f"{path}: expected 60 card ids, got {len(ids)}"
    return ids


def existing_corpus_episode_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.add(int(json.loads(line)["episode_id"]))
    return out


def test_split_episode_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    d = json.loads(path.read_text())
    return set(int(x) for x in d.get("test_episode_ids", []))


def discover_zips(zips_arg: list[str] | None) -> list[Path]:
    if not zips_arg:
        return sorted(DEFAULT_ZIP_DIR.glob("*.zip"))
    out = []
    for z in zips_arg:
        p = Path(z)
        if p.is_dir():
            out.extend(sorted(p.glob("*.zip")))
        else:
            out.append(p)
    return out


def process_episode(eid: int, raw: bytes, ref_deck: list[int], min_overlap: int,
                     leaderboard: dict[str, float], min_score: float, card_names: dict,
                     out_shard_dir: Path, funnel: Counter, per_team: Counter) -> list[dict]:
    """Returns the list of decision-row dicts contributed by this episode (possibly from BOTH
    sides if both qualify — mirrors build_expert_corpus_2b.py ЛОВУШКА #3), or [] if it doesn't
    qualify / fails to parse. Writes its own obs_shard file iff it contributes >=1 row."""
    try:
        data = json.loads(raw)
    except Exception:
        funnel["json_parse_error"] += 1
        return []

    info = data.get("info", {}) or {}
    agents = info.get("Agents", []) or []
    if len(agents) != 2:
        funnel["not_2_agents"] += 1
        return []
    names = [a.get("Name") or "" for a in agents]

    try:
        decks = true_decks_from_data(data)
    except Exception:
        funnel["deck_extract_error"] += 1
        return []
    if 0 not in decks or 1 not in decks:
        funnel["deck_not_extractable"] += 1
        return []

    qualifying_sides = []
    for i in range(2):
        nm = names[i]
        if not nm or nm == OUR_TEAM:
            continue
        score = leaderboard.get(nm)
        if score is None or score < min_score:
            continue
        overlap = multiset_overlap(decks[i], ref_deck)
        if overlap < min_overlap:
            continue
        qualifying_sides.append((i, nm, score, overlap))

    if not qualifying_sides:
        funnel["no_qualifying_side"] += 1
        return []

    # Parse via the canonical splicer — spill raw bytes to a temp file (parse_replay wants a
    # path, see module docstring), delete right after regardless of outcome.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tf.write(raw)
        tmp_path = tf.name
    try:
        pairs = parse_replay(tmp_path)
    except Exception:
        funnel["parse_replay_error"] += 1
        return []
    finally:
        os.unlink(tmp_path)

    by_agent: dict[int, list[dict]] = defaultdict(list)
    for p in pairs:
        by_agent[p["player_index"]].append(p)

    rows: list[dict] = []
    shard_lines: list[dict] = []
    for side, nm, score, overlap in qualifying_sides:
        agent_pairs = by_agent.get(side, [])
        if not agent_pairs:
            funnel["qualifying_side_no_decisions"] += 1
            continue
        opp_idx = 1 - side
        opp_deck_ids = decks.get(opp_idx, [])
        opp_names = [card_names.get(cid, "") for cid in opp_deck_ids]
        opp_arche = classify(opp_deck_ids, opp_names) or "UNKNOWN"
        opp_has_crustle = CRUSTLE_ID in opp_deck_ids
        deck_key = _deck_key(decks[side])
        won = agent_pairs[0]["won"]
        per_team[nm] += 1
        for pair in agent_pairs:
            obs = pair["obs_dict"]
            sel = obs.get("select") or {}
            cur = obs.get("current")
            turn_idx = cur.get("turn") if isinstance(cur, dict) else None
            obs_ref = len(shard_lines)
            shard_lines.append(obs)
            rows.append({
                "episode_id": eid,
                "expert": nm,
                "agent_index": side,
                "turn_idx": turn_idx,
                "step_idx": None,  # not computed, see module docstring
                "sel_context": pair["select_context"],
                "sel_context_name": _context_name(pair["select_context"]),
                "n_options": pair["n_options"],
                "action": pair["chosen_option_indices"],
                "min_count": sel.get("minCount"),
                "max_count": sel.get("maxCount"),
                "opp_archetype": opp_arche,
                "opp_has_crustle": opp_has_crustle,
                "subj_archetype": "mirror_grimmsnarl",  # deck matched ref within min_overlap
                "deck_key": deck_key,
                "game_won": won,
                "obs_ref": obs_ref,
                "source": "daily_zip_A",
                "leaderboard_score": score,
                "deck_overlap": overlap,
            })
        funnel["qualifying_side_ok"] += 1

    if shard_lines:
        shard_path = out_shard_dir / f"{eid}.jsonl"
        with open(shard_path, "w") as f:
            for obs in shard_lines:
                f.write(json.dumps(obs) + "\n")

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", nargs="+", default=None,
                     help="zip files or dirs; default = all *.zip in data/kaggle_daily_zips/")
    ap.add_argument("--min-score", type=float, default=950.0,
                     help="leaderboard score threshold (default ~top-200 on the freshest local "
                          "snapshot, see load_leaderboard_by_name)")
    ap.add_argument("--ref-deck", default=str(DEFAULT_REF_DECK))
    ap.add_argument("--min-overlap", type=int, default=58, help="<=2-card deviation of 60 by default")
    ap.add_argument("--existing-decisions", default=str(DEFAULT_EXISTING_DECISIONS))
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--limit-per-zip", type=int, default=0, help="debug only, 0 = no limit")
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_shard_dir = out_dir / "obs_shards"
    out_shard_dir.mkdir(parents=True, exist_ok=True)

    ref_deck = load_ref_deck(Path(args.ref_deck))
    existing_ids = existing_corpus_episode_ids(Path(args.existing_decisions))
    test_ids = test_split_episode_ids(Path(args.split))
    leaderboard, lb_path = load_leaderboard_by_name()
    card_meta = load_card_meta()
    card_names = card_meta.name  # {card_id: name}

    zip_paths = discover_zips(args.zips)
    if not zip_paths:
        sys.exit(f"no zip files found (args.zips={args.zips})")

    print(f"[build_corpus_2b_daily_extension] {len(zip_paths)} zip(s): {[p.name for p in zip_paths]}",
          file=sys.stderr)
    print(f"[build_corpus_2b_daily_extension] ref_deck={args.ref_deck} min_overlap={args.min_overlap} "
          f"min_score={args.min_score} leaderboard={lb_path} ({len(leaderboard)} teams) "
          f"existing_corpus_episodes={len(existing_ids)} test_split_episodes={len(test_ids)}",
          file=sys.stderr)

    funnel: Counter = Counter()
    per_team: Counter = Counter()
    seen_this_run: set[int] = set()
    decisions_path = out_dir / "decisions_extra.jsonl"
    t0 = time.time()
    n_decisions = 0

    with open(decisions_path, "w") as out_f:
        for zp in zip_paths:
            z = zipfile.ZipFile(zp)
            names = [n for n in z.namelist() if n != "manifest.csv"]
            if args.limit_per_zip:
                names = names[: args.limit_per_zip]
            print(f"[build_corpus_2b_daily_extension] {zp.name}: {len(names)} entries", file=sys.stderr)
            for i, entry in enumerate(names, 1):
                try:
                    eid = int(Path(entry).stem)
                except ValueError:
                    funnel["bad_filename"] += 1
                    continue
                funnel["n_entries_seen"] += 1
                if eid in existing_ids:
                    funnel["skip_already_in_corpus"] += 1
                    continue
                if eid in test_ids:
                    funnel["skip_in_test_split"] += 1
                    continue
                if eid in seen_this_run:
                    funnel["skip_dup_within_run"] += 1
                    continue
                raw = z.read(entry)
                rows = process_episode(eid, raw, ref_deck, args.min_overlap, leaderboard,
                                        args.min_score, card_names, out_shard_dir, funnel, per_team)
                if rows:
                    seen_this_run.add(eid)
                    for r in rows:
                        out_f.write(json.dumps(r) + "\n")
                        n_decisions += 1
                if i % args.progress_every == 0:
                    elapsed = time.time() - t0
                    print(f"[build_corpus_2b_daily_extension] ...{zp.name} {i}/{len(names)} "
                          f"({elapsed:.0f}s elapsed, {n_decisions} decisions so far)", file=sys.stderr)

    elapsed = time.time() - t0
    stats = {
        "params": vars(args),
        "leaderboard_path": str(lb_path) if lb_path else None,
        "n_leaderboard_teams": len(leaderboard),
        "n_existing_corpus_episodes_excluded_pool": len(existing_ids),
        "n_test_split_episodes_excluded_pool": len(test_ids),
        "funnel": dict(funnel.most_common()),
        "n_qualifying_episodes": len(seen_this_run),
        "n_decisions": n_decisions,
        "per_team_qualifying_sides": dict(per_team.most_common()),
        "elapsed_sec": round(elapsed, 1),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"[build_corpus_2b_daily_extension] done in {elapsed:.1f}s: "
          f"{len(seen_this_run)} qualifying episodes, {n_decisions} decisions -> {decisions_path}",
          file=sys.stderr)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
