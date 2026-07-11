#!/usr/bin/env python3
"""
audit_replay_offbyone.py — H1-1: audit whether the reported engine replay
off-by-one bug (forum thread 717141, 2026-07-09) affects OUR IL training
pipeline (src/replays/parse.py).

Background
----------
The bug report claims the game engine's internal visualize/battle export
(Export.cpp / Api.h) writes each decision's `entry[i].selected` field one
slot LATE: it actually belongs to the PREVIOUS decision (i-1), entry[0]
.selected is always null, and the real choice for entry[0] lives in
entry[1].selected.

Our parse.py, however, does NOT read that internal "visualize" export at
all. It reads the TOP-LEVEL kaggle-environments `steps` array and pairs

    obs    = steps[i][player]["observation"]      (has .select)
    action = steps[i + 1][player]["action"]

i.e. it already accounts for the standard RL convention that the action
recorded at frame i is the response to the observation shown at frame
i - 1. This script empirically checks, on real downloaded replay JSONs:

  1. Whether THIS pairing (source 1, what parse.py actually uses) shows
     any sign of an additional off-by-one (own vs. previous validity).
  2. Whether the internal `visualize` array (source 2, only present as
     steps[0][0]["visualize"], a flat merged decision log with
     `select` + `selected` per entry) reproduces the reported bug.
  3. Whether source 1 and source 2 agree step-by-step (and at what
     offset), when both are present in the same replay file.

Pure stdlib, no libcg, macOS-safe, read-only w.r.t. the repo except for
writing nothing (this script only reads replay JSONs).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = REPO_ROOT / "data" / "episodes" / "all_teams"

# Same skip set as src/replays/parse.py
_SKIP_CONTEXTS = {41, 42}  # IS_FIRST, MULLIGAN


# --------------------------------------------------------------------------
# Validity helpers
# --------------------------------------------------------------------------

def _select_info(sel: dict | None):
    """Return (context, minCount, maxCount, n_options) or None."""
    if not sel:
        return None
    opts = sel.get("option", [])
    return (
        sel.get("context", -1),
        sel.get("minCount", 0),
        sel.get("maxCount", len(opts)),
        len(opts),
    )


def _is_valid(action, sel_info) -> bool | None:
    """
    Check whether `action` (list[int]) is a valid response to a select
    described by sel_info = (context, minCount, maxCount, n_options).
    Returns None if sel_info is None (can't judge) or action is empty.
    """
    if sel_info is None or not action:
        return None
    _, min_c, max_c, n_opts = sel_info
    if n_opts == 0:
        return None
    if not (min_c <= len(action) <= max_c):
        return False
    if not all(isinstance(x, int) and 0 <= x < n_opts for x in action):
        return False
    return True


def _classify(own_valid, prev_valid) -> str:
    """a/b/c/neither classification per the audit spec."""
    if own_valid is None or prev_valid is None:
        return "skip"
    if (not own_valid) and prev_valid:
        return "a_shift_evidence"       # invalid own, valid prev -> shifted
    if own_valid and (not prev_valid):
        return "b_noshift_evidence"     # valid own, invalid prev -> not shifted
    if own_valid and prev_valid:
        return "c_both_valid"           # uninformative
    return "d_neither_valid"            # uninformative (both broken / unrelated)


# --------------------------------------------------------------------------
# Source 1: parse.py's actual pairing (top-level `steps` array)
# --------------------------------------------------------------------------

def analyze_source1(data: dict) -> dict:
    """
    Replicates parse.py's (obs=steps[i], action=steps[i+1]) pairing per
    player, WITHOUT parse.py's own "skip if invalid" filter (that filter
    would hide the very bug we're testing for). For each decision (after
    the first per player) classify a/b/c/d by comparing the action against
    its "own" paired select vs. the PREVIOUS decision's select (for that
    same player, in temporal order).
    """
    steps = data.get("steps", [])
    n_steps = len(steps)
    n_players = len(steps[0]) if steps else 0

    counts = Counter()
    informative_examples = []
    # semantic markers
    marker_len1 = Counter()       # minCount==maxCount==1 steps: a/b/c/d
    marker_bigjump = Counter()    # |n_opts_own - n_opts_prev| big: a/b/c/d

    for player in range(n_players):
        # Build ordered decision list for this player: (step_idx, sel_info, action)
        decisions = []
        for i in range(n_steps - 1):
            obs = steps[i][player].get("observation") or {}
            sel = obs.get("select")
            if sel is None:
                continue
            ctx = sel.get("context", -1)
            if ctx in _SKIP_CONTEXTS:
                continue
            sel_info = _select_info(sel)
            if sel_info is None or sel_info[3] == 0:
                continue
            next_step = steps[i + 1]
            if player >= len(next_step):
                continue
            action = next_step[player].get("action") or []
            if not action:
                continue
            if len(action) == 60 and all(isinstance(x, int) for x in action):
                continue  # deck submission, not a real select response
            decisions.append((i, sel_info, list(action)))

        for k in range(1, len(decisions)):
            _, own_info, action = decisions[k]
            _, prev_info, _ = decisions[k - 1]
            own_valid = _is_valid(action, own_info)
            prev_valid = _is_valid(action, prev_info)
            cls = _classify(own_valid, prev_valid)
            counts[cls] += 1

            if own_info[1] == own_info[2] == 1:
                marker_len1[cls] += 1
            if abs(own_info[3] - prev_info[3]) >= 3:
                marker_bigjump[cls] += 1
                if cls == "a_shift_evidence" and len(informative_examples) < 5:
                    informative_examples.append(
                        dict(player=player, step_idx=decisions[k][0],
                             action=action, own=own_info, prev=prev_info)
                    )

    return dict(counts=counts, marker_len1=marker_len1,
                marker_bigjump=marker_bigjump, examples=informative_examples)


# --------------------------------------------------------------------------
# Source 2: internal `visualize` export (steps[0][0]["visualize"])
# --------------------------------------------------------------------------

def _vis_actor(entry: dict, n_players: int) -> int | None:
    """Determine which player actually acted in this visualize entry, by
    finding which sublist of entry['action'] is non-empty."""
    a = entry.get("action")
    if not isinstance(a, list):
        return None
    for p, sub in enumerate(a):
        if p >= n_players:
            break
        if isinstance(sub, list) and sub:
            return p
    return None


def analyze_source2(data: dict) -> dict | None:
    steps = data.get("steps", [])
    if not steps or "visualize" not in steps[0][0]:
        return None
    vis = steps[0][0]["visualize"]
    n_players = len(steps[0])

    counts = Counter()
    marker_len1 = Counter()
    marker_bigjump = Counter()
    entry0_selected_is_none = vis[0].get("selected") is None if vis else None

    for e in range(1, len(vis)):
        sel = vis[e].get("select")
        selected = vis[e].get("selected")
        prev_sel = vis[e - 1].get("select")
        own_info = _select_info(sel)
        prev_info = _select_info(prev_sel)
        if selected is None or own_info is None or prev_info is None:
            continue
        own_valid = _is_valid(selected, own_info)
        prev_valid = _is_valid(selected, prev_info)
        cls = _classify(own_valid, prev_valid)
        counts[cls] += 1
        if own_info[1] == own_info[2] == 1:
            marker_len1[cls] += 1
        if abs(own_info[3] - prev_info[3]) >= 3:
            marker_bigjump[cls] += 1

    return dict(counts=counts, marker_len1=marker_len1,
                marker_bigjump=marker_bigjump,
                entry0_selected_is_none=entry0_selected_is_none,
                n_entries=len(vis), n_players=n_players)


# --------------------------------------------------------------------------
# Cross-source comparison
# --------------------------------------------------------------------------

def cross_compare(data: dict) -> dict | None:
    """
    Build seq1 (chronological non-empty actions read straight off the
    top-level `steps` array, in (step_idx, player) order) and seq2
    (chronological `selected` values off the internal `visualize` array,
    skipping entry 0). Compare elementwise equality at offsets -1/0/+1.
    """
    steps = data.get("steps", [])
    if not steps or "visualize" not in steps[0][0]:
        return None
    vis = steps[0][0]["visualize"]
    n_steps = len(steps)
    n_players = len(steps[0])

    seq1 = []
    for i in range(1, n_steps):
        for player in range(n_players):
            a = steps[i][player].get("action") or []
            if not a:
                continue
            if len(a) == 60 and all(isinstance(x, int) for x in a):
                continue
            seq1.append(tuple(a))

    seq2 = []
    for e in range(1, len(vis)):
        selected = vis[e].get("selected")
        if not selected:
            continue
        seq2.append(tuple(selected))

    def match_frac(offset: int) -> tuple[int, int]:
        matches = 0
        total = 0
        for k in range(len(seq2)):
            j = k + offset
            if 0 <= j < len(seq1):
                total += 1
                if seq1[j] == seq2[k]:
                    matches += 1
        return matches, total

    result = {}
    for off in (-1, 0, 1):
        m, t = match_frac(off)
        result[off] = (m, t, (m / t if t else None))

    return dict(len_seq1=len(seq1), len_seq2=len(seq2), offsets=result)


# --------------------------------------------------------------------------
# Per-file verdict
# --------------------------------------------------------------------------

def verdict_from_counts(counts: Counter) -> str:
    a = counts.get("a_shift_evidence", 0)
    b = counts.get("b_noshift_evidence", 0)
    informative = a + b
    if informative < 5:
        return "AMBIGUOUS"
    frac_a = a / informative
    if frac_a >= 0.8:
        return "SHIFTED"
    if frac_a <= 0.2:
        return "NOT_SHIFTED"
    return "MIXED"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Audit replay JSONs for the reported off-by-one export bug")
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="Directory of *-replay.json files")
    ap.add_argument("-n", "--num-files", type=int, default=250, help="Number of replays to sample")
    ap.add_argument("--seed", type=int, default=1234, help="Random sample seed")
    ap.add_argument("--out", default=str(REPO_ROOT / "scratchpad" / "audit_offbyone_report.md"))
    args = ap.parse_args()

    d = Path(args.dir)
    all_files = sorted(d.glob("*-replay.json"))
    if not all_files:
        raise SystemExit(f"No *-replay.json files found under {d}")

    rng = random.Random(args.seed)
    sample = all_files if len(all_files) <= args.num_files else rng.sample(all_files, args.num_files)
    sample = sorted(sample)

    print(f"Sampling {len(sample)} / {len(all_files)} replay files from {d} (seed={args.seed})")

    s1_total = Counter()
    s1_marker_len1 = Counter()
    s1_marker_bigjump = Counter()
    s1_examples = []
    s1_file_verdicts = Counter()

    s2_total = Counter()
    s2_marker_len1 = Counter()
    s2_marker_bigjump = Counter()
    s2_file_verdicts = Counter()
    s2_entry0_none_count = 0
    s2_entry0_checked = 0
    files_with_vis = 0

    cross_offset_totals = {-1: [0, 0], 0: [0, 0], 1: [0, 0]}
    files_with_cross = 0

    parse_errors = 0

    for idx, path in enumerate(sample):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            parse_errors += 1
            print(f"  WARN: failed to load {path.name}: {e}")
            continue

        r1 = analyze_source1(data)
        s1_total.update(r1["counts"])
        s1_marker_len1.update(r1["marker_len1"])
        s1_marker_bigjump.update(r1["marker_bigjump"])
        if len(s1_examples) < 8:
            s1_examples.extend(r1["examples"])
        s1_file_verdicts[verdict_from_counts(r1["counts"])] += 1

        r2 = analyze_source2(data)
        if r2 is not None:
            files_with_vis += 1
            s2_total.update(r2["counts"])
            s2_marker_len1.update(r2["marker_len1"])
            s2_marker_bigjump.update(r2["marker_bigjump"])
            s2_file_verdicts[verdict_from_counts(r2["counts"])] += 1
            if r2["entry0_selected_is_none"] is not None:
                s2_entry0_checked += 1
                if r2["entry0_selected_is_none"]:
                    s2_entry0_none_count += 1

        rc = cross_compare(data)
        if rc is not None and rc["len_seq2"] > 0:
            files_with_cross += 1
            for off, (m, t, _frac) in rc["offsets"].items():
                cross_offset_totals[off][0] += m
                cross_offset_totals[off][1] += t

        if (idx + 1) % 50 == 0:
            print(f"  ... {idx + 1}/{len(sample)} processed")

    # ---------------- Global verdicts ----------------
    a1, b1 = s1_total.get("a_shift_evidence", 0), s1_total.get("b_noshift_evidence", 0)
    informative1 = a1 + b1
    frac_a1 = a1 / informative1 if informative1 else None

    a2, b2 = s2_total.get("a_shift_evidence", 0), s2_total.get("b_noshift_evidence", 0)
    informative2 = a2 + b2
    frac_a2 = a2 / informative2 if informative2 else None

    def global_verdict(frac_a):
        if frac_a is None:
            return "AMBIGUOUS (insufficient informative steps)"
        if frac_a >= 0.8:
            return "SHIFTED"
        if frac_a <= 0.2:
            return "NOT_SHIFTED"
        return "MIXED"

    verdict1 = global_verdict(frac_a1)
    verdict2 = global_verdict(frac_a2)

    # ---------------- Report ----------------
    lines = []
    lines.append("# Audit: replay off-by-one export bug vs. our IL pipeline")
    lines.append("")
    lines.append(f"Sampled **{len(sample)}** / {len(all_files)} replay files from `{d}` (seed={args.seed}).")
    lines.append(f"Parse errors: {parse_errors}. Files containing an internal `visualize` export: {files_with_vis}.")
    lines.append("")

    lines.append("## VERDICT")
    lines.append("")
    lines.append(f"- **Our pipeline (src/replays/parse.py, source 1 = top-level `steps[].action`/`.observation`): {verdict1}**")
    lines.append(f"- Internal engine export (`visualize[].selected`, source 2, forum-reported bug target): {verdict2}")
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("For each decision (after the first, per player/sequence), the *assigned* action/selected")
    lines.append("value is checked for validity (index in range, length in [minCount, maxCount]) against:")
    lines.append("")
    lines.append("- **own** select (the one it's paired with in the representation under test)")
    lines.append("- **prev** select (the immediately preceding decision in the same representation)")
    lines.append("")
    lines.append("Classification per decision:")
    lines.append("")
    lines.append("- `a_shift_evidence`: invalid vs own, valid vs prev -> evidence FOR an off-by-one shift")
    lines.append("- `b_noshift_evidence`: valid vs own, invalid vs prev -> evidence AGAINST a shift (current pairing correct)")
    lines.append("- `c_both_valid`: valid against both -> uninformative")
    lines.append("- `d_neither_valid`: invalid against both -> uninformative / unrelated")
    lines.append("")
    lines.append("Global verdict = SHIFTED if a/(a+b) >= 0.8, NOT_SHIFTED if <= 0.2, else MIXED,")
    lines.append("computed over the whole sample (not per file) as the primary conclusion; per-file")
    lines.append("verdicts (same thresholds) are also tallied below for reference.")
    lines.append("")

    lines.append("## Source 1 — parse.py's actual pairing (steps[i].observation -> steps[i+1].action)")
    lines.append("")
    lines.append("This is EXACTLY the representation src/replays/parse.py uses to build (state, action)")
    lines.append("training pairs (see parse_replay(), lines ~176-222): `obs = steps[i][player_idx][\"observation\"]`,")
    lines.append("`action = steps[i + 1][player_idx].get(\"action\", [])`. This test does NOT apply parse.py's")
    lines.append("own final index-range filter (which would silently discard the very evidence we're after) —")
    lines.append("it independently re-checks validity (index range AND minCount/maxCount) against own vs prev.")
    lines.append("")
    lines.append("| class | count |")
    lines.append("|---|---|")
    for k in ["a_shift_evidence", "b_noshift_evidence", "c_both_valid", "d_neither_valid", "skip"]:
        lines.append(f"| {k} | {s1_total.get(k, 0)} |")
    lines.append("")
    lines.append(f"Informative steps (a+b): **{informative1}**; fraction shift-evidence a/(a+b) = "
                  f"{f'{frac_a1:.3f}' if frac_a1 is not None else 'n/a'}  -> **{verdict1}**")
    lines.append("")
    lines.append("Semantic marker — minCount==maxCount==1 steps (subset of the above):")
    lines.append("")
    lines.append("| class | count |")
    lines.append("|---|---|")
    for k in ["a_shift_evidence", "b_noshift_evidence", "c_both_valid", "d_neither_valid"]:
        lines.append(f"| {k} | {s1_marker_len1.get(k, 0)} |")
    lines.append("")
    lines.append("Semantic marker — steps where |n_options(own) - n_options(prev)| >= 3 (most informative,")
    lines.append("large option-count mismatch between own and previous decision):")
    lines.append("")
    lines.append("| class | count |")
    lines.append("|---|---|")
    for k in ["a_shift_evidence", "b_noshift_evidence", "c_both_valid", "d_neither_valid"]:
        lines.append(f"| {k} | {s1_marker_bigjump.get(k, 0)} |")
    lines.append("")
    lines.append("Per-file verdict tally (source 1):")
    lines.append("")
    lines.append("| verdict | files |")
    lines.append("|---|---|")
    for k in ["SHIFTED", "NOT_SHIFTED", "MIXED", "AMBIGUOUS"]:
        lines.append(f"| {k} | {s1_file_verdicts.get(k, 0)} |")
    lines.append("")
    if s1_examples:
        lines.append("Example `a_shift_evidence` cases found in source 1 (should be ~0 if pipeline is clean):")
        lines.append("")
        for ex in s1_examples[:8]:
            lines.append(f"- player={ex['player']} step_idx={ex['step_idx']} action={ex['action']} "
                          f"own(ctx,minC,maxC,nOpt)={ex['own']} prev={ex['prev']}")
        lines.append("")

    lines.append("## Source 2 — internal engine export (`steps[0][0][\"visualize\"]`, entry.select / entry.selected)")
    lines.append("")
    lines.append("This is the internal battle/visualize export the forum bug report (topic 717141) is about.")
    lines.append("It is a SEPARATE, flat, globally-ordered decision log distinct from the top-level `steps`")
    lines.append("array parse.py reads; it is NOT touched anywhere in src/replays/parse.py.")
    lines.append("")
    lines.append(f"`visualize[0][\"selected\"]` is None in {s2_entry0_none_count}/{s2_entry0_checked} files "
                  "checked (bug report predicts: always None).")
    lines.append("")
    lines.append("| class | count |")
    lines.append("|---|---|")
    for k in ["a_shift_evidence", "b_noshift_evidence", "c_both_valid", "d_neither_valid", "skip"]:
        lines.append(f"| {k} | {s2_total.get(k, 0)} |")
    lines.append("")
    lines.append(f"Informative steps (a+b): **{informative2}**; fraction shift-evidence a/(a+b) = "
                  f"{f'{frac_a2:.3f}' if frac_a2 is not None else 'n/a'}  -> **{verdict2}**")
    lines.append("")
    lines.append("Semantic marker — |n_options(own) - n_options(prev)| >= 3:")
    lines.append("")
    lines.append("| class | count |")
    lines.append("|---|---|")
    for k in ["a_shift_evidence", "b_noshift_evidence", "c_both_valid", "d_neither_valid"]:
        lines.append(f"| {k} | {s2_marker_bigjump.get(k, 0)} |")
    lines.append("")
    lines.append("Per-file verdict tally (source 2):")
    lines.append("")
    lines.append("| verdict | files |")
    lines.append("|---|---|")
    for k in ["SHIFTED", "NOT_SHIFTED", "MIXED", "AMBIGUOUS"]:
        lines.append(f"| {k} | {s2_file_verdicts.get(k, 0)} |")
    lines.append("")

    lines.append("## Cross-source comparison (per file, when both sources present)")
    lines.append("")
    lines.append("seq1 = chronological non-empty actions read directly off the top-level `steps` array")
    lines.append("(step_idx ascending, player ascending, excluding deck-submission 60-length actions).")
    lines.append("seq2 = chronological `selected` values off `visualize[1:]`. Elementwise equality checked")
    lines.append("at offsets -1/0/+1 (seq1[k+offset] == seq2[k]).")
    lines.append("")
    lines.append(f"Files with both sources and non-empty seq2: {files_with_cross}")
    lines.append("")
    lines.append("| offset | matches | total | match fraction |")
    lines.append("|---|---|---|---|")
    for off in (-1, 0, 1):
        m, t = cross_offset_totals[off]
        frac = f"{m / t:.3f}" if t else "n/a"
        lines.append(f"| {off:+d} | {m} | {t} | {frac} |")
    lines.append("")
    lines.append("Interpretation: if offset 0 has the highest match fraction, seq1 (our harness-level source)")
    lines.append("already agrees with seq2 (`selected`) index-for-index -- i.e. whatever labeling convention")
    lines.append("`selected` uses, our `steps[].action` reproduces the SAME sequence at the SAME position,")
    lines.append("so any bug purely internal to how `selected` is exported does not, by itself, imply our")
    lines.append("`action` sequence is shifted. If offset -1 or +1 dominates instead, the two sources are")
    lines.append("offset from each other and warrant a closer look.")
    lines.append("")

    lines.append("## What parse.py actually reads (quotes)")
    lines.append("")
    lines.append("```")
    lines.append("src/replays/parse.py:10-17")
    lines.append('  steps: list of "step frames", each frame is a list of 2 player dicts:')
    lines.append("      step[i][player] = {action, info, observation, reward, status, ...}")
    lines.append("    - observation.select: None (no decision) or dict with:")
    lines.append("        context (int, matches SelectContext enum), minCount, maxCount, option (list of option dicts)")
    lines.append("    - action: list of ints -- the RESPONSE to step[i-1]'s observation")
    lines.append("      (i.e. the chosen positional indices into step[i-1].select.option)")
    lines.append("    So training pair = (obs = step[i].observation, chosen = step[i+1].action)")
    lines.append("")
    lines.append("src/replays/parse.py:176-222 (parse_replay body)")
    lines.append("  for i in range(n_steps - 1):")
    lines.append("      for player_idx in range(n_players):")
    lines.append('          obs = steps[i][player_idx]["observation"]')
    lines.append('          sel = obs.get("select")')
    lines.append("          ...")
    lines.append("          next_step = steps[i + 1]")
    lines.append('          action = next_step[player_idx].get("action", [])')
    lines.append("          ...")
    lines.append("          if not all(isinstance(x, int) and 0 <= x < n_opts for x in action):")
    lines.append("              continue  # skip invalid")
    lines.append("```")
    lines.append("")
    lines.append("parse.py NEVER reads `steps[0][0][\"visualize\"]` or any `selected` key -- confirmed by")
    lines.append("grep: no occurrence of `visualize` or `selected` anywhere in src/replays/parse.py.")
    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    if verdict1 == "NOT_SHIFTED":
        lines.append("**H1-2 (dataset rebuild + retrain) is NOT needed.** Our IL pipeline (`src/replays/parse.py`)")
        lines.append("reads the top-level kaggle-environments `steps[].observation`/`steps[].action` fields, which")
        lines.append("follow the standard offset convention parse.py already accounts for (action at frame i is")
        lines.append("the response to observation at frame i-1) and show no additional off-by-one in this audit.")
        if verdict2 in ("SHIFTED", "MIXED"):
            lines.append("The forum-reported bug is real in the internal `visualize`/`selected` export (a separate,")
            lines.append("unused code path), but that path is never touched by our parser, so it does not")
            lines.append("contaminate our training data.")
    elif verdict1 == "SHIFTED":
        lines.append("**H1-2 (dataset rebuild + retrain) IS needed.** The evidence indicates parse.py's")
        lines.append("(obs=steps[i], action=steps[i+1]) pairing is itself off by one relative to the true")
        lines.append("decision structure -- every (state, action) pair used for IL is likely mislabeled.")
    else:
        lines.append("**Result is MIXED/AMBIGUOUS** -- recommend manual spot-check of a few flagged replays")
        lines.append("(see examples above) before deciding on H1-2.")
    lines.append("")

    report = "\n".join(lines)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\nReport written to {out_path}")
    print("\n" + "=" * 70)
    print(f"SOURCE 1 (our pipeline) VERDICT: {verdict1}")
    print(f"SOURCE 2 (internal export) VERDICT: {verdict2}")
    print("=" * 70)


if __name__ == "__main__":
    main()
