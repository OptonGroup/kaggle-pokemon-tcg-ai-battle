#!/usr/bin/env python3
"""Full per-game forensic pass over all 129 live tracked-2 episodes
(v044_resub N=68, v046_judgefix N=61) — extends the top-level win/loss CSVs
(data/episodes/{name}_results.csv) with per-game structural signals needed to
find what actually differs between wins and losses, not just narrate losses
(see PLAN.md §L / docs/gold/v047_v048_forensic.md for why: david valor's
"forensic loss-analysis trap", topic 713608).

Reuses src/replays/parse.py::true_decks_from_data (ground-truth 60-card decks,
not board-revealed) instead of re-deriving replay parsing.

Direction of prize counting (verified, src/cg/api.py:476 + past manual
forensics): a player's OWN `prize` count DECREASES when THAT player scores a
knockout (they claim their own prize card). So `6 - prize_remaining` = that
player's own knockout count so far.

Output:
  data/episodes/v044_v046_forensic.csv          — 129 rows, one per episode
  data/episodes/v044_v046_forensic_detail.json  — per-turn snapshots, flagged episodes only
  data/episodes/v044_v046_other_deck_clusters.json — signature clusters for archetype_truth=="other"
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.replays.parse import true_decks_from_data  # noqa: E402

EPISODES_DIR = Path("data/episodes")
RESULT_CSVS = {
    "v044_resub": EPISODES_DIR / "v044_resub_results.csv",
    "v046_judgefix": EPISODES_DIR / "v046_judgefix_results.csv",
}

# Archetype line signatures, copied 1:1 from submissions/v046_judgefix/main.py
# (detect_matchup(), lines ~131-161/538/603-618) — same priority order.
LINE_SIGNATURES = [
    ("crustle", {344, 345, 532}),
    ("hop", {288, 289, 299, 304, 307, 308, 309, 310, 878, 879}),
    ("starmie", {1030, 1031}),
    ("lucario", {677, 678}),
    ("alakazam", {741, 742, 743}),
    ("grimmsnarl", {646, 647, 648}),
    ("dipplin", {89, 90, 92, 93, 247}),
    ("abomasnow", {722, 723}),
    # Added after first forensic pass: "other" bucket clustering (see
    # v044_v046_other_deck_clusters.json) showed these as the two largest
    # unclassified clusters (19 and 9 of 129 games) — not in v046_judgefix's
    # own detect_matchup() because it never needed to recognize a mirror of
    # itself. Same class of gap as the v047 belief-net "no prototype for own
    # archetype" bug (project_deckbelief_three_bugs_found.md).
    ("archaludon_mirror", {169, 190, 666}),  # Duraludon, Archaludon ex, Cinderace
    ("dragapult", {119, 120, 121}),  # Dreepy, Drakloak, Dragapult ex
]

JUDGE, BOSS, XEROSIC = 1213, 1182, 1197
TRACKED_SUPPORTERS = {JUDGE: "judge", BOSS: "boss", XEROSIC: "xerosic"}

PRIZE_MILESTONES = (4, 8, 12, 16)


def classify_archetype(deck_ids: list[int]) -> str:
    ids = set(deck_ids)
    for name, sig in LINE_SIGNATURES:
        if ids & sig:
            return name
    return "other"


def deck_signature(deck_ids: list[int]) -> tuple[int, ...]:
    """Card ids appearing >=2 times in the 60-card deck (deck 'lines' are
    almost always 2-4 copies) — a coarse per-deck fingerprint for clustering
    the 'other' bucket without hand-building new LINE constants up front."""
    counts = Counter(deck_ids)
    return tuple(sorted(cid for cid, n in counts.items() if n >= 2))


def load_card_names() -> dict[int, str]:
    names = {}
    path = Path("data/EN_Card_Data.csv")
    if not path.exists():
        return names
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                names[int(row["Card ID"])] = row["Card Name"]
            except (KeyError, ValueError):
                continue
    return names


def board_archetype_at_frame(cur: dict, opp_idx: int) -> str:
    """detect_matchup()-equivalent from board state alone (active+bench ids),
    reconstructed from raw JSON — mirrors submissions/v046_judgefix/main.py's
    detect_matchup() without importing that module (heavy top-level deps)."""
    players = cur.get("players", [])
    if opp_idx >= len(players):
        return "generic"
    opp = players[opp_idx]
    ids = set()
    for slot in (opp.get("active") or []) + (opp.get("bench") or []):
        if isinstance(slot, dict) and "id" in slot:
            ids.add(slot["id"])
    for name, sig in LINE_SIGNATURES:
        if ids & sig:
            return name
    return "generic"


def analyze_episode(episode_id: str, our_idx: int, submission: str, result: str) -> tuple[dict, dict | None]:
    path = EPISODES_DIR / f"episode-{episode_id}-replay.json"
    data = json.loads(path.read_text())
    steps = data.get("steps", [])
    opp_idx = 1 - our_idx

    decks = true_decks_from_data(data)
    opp_deck = decks.get(opp_idx, [])
    archetype_truth = classify_archetype(opp_deck) if opp_deck else "unknown"
    sig = deck_signature(opp_deck) if opp_deck else ()

    # Walk our_idx's own observation stream once: turn, current, select context.
    our_frames = []  # (step_i, turn, cur)
    for i, step in enumerate(steps):
        if our_idx >= len(step):
            continue
        obs = step[our_idx].get("observation", {})
        cur = obs.get("current")
        if cur is None:
            continue
        our_frames.append((i, cur.get("turn"), cur, obs.get("select"), obs.get("logs") or []))

    n_turns = max((t for _, t, *_ in our_frames if t is not None), default=None)

    # Detect-lag: first turn where board-only archetype guess == ground truth.
    detect_lag_turn = None
    matched_frames = 0
    total_frames = 0
    for _, turn, cur, _sel, _logs in our_frames:
        total_frames += 1
        guess = board_archetype_at_frame(cur, opp_idx)
        truth_for_board = archetype_truth if archetype_truth != "other" else "generic"
        if guess == truth_for_board or (guess != "generic" and guess == archetype_truth):
            matched_frames += 1
            if detect_lag_turn is None and turn is not None:
                detect_lag_turn = turn
    detect_match_frac = matched_frames / total_frames if total_frames else None

    # Prize-trade trajectory: my_taken - opp_taken at each milestone turn (>=),
    # closest frame at-or-after that turn from our own perspective.
    prize_lead = {}
    for milestone in PRIZE_MILESTONES:
        chosen = None
        for _, turn, cur, _sel, _logs in our_frames:
            if turn is None:
                continue
            players = cur.get("players", [])
            if len(players) < 2:
                continue
            if turn >= milestone:
                chosen = players
                break
        if chosen is not None:
            my_taken = 6 - len(chosen[our_idx].get("prize") or [])
            opp_taken = 6 - len(chosen[opp_idx].get("prize") or [])
            prize_lead[f"prize_lead_turn{milestone}"] = my_taken - opp_taken
        else:
            prize_lead[f"prize_lead_turn{milestone}"] = None

    # Final snapshot = last our_idx frame with current.
    final_cur = our_frames[-1][2] if our_frames else None
    prize_lead_final = None
    result_reason = "unclear"
    if final_cur is not None:
        players = final_cur.get("players", [])
        if len(players) >= 2:
            my_taken = 6 - len(players[our_idx].get("prize") or [])
            opp_taken = 6 - len(players[opp_idx].get("prize") or [])
            prize_lead_final = my_taken - opp_taken

            winner_idx = our_idx if result == "win" else (opp_idx if result == "loss" else None)
            loser_idx = opp_idx if winner_idx == our_idx else (our_idx if winner_idx == opp_idx else None)
            if winner_idx is not None:
                winner_prize_left = len(players[winner_idx].get("prize") or [])
                # Kaggle-environments truncates the observation stream right after
                # the terminal action is submitted, so the LAST recorded frame is
                # captured one action *before* the engine applies its effects (the
                # decisive KO's prize decrement never shows up in a snapshot).
                # Empirically confirmed on episode 86944821: final frame showed
                # winner_prize_left=1 with a pending attack log that KOs the
                # opponent's active — the true prize count post-resolution is 0.
                # Threshold relaxed to <=1 to absorb this one-action staleness
                # rather than mislabeling most prize-sweep wins as "unclear".
                if winner_prize_left <= 1:
                    result_reason = "prize_race"
                elif loser_idx is not None:
                    loser = players[loser_idx]
                    loser_board_empty = not (loser.get("active") or []) and not (loser.get("bench") or [])
                    loser_deck_empty = (loser.get("deckCount") or 0) == 0
                    if loser_board_empty:
                        result_reason = "no_more_pokemon"
                    elif loser_deck_empty:
                        result_reason = "deck_out"

    # Bench-thinness: our own MAIN-context decision frames with bench<=1.
    main_frames = 0
    thin_frames = 0
    for _, _turn, cur, sel, _logs in our_frames:
        if not sel or sel.get("context") != 0:
            continue
        players = cur.get("players", [])
        if len(players) <= our_idx:
            continue
        main_frames += 1
        if len(players[our_idx].get("bench") or []) <= 1:
            thin_frames += 1
    bench_thin_frac = thin_frames / main_frames if main_frames else None

    # Judge/Boss/Xerosic play timeline, dedup by (cardId, serial, playerIndex).
    seen_plays = set()
    supporter_turns = {f"{name}_played_us": None for name in TRACKED_SUPPORTERS.values()}
    supporter_turns.update({f"{name}_played_opp": None for name in TRACKED_SUPPORTERS.values()})
    for _, turn, _cur, _sel, logs in our_frames:
        for log in logs:
            if log.get("type") != 10:
                continue
            cid = log.get("cardId")
            if cid not in TRACKED_SUPPORTERS:
                continue
            key = (cid, log.get("serial"), log.get("playerIndex"))
            if key in seen_plays:
                continue
            seen_plays.add(key)
            name = TRACKED_SUPPORTERS[cid]
            who = "us" if log.get("playerIndex") == our_idx else "opp"
            field = f"{name}_played_{who}"
            if supporter_turns.get(field) is None:
                supporter_turns[field] = turn

    row = {
        "episode_id": episode_id,
        "submission": submission,
        "our_idx": our_idx,
        "opponent_team": None,  # filled by caller from results CSV
        "archetype_truth": archetype_truth,
        "deck_signature_len": len(sig),
        "result": result,
        "n_turns": n_turns,
        "detect_lag_turn": detect_lag_turn,
        "detect_match_frac": round(detect_match_frac, 3) if detect_match_frac is not None else None,
        "bench_thin_frac": round(bench_thin_frac, 3) if bench_thin_frac is not None else None,
        "result_reason": result_reason,
        "prize_lead_final": prize_lead_final,
        **prize_lead,
        **supporter_turns,
    }

    # Flag "interesting" for detail dump.
    close_loss = result == "loss" and prize_lead_final is not None and prize_lead_final >= -1
    poor_detect = detect_lag_turn is None or (n_turns and detect_lag_turn is not None and detect_lag_turn > 6)
    rare_cluster = archetype_truth == "other"
    is_interesting = close_loss or poor_detect or rare_cluster

    detail = None
    if is_interesting:
        detail = {
            "episode_id": episode_id,
            "submission": submission,
            "flags": {
                "close_loss": close_loss,
                "poor_detect": poor_detect,
                "rare_cluster": rare_cluster,
            },
            "deck_signature": list(sig),
            "timeline": [
                {
                    "turn": turn,
                    "my_prize": len(cur.get("players", [{}, {}])[our_idx].get("prize") or [])
                    if len(cur.get("players", [])) > our_idx else None,
                    "opp_prize": len(cur.get("players", [{}, {}])[opp_idx].get("prize") or [])
                    if len(cur.get("players", [])) > opp_idx else None,
                    "my_bench": len(cur.get("players", [{}, {}])[our_idx].get("bench") or [])
                    if len(cur.get("players", [])) > our_idx else None,
                }
                for _, turn, cur, _sel, _logs in our_frames
                if turn is not None
            ],
        }

    return row, detail


def main():
    names = load_card_names()
    all_rows = []
    all_details = []
    other_signatures: Counter = Counter()
    other_examples: dict[tuple[int, ...], str] = {}
    errors = []

    for submission, csv_path in RESULT_CSVS.items():
        if not csv_path.exists():
            print(f"WARN: missing {csv_path}", file=sys.stderr)
            continue
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                episode_id = r["episode_id"]
                our_idx = int(r["our_idx"])
                result = r["result"]
                opponent_team = r["opponent_team"]
                try:
                    row, detail = analyze_episode(episode_id, our_idx, submission, result)
                except Exception as e:
                    errors.append((episode_id, str(e)))
                    print(f"  ERROR episode {episode_id}: {e}", file=sys.stderr)
                    continue
                row["opponent_team"] = opponent_team
                all_rows.append(row)
                if detail is not None:
                    all_details.append(detail)
                if row["archetype_truth"] == "other":
                    path = EPISODES_DIR / f"episode-{episode_id}-replay.json"
                    data = json.loads(path.read_text())
                    deck = true_decks_from_data(data).get(1 - our_idx, [])
                    sig = deck_signature(deck)
                    other_signatures[sig] += 1
                    other_examples.setdefault(sig, opponent_team)

    out_csv = EPISODES_DIR / "v044_v046_forensic.csv"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    out_detail = EPISODES_DIR / "v044_v046_forensic_detail.json"
    out_detail.write_text(json.dumps(all_details, indent=2))

    other_clusters = [
        {
            "signature": list(sig),
            "signature_names": [names.get(cid, f"id{cid}") for cid in sig],
            "count": count,
            "example_opponent": other_examples[sig],
        }
        for sig, count in other_signatures.most_common()
    ]
    out_clusters = EPISODES_DIR / "v044_v046_other_deck_clusters.json"
    out_clusters.write_text(json.dumps(other_clusters, indent=2))

    print(f"Wrote {len(all_rows)} rows to {out_csv}")
    print(f"Wrote {len(all_details)} flagged episode details to {out_detail}")
    print(f"Wrote {len(other_clusters)} 'other' deck clusters to {out_clusters}")
    if errors:
        print(f"\n{len(errors)} episodes failed to parse:", file=sys.stderr)
        for eid, err in errors:
            print(f"  {eid}: {err}", file=sys.stderr)

    for sub in RESULT_CSVS:
        sub_rows = [r for r in all_rows if r["submission"] == sub]
        w = sum(1 for r in sub_rows if r["result"] == "win")
        l = sum(1 for r in sub_rows if r["result"] == "loss")
        print(f"{sub}: N={len(sub_rows)} W-L={w}-{l}")


if __name__ == "__main__":
    main()
