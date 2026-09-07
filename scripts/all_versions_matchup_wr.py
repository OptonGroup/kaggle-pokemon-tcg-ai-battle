#!/usr/bin/env python3
"""Full-history matchup WR: every submission we ever made vs every opponent archetype.

Two phases, both resumable (checkpointed to CSV after every submission):

  phase 1 (cheap)  list_submission_episodes for all refs -> out/all_versions_wr/index.csv
                   exact W/L per version for the whole history, no replays touched.
  phase 2 (heavy)  for each episode not yet classified, fetch the replay, read the
                   opponent's initial deck, classify() it, and THROW THE REPLAY AWAY.
                   ~3.3MB/episode over the wire but ~0 bytes on disk.

Rows already classified in any out/*_wr_check/archetype_index.csv are reused, so the
21 submissions analysed earlier cost nothing.

Usage:
    .venv/bin/python scripts/all_versions_matchup_wr.py index      # phase 1
    .venv/bin/python scripts/all_versions_matchup_wr.py archetypes # phase 2 (resumable)
    .venv/bin/python scripts/all_versions_matchup_wr.py report     # aggregate
"""
import csv
import glob
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify_opponent_archetype import classify, find_initial_players

from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiListSubmissionEpisodesRequest,
    ApiGetEpisodeReplayRequest,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "all_versions_wr"
INDEX = OUT / "index.csv"
ARCHE = OUT / "archetypes.csv"
SUBS = OUT / "submissions.csv"

INDEX_FIELDS = ["version", "submission_id", "sub_date", "public_score", "episode_id",
                "our_index", "our_reward", "opp_team_id", "opp_team_name",
                "opp_submission_id", "create_time"]


def fetch_with_retry(fn, req, max_retries=8):
    delay = 3.0
    for attempt in range(max_retries):
        try:
            resp = fn(req)
        except Exception as e:
            if "429" in str(e):
                time.sleep(delay)
                delay = min(delay * 1.8, 60.0)
                continue
            time.sleep(delay)
            continue
        if getattr(resp, "status_code", 200) == 200:
            return resp
        if resp.status_code == 429:
            time.sleep(delay)
            delay = min(delay * 1.8, 60.0)
            continue
        time.sleep(delay)
    return None


def load_submissions():
    """All submissions ever, newest first: [{ref, version, date, score}]."""
    rows = list(csv.DictReader(open(SUBS)))
    out = []
    for r in rows:
        ver = r["fileName"].replace(".tar.gz", "")
        out.append({"ref": int(r["ref"]), "version": ver, "date": r["date"],
                    "score": r.get("publicScore") or ""})
    return out


# ---------------------------------------------------------------- phase 1
def phase_index():
    subs = load_submissions()
    done = set()
    rows = []
    if INDEX.exists():
        rows = list(csv.DictReader(open(INDEX)))
        done = {int(r["submission_id"]) for r in rows}
    client = KaggleClient()
    for i, s in enumerate(subs, 1):
        if s["ref"] in done:
            continue
        req = ApiListSubmissionEpisodesRequest()
        req.submission_id = s["ref"]
        resp = fetch_with_retry(
            client.competitions.competition_api_client.list_submission_episodes, req)
        if resp is None:
            print(f"[{i}/{len(subs)}] {s['version']} ({s['ref']}): LIST FAILED", flush=True)
            continue
        eps = [e for e in resp.episodes if str(e.state).endswith("COMPLETED")]
        n = 0
        for ep in eps:
            ours = next((a for a in ep.agents if a.submission_id == s["ref"]), None)
            opp = next((a for a in ep.agents if a.submission_id != s["ref"]), None)
            if ours is None:
                continue
            rows.append({
                "version": s["version"], "submission_id": s["ref"], "sub_date": s["date"],
                "public_score": s["score"], "episode_id": ep.id,
                "our_index": ours.index, "our_reward": ours.reward,
                "opp_team_id": opp.team_id if opp else "",
                "opp_team_name": opp.team_name if opp else "",
                "opp_submission_id": opp.submission_id if opp else "",
                "create_time": str(ep.create_time),
            })
            n += 1
        print(f"[{i}/{len(subs)}] {s['version']} ({s['ref']}): {n} episodes", flush=True)
        write_index(rows)
        time.sleep(0.4)
    write_index(rows)
    print(f"\ntotal episodes: {len(rows)}  ->  {INDEX}")


def write_index(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(INDEX, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- phase 2
def load_known_archetypes():
    """episode_id -> (archetype, key_cards) from every earlier wr_check run."""
    known = {}
    for path in glob.glob(str(ROOT / "out" / "*_wr_check" / "archetype_index.csv")):
        for r in csv.DictReader(open(path)):
            a = r.get("archetype", "")
            if a and a not in ("NO_REPLAY", "NO_DECK_DATA"):
                known[r["episode_id"]] = (a, r.get("opp_key_cards", ""))
    return known


def classify_episode(client, row):
    req = ApiGetEpisodeReplayRequest()
    req.episode_id = int(row["episode_id"])
    resp = fetch_with_retry(client.competitions.competition_api_client.get_episode_replay, req)
    if resp is None:
        return {**row, "archetype": "FETCH_FAILED", "opp_key_cards": ""}
    try:
        replay = json.loads(resp.content)
        players = find_initial_players(replay)
    except Exception:
        players = None
    if players is None:
        return {**row, "archetype": "NO_DECK_DATA", "opp_key_cards": ""}
    opp_index = 1 - int(row["our_index"])
    deck = players[opp_index].get("deck", [])
    ids = {c["id"] for c in deck}
    names = [c["name"] for c in deck]
    arche = classify(ids, names) or "UNKNOWN"
    top = ", ".join(f"{n}x{c}" for n, c in Counter(names).most_common(6))
    time.sleep(0.5)
    return {**row, "archetype": arche, "opp_key_cards": top}


def phase_archetypes():
    idx = list(csv.DictReader(open(INDEX)))
    fields = INDEX_FIELDS + ["archetype", "opp_key_cards"]
    have = {}
    if ARCHE.exists():
        for r in csv.DictReader(open(ARCHE)):
            if r["archetype"] not in ("FETCH_FAILED",):
                have[r["episode_id"]] = r
    known = load_known_archetypes()
    print(f"reusing {len(have)} checkpointed + {len(known)} from earlier wr_check dirs")

    todo = []
    for r in idx:
        ep = r["episode_id"]
        if ep in have:
            continue
        if ep in known:
            a, k = known[ep]
            have[ep] = {**r, "archetype": a, "opp_key_cards": k}
            continue
        todo.append(r)

    # Kaggle throttles GetEpisodeReplay to ~5/min, so the fetch order decides how
    # fast coverage grows: an opponent's archetype is a property of THEIR deck, so
    # one replay per never-seen team classifies every game against that team by
    # propagation (see all_versions_report.py). Teams already classified go last —
    # they only refine what propagation already covers.
    seen_teams = {r.get("opp_team_id") for r in have.values() if r.get("opp_team_id")}
    first, rest = [], []
    for r in todo:
        t = r.get("opp_team_id")
        if t and t not in seen_teams:
            seen_teams.add(t)
            first.append(r)
        else:
            rest.append(r)
    todo = first + rest
    print(f"to fetch: {len(todo)} episodes "
          f"({len(first)} новых команд сначала, {len(rest)} уточняющих)")
    flush(have, fields)

    client = KaggleClient()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(classify_episode, client, r): r["episode_id"] for r in todo}
        done = 0
        for fut in as_completed(futs):
            res = fut.result()
            have[res["episode_id"]] = res
            done += 1
            if done % 25 == 0 or done == len(todo):
                flush(have, fields)
                print(f"  {done}/{len(todo)}", flush=True)
    flush(have, fields)
    print(f"done -> {ARCHE}")


def flush(have, fields):
    with open(ARCHE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in have.values():
            w.writerow({k: r.get(k, "") for k in fields})


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "index"
    if cmd == "index":
        phase_index()
    elif cmd == "archetypes":
        phase_archetypes()
    else:
        sys.exit(f"unknown command {cmd}")
