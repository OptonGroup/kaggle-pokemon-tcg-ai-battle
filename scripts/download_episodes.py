#!/usr/bin/env python3
"""Download all episode replays + our agent logs for one or more Kaggle submissions.

Uses kagglesdk directly (bypasses the CLI's KGAT-token issue, see memory
project_kaggle_kgat_token). Rate-limited: max 2 workers, 0.6s sleep between
calls, explicit 429 backoff.

Usage:
    .venv/bin/python scripts/download_episodes.py <out_dir> <label>=<submission_id> [<label>=<id> ...]
    # example:
    .venv/bin/python scripts/download_episodes.py out/v062_full_download A=54887093 B=54887100

Produces under <out_dir>:
    replays/{episode_id}.json                     full replays
    agent_logs_raw/{episode_id}-agent-{idx}.json  raw per-step stdout/stderr of OUR agent
    index.csv                                     episode_id, copy, our_index, our_reward,
                                                  opp team/submission/team_id, timestamps

Follow with:
    python3 scripts/extract_agent_logs.py <out_dir>       # decision-trace .txt
    python3 scripts/replay_digest.py --batch <out_dir>    # readable digests
"""
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiListSubmissionEpisodesRequest,
    ApiGetEpisodeReplayRequest,
    ApiGetEpisodeAgentLogsRequest,
)


def fetch_with_retry(fn, *args, max_retries=8, **kwargs):
    delay = 3.0
    for attempt in range(max_retries):
        try:
            resp = fn(*args, **kwargs)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 429 or "429" in str(e):
                print(f"    429 rate-limited, backing off {delay:.1f}s (attempt {attempt+1})", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 1.8, 60.0)
                continue
            print(f"    error on attempt {attempt+1}: {e}", file=sys.stderr)
            time.sleep(delay)
            continue
        if getattr(resp, "status_code", 200) == 200:
            return resp
        if resp.status_code == 429:
            print(f"    429 rate-limited, backing off {delay:.1f}s (attempt {attempt+1})", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 1.8, 60.0)
            continue
        print(f"    HTTP {resp.status_code} on attempt {attempt+1}", file=sys.stderr)
        time.sleep(delay)
    return None


def process_episode(client, copy_label, sub_id, ep, replay_dir, rawlog_dir):
    ep_id = ep.id
    our_agent = next((a for a in ep.agents if a.submission_id == sub_id), None)
    if our_agent is None:
        return {"episode_id": ep_id, "copy": copy_label, "status": "no_our_agent"}
    opp_agent = next((a for a in ep.agents if a.submission_id != sub_id), None)
    result = {
        "episode_id": ep_id,
        "copy": copy_label,
        "submission_id": sub_id,
        "our_index": our_agent.index,
        "our_reward": our_agent.reward,
        "opp_team_name": opp_agent.team_name if opp_agent else None,
        "opp_submission_id": opp_agent.submission_id if opp_agent else None,
        "opp_team_id": opp_agent.team_id if opp_agent else None,
        "create_time": str(ep.create_time),
        "end_time": str(ep.end_time),
    }

    replay_path = replay_dir / f"{ep_id}.json"
    if not replay_path.exists():
        req = ApiGetEpisodeReplayRequest()
        req.episode_id = ep_id
        resp = fetch_with_retry(client.competitions.competition_api_client.get_episode_replay, req)
        if resp is None:
            result["status"] = "replay_failed"
            return result
        replay_path.write_bytes(resp.content)
        time.sleep(0.6)

    raw_log_path = rawlog_dir / f"{ep_id}-agent-{our_agent.index}.json"
    if not raw_log_path.exists():
        req2 = ApiGetEpisodeAgentLogsRequest()
        req2.episode_id = ep_id
        req2.agent_index = our_agent.index
        resp2 = fetch_with_retry(client.competitions.competition_api_client.get_episode_agent_logs, req2)
        if resp2 is None:
            result["status"] = "logs_failed"
            return result
        raw_log_path.write_bytes(resp2.content)
        time.sleep(0.6)

    result["status"] = "ok"
    return result


def main():
    out_dir = Path(sys.argv[1])
    submissions = {}
    for arg in sys.argv[2:]:
        label, _, sid = arg.partition("=")
        submissions[label] = int(sid)
    if not submissions:
        sys.exit("usage: download_episodes.py <out_dir> <label>=<submission_id> ...")

    replay_dir = out_dir / "replays"
    rawlog_dir = out_dir / "agent_logs_raw"
    for d in (replay_dir, rawlog_dir):
        d.mkdir(parents=True, exist_ok=True)

    client = KaggleClient()
    all_results = []
    for copy_label, sub_id in submissions.items():
        print(f"=== Listing episodes for {copy_label} (submission {sub_id}) ===")
        req = ApiListSubmissionEpisodesRequest()
        req.submission_id = sub_id
        resp = client.competitions.competition_api_client.list_submission_episodes(req)
        episodes = [e for e in resp.episodes if str(e.state).endswith("COMPLETED")]
        print(f"  {len(episodes)} completed episodes")

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(process_episode, client, copy_label, sub_id, ep,
                            replay_dir, rawlog_dir): ep.id
                for ep in episodes
            }
            done = 0
            for fut in as_completed(futures):
                all_results.append(fut.result())
                done += 1
                if done % 10 == 0 or done == len(episodes):
                    print(f"  {done}/{len(episodes)} done")

    csv_path = out_dir / "index.csv"
    fieldnames = sorted({k for r in all_results for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            w.writerow(r)

    ok = sum(1 for r in all_results if r.get("status") == "ok")
    wins = sum(1 for r in all_results if r.get("status") == "ok" and (r.get("our_reward") or 0) > 0)
    losses = sum(1 for r in all_results if r.get("status") == "ok" and (r.get("our_reward") or 0) < 0)
    print(f"\nTotal: {len(all_results)}, ok={ok}, wins={wins}, losses={losses}")
    print(f"Index CSV: {csv_path}")


if __name__ == "__main__":
    main()
