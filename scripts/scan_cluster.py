#!/usr/bin/env python3
"""scan_cluster.py — harvest a specific team's replays by mining around known hits.

A blind walk over episode ids wastes ~98% of its downloads: any one team plays a
few dozen games a day out of ~180k episodes competition-wide. But matchmaking
schedules a team's games in bursts — observed gaps between consecutive
Majkel1337 episodes are 0, 4, 5, 17... not hundreds. So: start from known hits,
sweep their neighbourhoods, and expand around every new hit found.

    KAGGLE_API_TOKEN=KGAT_... python3 scripts/scan_cluster.py \
        --seeds 86120446,86119857 --radius 40 --targets-file targets.txt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent import futures
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
REPLAY_URL = "https://www.kaggle.com/api/v1/competitions/episodes/{}/replay"

_LOCAL = threading.local()


class RateLimited(Exception):
    """429 — applies to the whole account and persists while we keep asking."""


def _token() -> str:
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        path = Path.home() / ".kaggle" / "access_token"
        if path.exists():
            token = path.read_text().strip()
    if not token:
        raise SystemExit("no KAGGLE_API_TOKEN and no ~/.kaggle/access_token")
    return token


def _session():
    session = getattr(_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {_token()}"
        _LOCAL.session = session
    return session


def fetch(episode_id: int, raw_dir: Path, timeout: int) -> dict | None:
    dest = raw_dir / f"episode-{episode_id}-replay.json"
    if dest.exists():
        try:
            return json.loads(dest.read_text())
        except ValueError:
            dest.unlink(missing_ok=True)
    try:
        resp = _session().get(REPLAY_URL.format(episode_id), timeout=timeout)
    except Exception:
        return None
    if resp.status_code == 429:
        raise RateLimited(str(episode_id))
    if resp.status_code != 200 or not resp.content:
        return None
    dest.write_bytes(resp.content)
    try:
        return json.loads(resp.content)
    except ValueError:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", required=True, help="comma-separated known episode ids")
    p.add_argument("--radius", type=int, default=40, help="ids to sweep either side of a hit")
    p.add_argument("--targets-file", required=True)
    p.add_argument("--out-dir", default=str(REPO / "data" / "cluster_scan"))
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--max-fetch", type=int, default=4000, help="stop after this many downloads")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    targets = {ln.strip().lower() for ln in open(args.targets_file) if ln.strip()}
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    seen: set[int] = set()
    queue: list[int] = []
    for seed in seeds:
        queue.extend(range(seed - args.radius, seed + args.radius + 1))

    hits_path = out_dir / "hits.jsonl"
    lock = threading.Lock()
    stop = threading.Event()
    stats = {"fetched": 0, "hits": 0, "misses": 0}
    t0 = time.perf_counter()

    def work(episode_id: int) -> list[int]:
        """Fetch one episode; return new ids to sweep if a target played in it."""
        if stop.is_set() or episode_id in seen:
            return []
        with lock:
            if episode_id in seen:
                return []
            seen.add(episode_id)
            if stats["fetched"] >= args.max_fetch:
                stop.set()
                return []

        time.sleep(args.sleep * (0.5 + random.random()))
        try:
            data = fetch(episode_id, raw_dir, args.timeout)
        except RateLimited:
            if not stop.is_set():
                stop.set()
                print(f"[cluster] RATE LIMITED at {episode_id} — stopping.", flush=True)
            return []

        with lock:
            stats["fetched"] += 1
        if not data:
            with lock:
                stats["misses"] += 1
            return []

        names = [str(n) for n in ((data.get("info") or {}).get("TeamNames") or [])]
        matched = [n for n in names if n.lower() in targets]
        if not matched:
            return []

        steps = data.get("steps") or []
        decks = []
        if len(steps) > 1:
            for i in range(2):
                action = steps[1][i].get("action") if i < len(steps[1]) else None
                decks.append(action if isinstance(action, list) and len(action) == 60 else None)
        rec = {
            "episode_id": (data.get("info") or {}).get("EpisodeId"),
            "teams": names,
            "decks": decks,
            "rewards": [s.get("reward") for s in (steps[-1] or []) if isinstance(s, dict)],
            "n_steps": len(steps),
        }
        with lock:
            stats["hits"] += 1
            with open(hits_path, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            elapsed = time.perf_counter() - t0
            print(f"[cluster] HIT {episode_id} {names} | hits={stats['hits']} "
                  f"fetched={stats['fetched']} rate={stats['fetched']/max(elapsed,1):.1f}/s",
                  flush=True)
        # A hit means this team is active around here — sweep wider.
        return [e for e in range(episode_id - args.radius, episode_id + args.radius + 1)
                if e not in seen]

    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while queue and not stop.is_set():
            batch, queue = queue[:400], queue[400:]
            for extra in pool.map(work, batch):
                queue.extend(extra)

    elapsed = time.perf_counter() - t0
    print(f"\n[cluster] DONE fetched={stats['fetched']} hits={stats['hits']} "
          f"misses={stats['misses']} elapsed={elapsed/60:.1f}min -> {hits_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
