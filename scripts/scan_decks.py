#!/usr/bin/env python3
"""scan_decks.py — mine 60-card deck lists from ANY team's episode replays.

GetEpisodeReplay is not permission-scoped: `kaggle competitions replay <id>`
downloads any episode in the competition, and the replay carries both decks
unredacted plus the leaderboard display names. Episode ids form one global
monotonic counter across the whole competition (~130 new episodes/minute), so
finding a specific team means walking ids and checking info.TeamNames.

Replays are 1-6 MB each and are deleted after parsing unless --keep.

    python3 scripts/scan_decks.py --start 86121184 --count 2000
    python3 scripts/scan_decks.py --start 86121184 --count 2000 --targets-file targets.txt

Progress is checkpointed to the output dir, so re-running resumes where it
stopped. Hits are appended to decks.jsonl as one record per episode.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KAGGLE = REPO / ".venv" / "bin" / "kaggle"

# Top-10 as of 2026-07-15 leaderboard pull.
DEFAULT_TARGETS = [
    "Majkel1337",
    "Yushin Ito",
    "Jack",
    "vibechu",
    "THIRD PTCG Club",
    "懒惰的金枪鱼",
    "__Taichicchi__",
    "Luca",
    "jiatu.l",
    "bono",
]


# The `kaggle` CLI authenticates only via ~/.kaggle/kaggle.json (username + key)
# and rejects the KGAT_ tokens the site issues now, so shelling out to it fails
# with "Authentication required". The SDK takes the token but points PROD at
# api.kaggle.com, which 404s every call; www.kaggle.com serves the same v1 API.
# Hitting REST directly sidesteps both problems.
_REPLAY_URL = "https://www.kaggle.com/api/v1/competitions/episodes/{}/replay"

_SESSION_LOCAL = threading.local()


def _token() -> str:
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        token_file = Path.home() / ".kaggle" / "access_token"
        if token_file.exists():
            token = token_file.read_text().strip()
    if not token:
        raise RuntimeError("no KAGGLE_API_TOKEN and no ~/.kaggle/access_token")
    return token


def _session():
    """One pooled session per worker thread."""
    session = getattr(_SESSION_LOCAL, "session", None)
    if session is None:
        import requests

        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {_token()}"
        _SESSION_LOCAL.session = session
    return session


class RateLimited(Exception):
    """Kaggle returned 429 — back off, do not treat as a missing episode."""


def fetch_replay(episode_id: int, tmp_dir: Path, timeout: int) -> Path | None:
    dest = tmp_dir / f"episode-{episode_id}-replay.json"
    if dest.exists():
        return dest
    try:
        resp = _session().get(_REPLAY_URL.format(episode_id), timeout=timeout)
    except Exception:
        return None
    # 429 applies to the whole account and persists — every further request
    # extends it. It must stop the scan, not look like a missing episode.
    if resp.status_code == 429:
        raise RateLimited(f"429 on episode {episode_id}")
    # Ids are a global counter across every Kaggle simulation competition, so a
    # 404 usually just means the episode belongs to a different one.
    if resp.status_code != 200 or not resp.content:
        return None
    dest.write_bytes(resp.content)
    return dest


def parse_replay(path: Path) -> dict | None:
    """Pull team names + both 60-card decks out of a replay."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    info = data.get("info") or {}
    names = info.get("TeamNames")
    steps = data.get("steps")
    if not names or not steps or len(steps) < 2 or len(steps[1]) < 2:
        return None

    decks = []
    for agent_idx in range(2):
        action = steps[1][agent_idx].get("action")
        decks.append(action if isinstance(action, list) and len(action) == 60 else None)

    rewards = None
    if steps[-1] and isinstance(steps[-1], list):
        rewards = [s.get("reward") for s in steps[-1] if isinstance(s, dict)]

    return {
        "episode_id": info.get("EpisodeId"),
        "teams": names,
        "decks": decks,
        "rewards": rewards,
        "n_steps": len(steps),
    }


def load_checkpoint(out_dir: Path) -> dict:
    path = out_dir / "checkpoint.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {"scanned": [], "last_id": None, "hits": 0, "errors": 0}


def save_checkpoint(out_dir: Path, ckpt: dict) -> None:
    (out_dir / "checkpoint.json").write_text(json.dumps(ckpt, indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Mine deck lists from competition episode replays")
    p.add_argument("--start", type=int, required=True, help="episode id to start from")
    p.add_argument("--count", type=int, default=1000, help="how many ids to walk")
    p.add_argument("--step", type=int, default=-1,
                   help="id increment; negative walks backwards in time (default -1)")
    p.add_argument("--out-dir", type=str, default=str(REPO / "data" / "deck_scan"))
    p.add_argument("--targets-file", type=str, default="",
                   help="file with one target team name per line (default: hardcoded top-10)")
    p.add_argument("--all-teams", action="store_true",
                   help="record every episode, not just target teams")
    # 8 workers with no delay looks fine for the first few hundred requests and
    # then earns a 429 on the whole account — the limit accumulates. Stay gentle.
    p.add_argument("--workers", type=int, default=3, help="parallel downloads")
    p.add_argument("--sleep", type=float, default=0.5, help="base delay per request")
    p.add_argument("--timeout", type=int, default=90, help="per-download timeout")
    p.add_argument("--keep", action="store_true", help="keep raw replay JSONs")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    tmp_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.targets_file:
        targets = [ln.strip() for ln in open(args.targets_file) if ln.strip()]
    else:
        targets = DEFAULT_TARGETS
    target_set = {t.lower() for t in targets}

    ckpt = load_checkpoint(out_dir)
    scanned = set(ckpt.get("scanned", []))
    hits_path = out_dir / "decks.jsonl"

    print(f"[scan] start={args.start} count={args.count} step={args.step}", flush=True)
    print(f"[scan] targets={len(targets)} all_teams={args.all_teams} out={out_dir}", flush=True)
    print(f"[scan] resuming: {len(scanned)} ids already scanned, {ckpt.get('hits', 0)} hits", flush=True)

    hits = ckpt.get("hits", 0)
    errors = ckpt.get("errors", 0)
    t0 = time.perf_counter()
    processed = 0
    lock = threading.Lock()

    todo = [args.start + i * args.step for i in range(args.count)]
    todo = [e for e in todo if e not in scanned]

    stop = threading.Event()

    def handle(episode_id: int) -> None:
        """Fetch one episode and record it if a target team played in it."""
        nonlocal hits, errors, processed

        if stop.is_set():
            return
        if args.sleep:
            time.sleep(args.sleep * (0.5 + random.random()))
        try:
            path = fetch_replay(episode_id, tmp_dir, args.timeout)
        except RateLimited as err:
            if not stop.is_set():
                stop.set()
                print(f"[scan] RATE LIMITED ({err}) — stopping. Checkpoint saved; "
                      f"resume later with a lower --workers / higher --sleep.", flush=True)
                ckpt.update({"scanned": sorted(scanned), "hits": hits, "errors": errors})
                save_checkpoint(out_dir, ckpt)
            return
        if path is None:
            # Ids are a global counter across all Kaggle simulation competitions,
            # so a miss usually means the episode belongs to another one, not that
            # we are being throttled.
            with lock:
                errors += 1
                scanned.add(episode_id)
            return

        rec = parse_replay(path)
        if not args.keep:
            path.unlink(missing_ok=True)

        with lock:
            scanned.add(episode_id)
            processed += 1
            if rec:
                names = [str(n) for n in (rec["teams"] or [])]
                matched = [n for n in names if n.lower() in target_set]
                if matched or args.all_teams:
                    with open(hits_path, "a") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if matched:
                        hits += 1
                        print(f"[scan] HIT ep={episode_id} teams={names} matched={matched}",
                              flush=True)
            if processed % 50 == 0:
                elapsed = time.perf_counter() - t0
                rate = processed / elapsed if elapsed > 0 else 0.0
                eta = (len(todo) - processed) / rate / 60 if rate > 0 else 0.0
                print(f"[scan] {processed}/{len(todo)} | id={episode_id} hits={hits} "
                      f"errors={errors} rate={rate:.2f}/s eta={eta:.0f}min", flush=True)
                ckpt.update({"scanned": sorted(scanned), "last_id": episode_id,
                             "hits": hits, "errors": errors})
                save_checkpoint(out_dir, ckpt)

    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(handle, todo))

    ckpt.update({"scanned": sorted(scanned), "last_id": args.start + (args.count - 1) * args.step,
                 "hits": hits, "errors": errors})
    save_checkpoint(out_dir, ckpt)
    elapsed = time.perf_counter() - t0
    print(f"[scan] DONE processed={processed} hits={hits} errors={errors} "
          f"elapsed={elapsed:.0f}s -> {hits_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
