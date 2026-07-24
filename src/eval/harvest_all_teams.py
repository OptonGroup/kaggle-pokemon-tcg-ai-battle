"""
harvest_all_teams.py — Пилотный харвест эпизодов всех команд.

Алгоритм:
  1. Читает лидерборд (первые N страниц, фильтруем по min_score).
  2. Для каждой команды берёт лучший публичный сабмит.
  3. Получает список эпизодов сабмита через kaggle CLI.
  4. Скачивает реплеи (replay JSON) для эпизодов, которых ещё нет локально.
  5. Сохраняет в data/episodes/all_teams/.

Запуск:
    .venv/bin/python src/eval/harvest_all_teams.py [--max-teams N] [--min-score S] [--dry-run]

Аргументы:
    --max-teams N     Ограничить кол-во команд (по умолчанию: 20 для пилота)
    --min-score S     Минимальный рейтинг команды (по умолчанию: 900)
    --dry-run         Не качать, только напечатать план
    --sleep SECS      Пауза между API-вызовами (по умолчанию: 0.5)
"""

import csv
import subprocess
import sys
import os
import time
import re
import json
import argparse
from pathlib import Path

REPO = Path(__file__).parent.parent.parent
EPISODES_DIR = REPO / "data" / "episodes"
ALL_TEAMS_DIR = REPO / "data" / "episodes" / "all_teams"
KAGGLE = str(REPO / ".venv" / "bin" / "kaggle")
COMP = "pokemon-tcg-ai-battle"

CRUSTLE_IDS = {344, 345}  # Dwebble, Crustle


def run(cmd: list[str], sleep_after: float = 0.0, timeout: int = 60) -> str:
    """Run a kaggle CLI command and return stdout."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    if sleep_after:
        time.sleep(sleep_after)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def get_leaderboard(min_score: float = 900.0, max_teams: int = 20) -> list[dict]:
    """Fetch leaderboard pages until we have enough teams >= min_score."""
    teams = []
    page_token = None
    page = 0
    while True:
        cmd = [KAGGLE, "competitions", "leaderboard", COMP, "--show", "--csv", "--page-size", "200"]
        if page_token:
            cmd += ["--page-token", page_token]
        out = run(cmd, sleep_after=0.5)
        lines = out.split("\n")

        # Extract next page token
        next_token = None
        data_lines = []
        for line in lines:
            if line.startswith("Next Page Token"):
                next_token = line.split("=", 1)[1].strip()
            elif line.startswith("teamId") or "," in line:
                data_lines.append(line)

        # Parse CSV rows (skip header). Use a real CSV reader, not line.split(","):
        # team names are quoted and some contain commas ("Jhu, Jhe-Yi", "Pikachu, I
        # choose you!"), which shifts every later field so float(parts[3]) raises and
        # the team is dropped without a word. Measured: 2 teams silently lost per
        # 1200 rows of the live leaderboard, one of them inside our own score band.
        for parts in csv.reader(data_lines):
            if not parts or parts[0].strip() == "teamId" or len(parts) < 4:
                continue
            try:
                team_id = int(parts[0].strip())
                team_name = parts[1].strip()
                score = float(parts[3].strip())
                if score >= min_score:
                    teams.append({"teamId": team_id, "teamName": team_name, "score": score})
            except (ValueError, IndexError):
                continue

        page += 1
        if not next_token or len(teams) >= max_teams * 3:
            break
        if not any(t["score"] >= min_score for t in teams[-10:]):
            # Last batch was all below threshold — stop
            break
        page_token = next_token

    # Sort by score desc, dedupe, take top N
    seen = set()
    result = []
    for t in sorted(teams, key=lambda x: -x["score"]):
        if t["teamId"] not in seen:
            seen.add(t["teamId"])
            result.append(t)
        if len(result) >= max_teams:
            break
    return result


def get_best_submission(team_id: int, sleep: float = 0.5) -> int | None:
    """Return the submission_id with the highest score for a team."""
    try:
        out = run([KAGGLE, "competitions", "team-submissions", str(team_id), "--csv"], sleep_after=sleep)
    except RuntimeError as e:
        print(f"  WARN: team-submissions failed for {team_id}: {e}")
        return None
    lines = [l for l in out.split("\n") if l and not l.startswith("id")]
    best_sub = None
    best_score = -1.0
    for line in lines:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            sub_id = int(parts[0].strip())
            score = float(parts[2].strip())
            if score > best_score:
                best_score = score
                best_sub = sub_id
        except (ValueError, IndexError):
            continue
    return best_sub


def get_episodes(submission_id: int, sleep: float = 0.3) -> list[int]:
    """Return list of episode IDs for a submission (PUBLIC only)."""
    try:
        out = run([KAGGLE, "competitions", "episodes", str(submission_id), "--csv"], sleep_after=sleep)
    except RuntimeError as e:
        print(f"  WARN: episodes failed for sub {submission_id}: {e}")
        return []
    episode_ids = []
    for line in out.split("\n"):
        if not line or line.startswith("id") or line.startswith("Use"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            ep_id = int(parts[0].strip())
            # Only PUBLIC episodes
            if "EPISODE_TYPE_PUBLIC" in line:
                episode_ids.append(ep_id)
        except (ValueError, IndexError):
            continue
    return episode_ids


def collect_existing_episode_ids() -> set[int]:
    """Collect all episode IDs we already have locally."""
    existing = set()
    for path in EPISODES_DIR.rglob("*.json"):
        for m in re.findall(r'\d{8,}', path.name):
            try:
                existing.add(int(m))
            except ValueError:
                pass
    return existing


def download_replay(episode_id: int, out_dir: Path, sleep: float = 0.5) -> bool:
    """Download replay JSON for an episode. Returns True on success."""
    out_path = out_dir / f"episode-{episode_id}-replay.json"
    if out_path.exists():
        return True
    try:
        run([KAGGLE, "competitions", "replay", str(episode_id), "-p", str(out_dir), "-q"], sleep_after=sleep)
        return out_path.exists()
    except RuntimeError as e:
        print(f"  WARN: replay download failed for {episode_id}: {e}")
        return False


def check_replay_crustle(path: Path) -> tuple[bool, int]:
    """Check if replay has a Crustle deck. Returns (has_crustle, winner_player_idx)."""
    try:
        with open(path) as f:
            data = json.load(f)
        steps = data.get("steps", [])
        rewards = data.get("rewards", [])
        winner = next((i for i, r in enumerate(rewards) if r > 0), -1)
        has_crustle = False
        if len(steps) > 1:
            s1 = steps[1]
            for pi in range(min(2, len(s1))):
                deck = s1[pi].get("action", [])
                if len(deck) == 60:
                    if CRUSTLE_IDS.intersection(deck):
                        has_crustle = True
        return has_crustle, winner
    except Exception:
        return False, -1


def main():
    # Disable output buffering so progress is visible when run in background
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Harvest episodes from all teams")
    parser.add_argument("--max-teams", type=int, default=20, help="Max teams to process (default: 20)")
    parser.add_argument("--min-score", type=float, default=900.0, help="Min leaderboard score (default: 900)")
    parser.add_argument("--dry-run", action="store_true", help="Don't download, just print plan")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between API calls (default: 0.5)")
    args = parser.parse_args()

    ALL_TEAMS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Fetching leaderboard (min_score={args.min_score}, max_teams={args.max_teams})...")
    teams = get_leaderboard(min_score=args.min_score, max_teams=args.max_teams)
    print(f"  Found {len(teams)} teams with score >= {args.min_score}")
    for t in teams[:5]:
        print(f"    {t['teamName']} (id={t['teamId']}, score={t['score']})")
    if len(teams) > 5:
        print(f"    ... and {len(teams)-5} more")

    print(f"\n[2/5] Collecting existing episode IDs for deduplication...")
    existing = collect_existing_episode_ids()
    print(f"  Already have {len(existing)} episode IDs locally")

    print(f"\n[3/5] Getting best submissions and episode lists...")
    plan: list[tuple[dict, int, list[int]]] = []  # (team, sub_id, [ep_ids])
    total_new = 0
    for team in teams:
        sub_id = get_best_submission(team["teamId"], sleep=args.sleep)
        if sub_id is None:
            print(f"  SKIP {team['teamName']}: no submission found")
            continue
        ep_ids = get_episodes(sub_id, sleep=args.sleep)
        new_ep_ids = [e for e in ep_ids if e not in existing]
        plan.append((team, sub_id, new_ep_ids))
        total_new += len(new_ep_ids)
        print(f"  {team['teamName']} (score={team['score']}): sub={sub_id}, "
              f"{len(ep_ids)} episodes, {len(new_ep_ids)} new")

    print(f"\n  Plan: {total_new} new episodes to download across {len(plan)} teams")

    if args.dry_run:
        print("\n[DRY RUN] Would download the episodes listed above. Exiting.")
        return

    print(f"\n[4/5] Downloading {total_new} new replay files...")
    downloaded = 0
    crustle_count = 0
    winner_main_decisions_est = 0

    for team, sub_id, new_ep_ids in plan:
        for ep_id in new_ep_ids:
            if ep_id in existing:
                continue
            ok = download_replay(ep_id, ALL_TEAMS_DIR, sleep=args.sleep)
            if ok:
                downloaded += 1
                existing.add(ep_id)
                replay_path = ALL_TEAMS_DIR / f"episode-{ep_id}-replay.json"
                has_crustle, winner = check_replay_crustle(replay_path)
                if has_crustle:
                    crustle_count += 1
                # Estimate winner decisions: ~20 MAIN decisions per game
                if winner >= 0:
                    winner_main_decisions_est += 20
                print(f"  [{downloaded}/{total_new}] ep={ep_id} "
                      f"crustle={'YES' if has_crustle else 'no'} winner=player{winner}")

    print(f"\n[5/5] Summary:")
    print(f"  Downloaded: {downloaded} new replay files -> data/episodes/all_teams/")
    print(f"  Crustle-bearing games: {crustle_count} ({100*crustle_count/max(downloaded,1):.1f}%)")
    print(f"  Estimated winner-side MAIN decisions: ~{winner_main_decisions_est}")
    print(f"  Total episode IDs known: {len(existing)}")


if __name__ == "__main__":
    main()
