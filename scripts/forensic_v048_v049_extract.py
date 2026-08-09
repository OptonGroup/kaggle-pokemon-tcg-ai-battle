"""
Extract win/loss + basic match summary for each downloaded replay in
out/forensic_v048_v049/{v048,v049}/replays/, determine our agent_index
(TeamNames vs opponent), and write a per-episode summary JSON + CSV index.

Usage: .venv/bin/python scripts/forensic_v048_v049_extract.py
"""
import json
import csv
from pathlib import Path

BASE = Path("out/forensic_v048_v049")
OUR_TEAM_HINTS = ("maxim simakov",)


def load_replay(path: Path):
    with open(path) as f:
        return json.load(f)


def summarize(replay: dict, ep_id: str):
    info = replay.get("info", {})
    team_names = info.get("TeamNames") or info.get("teamNames") or []
    rewards = replay.get("rewards")
    if rewards is None:
        steps = replay.get("steps", [])
        if steps:
            last = steps[-1]
            rewards = [s.get("reward") for s in last] if isinstance(last, list) else None

    # figure out which index is "us" -- team name won't literally say archaludon,
    # so fall back to: the submission that isn't a generic/public bot name.
    # Best-effort: record both team names, let human/next-agent map index->us.
    our_idx = None
    for i, name in enumerate(team_names):
        if name and any(h in name.lower() for h in OUR_TEAM_HINTS):
            our_idx = i
            break

    steps = replay.get("steps", [])
    num_steps = len(steps)

    result = {
        "episode_id": ep_id,
        "team_names": team_names,
        "rewards": rewards,
        "our_idx_guess": our_idx,
        "num_steps": num_steps,
    }
    if rewards and len(rewards) == 2:
        r0, r1 = rewards[0], rewards[1]
        result["winner_idx"] = 0 if (r0 is not None and r1 is not None and r0 > r1) else (
            1 if (r0 is not None and r1 is not None and r1 > r0) else None
        )
        if result["winner_idx"] is not None and our_idx is not None:
            result["our_result"] = "WIN" if result["winner_idx"] == our_idx else "LOSS"
        elif r0 == r1:
            result["our_result"] = "DRAW"
        else:
            result["our_result"] = "UNKNOWN"
    return result


def main():
    for version in ("v048", "v049"):
        replay_dir = BASE / version / "replays"
        out_rows = []
        for rp in sorted(replay_dir.glob("*.json")):
            ep_id = rp.stem.replace("episode-", "").replace("-replay", "")
            try:
                replay = load_replay(rp)
            except Exception as e:
                out_rows.append({"episode_id": ep_id, "error": str(e)})
                continue
            summ = summarize(replay, ep_id)
            out_rows.append(summ)

        summary_path = BASE / version / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(out_rows, f, indent=2)

        csv_path = BASE / version / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode_id", "our_idx_guess", "winner_idx", "our_result", "num_steps", "team_names", "rewards"])
            for row in out_rows:
                w.writerow([
                    row.get("episode_id"),
                    row.get("our_idx_guess"),
                    row.get("winner_idx"),
                    row.get("our_result"),
                    row.get("num_steps"),
                    row.get("team_names"),
                    row.get("rewards"),
                ])
        n_win = sum(1 for r in out_rows if r.get("our_result") == "WIN")
        n_loss = sum(1 for r in out_rows if r.get("our_result") == "LOSS")
        n_unk = sum(1 for r in out_rows if r.get("our_result") not in ("WIN", "LOSS"))
        print(f"{version}: {len(out_rows)} episodes, WIN={n_win} LOSS={n_loss} other/unknown={n_unk}")


if __name__ == "__main__":
    main()
