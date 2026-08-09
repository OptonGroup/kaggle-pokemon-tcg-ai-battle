#!/usr/bin/env python3
"""Build per-episode result summaries for v044_resub / v046_judgefix from
downloaded replays + own-agent logs in data/episodes/.

Output:
  data/episodes/v044_resub_results.csv
  data/episodes/v046_judgefix_results.csv
Each row: episode_id, our_idx, opponent_team, result (win/loss/draw), reward,
n_steps, n_log_lines_stdout, n_log_lines_stderr, has_agent_output (whether the
agent's own log has any non-empty stdout/stderr across the game).
"""
import csv
import json
import sys
from pathlib import Path

EPISODES_DIR = Path("data/episodes")

SUBMISSIONS = {
    "v044_resub": (54822470, "/tmp/v044_eps.txt"),
    "v046_judgefix": (54822473, "/tmp/v046_eps.txt"),
}


def summarize(name: str, ep_id_file: str) -> None:
    ep_ids = [line.strip() for line in Path(ep_id_file).read_text().splitlines() if line.strip()]
    rows = []
    wins = losses = draws = 0
    for ep_id in ep_ids:
        replay_path = EPISODES_DIR / f"episode-{ep_id}-replay.json"
        if not replay_path.exists():
            print(f"  WARNING: missing replay for {ep_id}", file=sys.stderr)
            continue
        replay = json.loads(replay_path.read_text())
        team_names = replay["info"]["TeamNames"]
        if "Maxim Simakov" not in team_names:
            print(f"  WARNING: our team not found in episode {ep_id}", file=sys.stderr)
            continue
        our_idx = team_names.index("Maxim Simakov")
        opp_idx = 1 - our_idx
        opponent = team_names[opp_idx]
        rewards = replay["rewards"]
        our_reward = rewards[our_idx]
        opp_reward = rewards[opp_idx]
        if our_reward is None or opp_reward is None:
            result = "unknown"
        elif our_reward > opp_reward:
            result = "win"
            wins += 1
        elif our_reward < opp_reward:
            result = "loss"
            losses += 1
        else:
            result = "draw"
            draws += 1
        n_steps = len(replay.get("steps", []))

        log_path = EPISODES_DIR / f"episode-{ep_id}-agent-{our_idx}-logs.json"
        n_stdout = n_stderr = 0
        has_output = False
        if log_path.exists():
            log_data = json.loads(log_path.read_text())
            for step_entries in log_data:
                for entry in step_entries:
                    out = entry.get("stdout", "")
                    err = entry.get("stderr", "")
                    if out.strip():
                        n_stdout += 1
                        has_output = True
                    if err.strip():
                        n_stderr += 1
                        has_output = True

        rows.append({
            "episode_id": ep_id,
            "our_idx": our_idx,
            "opponent_team": opponent,
            "result": result,
            "our_reward": our_reward,
            "opp_reward": opp_reward,
            "n_steps": n_steps,
            "n_log_lines_stdout": n_stdout,
            "n_log_lines_stderr": n_stderr,
            "has_agent_output": has_output,
        })

    out_path = EPISODES_DIR / f"{name}_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    total = wins + losses + draws
    wr = wins / total if total else 0.0
    print(f"{name}: {len(rows)} episodes written to {out_path}")
    print(f"  W-L-D: {wins}-{losses}-{draws}  WR={wr:.1%}")


if __name__ == "__main__":
    for name, (sub_id, ep_file) in SUBMISSIONS.items():
        print(f"=== {name} (submission {sub_id}) ===")
        summarize(name, ep_file)
