"""
Build a per-episode forensic dossier combining:
  - win/loss result (from summary.json)
  - our agent's [dbg] stderr log lines (from logs/)
  - matchup-detection trace (first detection + any label changes)
  - a compact per-episode text block, concatenated into one markdown file per version
  - a top-level index sorted LOSS-first (most interesting for bug hunting)

Output: out/forensic_v048_v049/{version}/DOSSIER.md
        out/forensic_v048_v049/{version}/dossier_index.json

Usage: .venv/bin/python scripts/forensic_v048_v049_build_dossier.py
"""
import json
import re
from pathlib import Path

BASE = Path("out/forensic_v048_v049")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def extract_dbg_lines(log_data):
    """log_data: list of steps, each step is a list of per-agent dicts with stdout/stderr."""
    lines = []
    for step_idx, step in enumerate(log_data):
        if not step:
            continue
        entry = step[0] if isinstance(step, list) else step
        stderr = entry.get("stderr", "") or ""
        for line in stderr.splitlines():
            if line.strip():
                lines.append((step_idx, line.strip()))
    return lines


def build_version(version):
    summary = load_json(BASE / version / "summary.json")
    logs_dir = BASE / version / "logs"

    rows = []
    for ep in summary:
        ep_id = ep["episode_id"]
        idx = ep.get("our_idx_guess")
        result = ep.get("our_result", "UNKNOWN")
        log_path = logs_dir / f"episode-{ep_id}-agent-{idx}-logs.json"
        dbg_lines = []
        if log_path.exists():
            try:
                log_data = load_json(log_path)
                dbg_lines = extract_dbg_lines(log_data)
            except Exception as e:
                dbg_lines = [(-1, f"[ERROR loading log: {e}]")]

        # matchup trace: lines with 'matchup=' or 'OPPONENT DECK'
        matchup_events = [l for _, l in dbg_lines if "OPPONENT DECK" in l or "matchup=" in l]
        # dedup consecutive identical matchup labels to spot flip-flopping
        matchup_labels = []
        for _, l in dbg_lines:
            m = re.search(r"matchup=(\S+)", l)
            if m:
                matchup_labels.append(m.group(1))
        label_changes = []
        prev = None
        for lab in matchup_labels:
            if lab != prev:
                label_changes.append(lab)
                prev = lab

        rows.append({
            "episode_id": ep_id,
            "our_idx": idx,
            "opponent_team": [t for i, t in enumerate(ep["team_names"]) if i != idx] if idx is not None else ep["team_names"],
            "result": result,
            "num_steps": ep.get("num_steps"),
            "rewards": ep.get("rewards"),
            "num_dbg_lines": len(dbg_lines),
            "matchup_label_sequence": label_changes,
            "flip_flopped": len(label_changes) > 2,
            "log_path": str(log_path) if log_path.exists() else None,
        })

    # sort: losses first, then by flip-flop, then episode id
    rows.sort(key=lambda r: (r["result"] != "LOSS", not r["flip_flopped"], r["episode_id"]))

    index_path = BASE / version / "dossier_index.json"
    with open(index_path, "w") as f:
        json.dump(rows, f, indent=2)

    # build markdown dossier with full dbg dumps, losses first
    md_path = BASE / version / "DOSSIER.md"
    n_win = sum(1 for r in rows if r["result"] == "WIN")
    n_loss = sum(1 for r in rows if r["result"] == "LOSS")
    with open(md_path, "w") as f:
        f.write(f"# {version} forensic dossier\n\n")
        f.write(f"Total episodes: {len(rows)}, WIN={n_win}, LOSS={n_loss} ({n_win/(n_win+n_loss)*100:.1f}% WR)\n\n")
        f.write("Sorted: LOSSES first, then flip-flopped matchup detection first, then episode id.\n\n")
        f.write("---\n\n")
        for r in rows:
            f.write(f"## Episode {r['episode_id']} — {r['result']} (steps={r['num_steps']}, our_idx={r['our_idx']})\n\n")
            f.write(f"Opponent: {r['opponent_team']}\n\n")
            f.write(f"Matchup label sequence: {r['matchup_label_sequence']}")
            if r["flip_flopped"]:
                f.write("  ⚠️ FLIP-FLOPPED")
            f.write("\n\n")
            log_path = r["log_path"]
            if log_path and Path(log_path).exists():
                log_data = load_json(log_path)
                dbg_lines = extract_dbg_lines(log_data)
                f.write("<details><summary>Full [dbg] log (" + str(len(dbg_lines)) + " lines)</summary>\n\n```\n")
                for step_idx, line in dbg_lines:
                    f.write(f"[step {step_idx}] {line}\n")
                f.write("```\n</details>\n\n")
            else:
                f.write("_(no log available)_\n\n")
            f.write("---\n\n")

    print(f"{version}: dossier written to {md_path} ({len(rows)} episodes, WR={n_win/(n_win+n_loss)*100:.1f}%)")
    print(f"  flip-flopped episodes: {sum(1 for r in rows if r['flip_flopped'])}")


def main():
    for version in ("v048", "v049"):
        build_version(version)


if __name__ == "__main__":
    main()
