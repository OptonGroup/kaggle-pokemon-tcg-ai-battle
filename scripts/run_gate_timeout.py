#!/usr/bin/env python3
"""
Safe gate runner with hard timeout.

Usage:
    python scripts/run_gate_timeout.py <candidate> <opponent> <n_games> <seed> <out_json> [timeout_s]

Default timeout: 60s * n_games + 300s (startup overhead)
"""
import sys
import os
import subprocess
import json
import re
import datetime

def main():
    args = sys.argv[1:]
    if len(args) < 5:
        print("Usage: run_gate_timeout.py <candidate> <opponent> <n_games> <seed> <out_json> [timeout_s]")
        sys.exit(1)

    candidate = args[0]
    opponent = args[1]
    n_games = int(args[2])
    seed = int(args[3])
    out_json = args[4]
    timeout_s = int(args[5]) if len(args) >= 6 else (n_games * 60 + 300)

    image_name = "pokemon-tcg-sim"

    cmd = [
        "docker", "run", "--rm",
        "--platform", "linux/amd64",
        "--memory=4g",
        image_name,
        "python", "src/eval/gate_runner.py",
        candidate, opponent, str(n_games), str(seed),
    ]

    print(f"=== Mirror Gate: {candidate} vs {opponent} ===")
    print(f"  n_games={n_games}  seed={seed}")
    print(f"  timeout_s={timeout_s}")
    print(f"  out_json={out_json}")
    print(f"  cmd={' '.join(cmd)}")
    print()

    sys.stdout.flush()

    exit_code = 0
    timed_out = False
    log_output = ""

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            text=True,
        )
        log_output = result.stdout
        exit_code = result.returncode
        print(log_output)
    except subprocess.TimeoutExpired as e:
        timed_out = True
        log_output = e.stdout or ""
        if log_output:
            print(log_output)
        print(f"\nERROR: Docker run timed out after {timeout_s}s", flush=True)
        exit_code = 124
    except Exception as e:
        print(f"\nERROR: Unexpected exception: {e}", flush=True)
        exit_code = 1

    sys.stdout.flush()

    # Parse results
    def parse_int(pattern, text):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else None

    def parse_float(pattern, text):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    wins_A = parse_int(r"Candidate wins\s*:\s*(\d+)", log_output)
    wins_B = parse_int(r"Champion wins\s*:\s*(\d+)", log_output)
    draws = parse_int(r"Draws\s*:\s*(\d+)", log_output)
    winrate = parse_float(r"Winrate \(excl\. draws\):\s*([0-9.]+)", log_output)
    wilson_lo = parse_float(r"Wilson 95% CI\s*:\s*\[([0-9.]+)", log_output)
    wilson_hi = parse_float(r"Wilson 95% CI\s*:.*?,\s*([0-9.]+)\]", log_output)
    m_verdict = re.search(r"VERDICT:\s*(PASS|FAIL)", log_output)
    verdict = m_verdict.group(1) if m_verdict else ("TIMEOUT" if timed_out else "UNKNOWN")

    data = {
        "candidate": candidate,
        "opponent": opponent,
        "n_games": n_games,
        "seed": seed,
        "wins_A": wins_A,
        "wins_B": wins_B,
        "draws": draws,
        "winrate_A": winrate,
        "wilson_lo": wilson_lo,
        "wilson_hi": wilson_hi,
        "verdict": verdict,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

    print()
    print(f"=== Parsed Results ===")
    print(f"  wins_A={wins_A}  wins_B={wins_B}  draws={draws}")
    print(f"  winrate_A={winrate}  wilson_lo={wilson_lo}  wilson_hi={wilson_hi}")
    print(f"  verdict={verdict}  exit_code={exit_code}  timed_out={timed_out}")
    print(f"  Written to {out_json}")
    sys.stdout.flush()

    sys.exit(exit_code if not timed_out else 124)


if __name__ == "__main__":
    main()
