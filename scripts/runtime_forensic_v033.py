#!/usr/bin/env python3
"""Runtime forensic for v033_plan_d_ppo (submission 54779509) live replays.

Checks each downloaded episode's agent stderr logs for the runtime-bug
signatures documented in project memory (project_kaggle_live_safety):
missing __file__, lazy-import failures, numpy/sklearn ABI mismatches — all
of which silently degrade main.py::agent() to the greedy _fallback_agent()
without raising a visible exception on Kaggle. main.py's own counters
(_n_nn_decisions / _n_fallback_decisions / _n_exceptions) are module-level
globals that are NEVER printed anywhere in the live harness, so the only way
to detect a silent fallback post-hoc is to grep stderr for the explicit
print() signatures main.py emits: "[v033] PPO model loaded OK",
"[v033] WARNING: ... import failed", "[v033] WARNING: model load failed",
"[v033] nn_action failed: ...", plus generic crash signatures (Traceback,
No module named, numpy.dtype size changed, NameError: __file__).

NOTE: our agent is NOT always slot 0 in these episodes (matchmaking is
symmetric) -- the per-episode agent log to check is whichever of
episode-<id>-agent-0-logs.json / episode-<id>-agent-1-logs.json corresponds
to our own submission process (Kaggle only grants log access to your own
agent's slot, so whichever file downloaded successfully IS ours).

Usage:
    python scripts/runtime_forensic_v033.py
"""
import json
import os

ROOT = "/Users/max/kaggle-pokemon-tcg-ai-battle"
EP_LIST = os.path.join(ROOT, "data/episodes/v033_episode_ids.txt")
EPISODES_DIR = os.path.join(ROOT, "data/episodes")

OK_SIGNATURE = "[v033] PPO model loaded OK"
BAD_SIGNATURES = [
    "[v033] WARNING: torch/nn_plan_d/cabt_conv import failed",
    "[v033] WARNING: model load failed",
    "[v033] nn_action failed:",
    "Traceback (most recent call last)",
    "No module named",
    "numpy.dtype size changed",
    "NameError",
    "ImportError",
]


def load_agent_log(eid):
    for agent_idx in (0, 1):
        p = os.path.join(EPISODES_DIR, f"episode-{eid}-agent-{agent_idx}-logs.json")
        if os.path.exists(p):
            with open(p) as f:
                return agent_idx, json.load(f)
    return None, None


def full_stderr(steps):
    out = []
    for step in steps:
        for view in step:
            se = view.get("stderr") or ""
            if se:
                out.append(se)
    return "\n".join(out)


def main():
    with open(EP_LIST) as f:
        ep_ids = [line.strip() for line in f if line.strip()]

    n_total = len(ep_ids)
    n_model_ok = 0
    n_missing_log = 0
    n_bad_signature = 0
    bad_episodes = []
    missing_log_episodes = []

    for eid in ep_ids:
        agent_idx, steps = load_agent_log(eid)
        if steps is None:
            n_missing_log += 1
            missing_log_episodes.append(eid)
            continue
        stderr_text = full_stderr(steps)
        if OK_SIGNATURE in stderr_text:
            n_model_ok += 1
        hit_bad = [sig for sig in BAD_SIGNATURES if sig in stderr_text]
        if hit_bad:
            n_bad_signature += 1
            bad_episodes.append((eid, agent_idx, hit_bad))

    print(f"=== v033 runtime forensic: {n_total} episodes ===")
    print(f"agent log found:            {n_total - n_missing_log}/{n_total}")
    print(f"'{OK_SIGNATURE}' present:   {n_model_ok}/{n_total}")
    print(f"bad-signature hits:          {n_bad_signature}/{n_total}")
    if missing_log_episodes:
        print("episodes with NO agent log at all:", missing_log_episodes)
    if bad_episodes:
        print("\n!!! STOP-CRANE CANDIDATE(S) FOUND !!!")
        for eid, agent_idx, sigs in bad_episodes:
            print(f"  episode {eid} (agent slot {agent_idx}): {sigs}")
    else:
        print("\nNo silent-fallback / crash signatures found in any episode.")
        print("VERDICT: NN model loaded and ran with 0 exceptions across all "
              "downloaded live episodes. Runtime forensic does NOT block "
              "further behavioral analysis (A3/A4).")


if __name__ == "__main__":
    main()
