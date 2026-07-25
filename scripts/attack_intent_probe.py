#!/usr/bin/env python3
"""attack_intent_probe.py — cheap, robust resolution of the sequencing confound that
scripts/forward_sim_divergence.py hit engine-protocol friction on.

The pilot's own design (submissions/*/main.py _ko_guard docstring ~852-866) is
"develop first, attack last": non-game-winning KOs are NOT front-run by the KO guard;
the deterministic SCORER sequences attacks BELOW development, so the attack is taken on a
LATER decision once development is exhausted. So at an attack-divergence (expert attacked,
pilot developed), the pilot INTENDS to attack this turn iff its scorer already ranks the
attack option positively and above 'end turn' — even though it scored a development option
higher right now. This reads that intent straight off the pilot's own [alt] score trace,
with NO engine rollout (hence no hidden-info placeholders, no multi-select protocol issues,
no fake-data branches — the failure modes that made the full forward-sim unreliable).

Classification per attack-divergence (expert's chosen option is an ATTACK, pilot chose
something else, and the pilot reached the SCORER so a full [alt] ranking exists):
  intends_attack  : attack_score > 0 AND attack_score > end_turn_score
                    -> the attack is a live, positively-scored option ranked above passing;
                       consistent with "develop now, attack later this turn" (benign).
  real_miss       : attack_score <= 0 OR attack_score <= end_turn_score (incl. attack
                    scored negative, or the pilot would rather end the turn) -> the pilot
                    will not take this attack -> genuine, non-sequencing miss.
  guard/other     : decided by a guard (no scorer ranking) or attack option unscored.

Caveat: a scorer-intent proxy, not a full state rollout — it cannot prove the attack option
survives to a later decision (e.g. the attacker could be forced out first). But it is
directly the pilot's own ranked preference on the exact board, and is not contaminated by
reconstructed hidden state. Reported alongside the confirmed-waste floor from auditor C.

Usage: .venv/bin/python scripts/attack_intent_probe.py [--limit-eps N]
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
MAIN_REPO = Path("/Users/max/kaggle-pokemon-tcg-ai-battle")
sys.path.insert(0, str(WORKTREE / "src"))

import cg_native, cg_native.api, cg_native.game, cg_native.sim, cg_native.utils  # noqa: E402
for _n, _m in [("cg", cg_native), ("cg.api", cg_native.api), ("cg.game", cg_native.game),
               ("cg.sim", cg_native.sim), ("cg.utils", cg_native.utils)]:
    sys.modules.setdefault(_n, _m)
from replays.parse import parse_replay  # noqa: E402

OPT_ATTACK, OPT_END = 13, 14
CTX_MAIN = 0
# [alt] line: "t{turn} ctx{ctx}: #{i} s={score} '{reason}'; #{j} s=... ; ..."
ALT_TOKEN = re.compile(r"#(\d+)\s+s=(-?\d+)")


def load_pilot(sub_name: str):
    sub = (WORKTREE / "submissions" / sub_name).resolve()
    sys.path.insert(0, str(sub))
    old = os.getcwd()
    try:
        os.chdir(sub)
        spec = importlib.util.spec_from_file_location(f"_p_{sub_name}", str(sub / "main.py"))
        m = importlib.util.module_from_spec(spec)
        m.__file__ = str(sub / "main.py")
        spec.loader.exec_module(m)
    finally:
        os.chdir(old)
    return m


def agent_with_trace(pilot, obs):
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        choice = pilot.agent(obs)
    return choice, buf.getvalue()


def parse_alt_scores(trace: str):
    """Return {option_index: score} from the LAST [alt] line, or {} if none."""
    alt = None
    for line in trace.splitlines():
        s = line.strip()
        if s.startswith("[alt] "):
            alt = s
    if alt is None:
        return {}
    return {int(i): int(sc) for i, sc in ALT_TOKEN.findall(alt)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="v074_fullmetal")
    ap.add_argument("--day", default="2026-07-20")
    ap.add_argument("--daily-root", default=str(MAIN_REPO / "data" / "episodes" / "official"))
    ap.add_argument("--ref-decks", default=str(MAIN_REPO / "data" / "deck_scan" / "top10_decks.json"))
    ap.add_argument("--targets", nargs="+", default=["jiatu.l", "bono"])
    ap.add_argument("--min-overlap", type=int, default=54)
    ap.add_argument("--limit-eps", type=int, default=0)
    ap.add_argument("--out", default=str(WORKTREE / "out" / "v074_fidelity" / "attack_intent.json"))
    args = ap.parse_args()

    ref = json.loads(Path(args.ref_decks).read_text())
    targets = set(args.targets)
    pilot = load_pilot(args.candidate)

    def overlap(a, b):
        return sum((Counter(a) & Counter(b)).values())

    def true_decks(data):
        st = data.get("steps", [])
        out = {}
        if len(st) >= 2:
            for pi in range(min(2, len(st[1]))):
                a = st[1][pi].get("action", [])
                if len(a) == 60 and all(isinstance(x, int) for x in a):
                    out[pi] = a
        return out

    day_dir = Path(args.daily_root) / args.day
    files = sorted(day_dir.glob("*.json"))
    picked = []
    for fp in files:
        try:
            if any(t.encode() in fp.read_bytes() for t in targets):
                picked.append(fp)
        except Exception:
            pass
        if args.limit_eps and len(picked) >= args.limit_eps:
            break

    tally = Counter()
    score_hist = {"intends_attack": [], "real_miss": []}
    samples = []
    for fp in picked:
        try:
            data = json.loads(fp.read_bytes())
        except Exception:
            continue
        names = [a.get("Name") for a in data.get("info", {}).get("Agents", [])]
        td = true_decks(data)
        experts = [pi for pi, nm in enumerate(names)
                   if nm in targets and nm in ref and td.get(pi) is not None
                   and overlap(td[pi], ref[nm]) >= args.min_overlap]
        if not experts:
            continue
        try:
            pairs = parse_replay(fp)
        except Exception:
            continue
        for pi in experts:
            for p in pairs:
                if p["player_index"] != pi:
                    continue
                obs = p["obs_dict"]
                sel = obs.get("select") or {}
                if sel.get("context") != CTX_MAIN:
                    continue
                opts = sel.get("option") or []
                expert = list(p["chosen_option_indices"])
                ei = expert[0] if expert else None
                if not (isinstance(ei, int) and 0 <= ei < len(opts) and opts[ei].get("type") == OPT_ATTACK):
                    continue  # only expert-attacked MAIN decisions
                pilot.agent({"select": None})
                our, trace = agent_with_trace(pilot, obs)
                if our is not None and set(our) == set(expert):
                    tally["agreed_attacked"] += 1
                    continue
                # divergence: expert attacked, pilot did not
                scores = parse_alt_scores(trace)
                end_idx = next((i for i, o in enumerate(opts) if o.get("type") == OPT_END), None)
                if ei not in scores:
                    tally["guard_or_unscored"] += 1
                    continue
                atk_s = scores[ei]
                end_s = scores.get(end_idx, 0) if end_idx is not None else 0
                if atk_s > 0 and atk_s > end_s:
                    cls = "intends_attack"
                else:
                    cls = "real_miss"
                tally[cls] += 1
                score_hist[cls].append(atk_s)
                if len(samples) < 60:
                    samples.append({"episodeId": p.get("episodeId"), "turn": obs.get("current", {}).get("turn"),
                                    "won": p.get("won"), "attack_idx": ei, "attack_score": atk_s,
                                    "end_score": end_s, "our": list(our) if our else our,
                                    "our_score": scores.get(our[0]) if our else None, "class": cls})

    div = tally["intends_attack"] + tally["real_miss"]
    report = {
        "candidate": args.candidate, "day": args.day,
        "expert_attacked_MAIN_decisions": {
            "pilot_agreed_attacked": tally["agreed_attacked"],
            "divergences_scored": div,
            "divergences_guard_or_unscored": tally["guard_or_unscored"],
        },
        "of_scored_divergences": {
            "intends_attack_later": tally["intends_attack"],
            "intends_attack_pct": round(tally["intends_attack"] / div, 4) if div else None,
            "real_miss": tally["real_miss"],
            "real_miss_pct": round(tally["real_miss"] / div, 4) if div else None,
        },
        "attack_score_median": {
            k: (sorted(v)[len(v) // 2] if v else None) for k, v in score_hist.items()
        },
        "samples": samples,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["expert_attacked_MAIN_decisions"], indent=2))
    print(json.dumps(report["of_scored_divergences"], indent=2))
    print("attack_score median:", report["attack_score_median"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
