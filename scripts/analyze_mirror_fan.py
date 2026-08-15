#!/usr/bin/env python3
"""analyze_mirror_fan.py — B1 analysis on top of scan_mirror_fan_field.py output
(mirror_games.csv). Computes:
  - fan-vs-noFan asymmetric WR (Fan side's win rate) with Wilson 95% CI
  - fan-vs-fan / noFan-vs-noFan counts (not informative for the comparison, just sized)
  - rating-controlled subset: |rating_a - rating_b| < 50 among asymmetric games
  - Fan adoption share among mirror_grimmsnarl deck-sides, per day (trend)
  - comparison against the 07-24 claim (75% w/ Fan vs 56% w/o)

Usage:
    .venv/bin/python scripts/analyze_mirror_fan.py <mirror_games_csv> <out_dir>
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def wilson_ci(wins: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def two_prop_ztest(x1: float, n1: float, x2: float, n2: float) -> tuple[float, float]:
    if n1 <= 0 or n2 <= 0:
        return (0.0, 1.0)
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(max(p_pool * (1 - p_pool), 1e-12) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    # two-sided normal p-value
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return (z, p_value)


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: analyze_mirror_fan.py <mirror_games_csv> <out_dir>")
    csv_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    total_games = len(rows)

    fan_vs_fan = 0
    nofan_vs_nofan = 0
    asym_rows = []
    for r in rows:
        fa, fb = int(r["has_fan_a"]), int(r["has_fan_b"])
        if fa == 1 and fb == 1:
            fan_vs_fan += 1
        elif fa == 0 and fb == 0:
            nofan_vs_nofan += 1
        else:
            asym_rows.append(r)

    n_asym = len(asym_rows)
    fan_wins = 0
    for r in asym_rows:
        fa = int(r["has_fan_a"])
        fan_side = "team_a" if fa == 1 else "team_b"
        if r["winner"] == fan_side:
            fan_wins += 1
    fan_wr = fan_wins / n_asym if n_asym else None
    fan_ci = wilson_ci(fan_wins, n_asym) if n_asym else (None, None)

    # Rating-controlled subset
    ctrl_rows = []
    for r in asym_rows:
        ra, rb = parse_float(r["rating_a"]), parse_float(r["rating_b"])
        if ra is None or rb is None:
            continue
        if abs(ra - rb) < 50:
            ctrl_rows.append(r)
    n_ctrl = len(ctrl_rows)
    ctrl_fan_wins = 0
    for r in ctrl_rows:
        fa = int(r["has_fan_a"])
        fan_side = "team_a" if fa == 1 else "team_b"
        if r["winner"] == fan_side:
            ctrl_fan_wins += 1
    ctrl_wr = ctrl_fan_wins / n_ctrl if n_ctrl else None
    ctrl_ci = wilson_ci(ctrl_fan_wins, n_ctrl) if n_ctrl else (None, None)

    # How many asymmetric rows have both ratings available at all (denominator context)
    n_asym_rated = sum(
        1 for r in asym_rows
        if parse_float(r["rating_a"]) is not None and parse_float(r["rating_b"]) is not None
    )

    # Adoption trend: among all mirror_grimmsnarl deck-SIDES seen (both a and b of every
    # mirror game, i.e. 2*total_games sides), what fraction carries Fan, overall and per day.
    # NOTE: this is restricted to sides observed in mirror-vs-mirror games (both players on
    # Grimmsnarl) — it does NOT cover Grimmsnarl decks that happened to face a non-mirror
    # opponent in this window (out of scope for this pass; would need a second full scan).
    side_total = Counter()
    side_fan = Counter()
    for r in rows:
        day = r["day"]
        side_total[day] += 2
        side_fan[day] += int(r["has_fan_a"]) + int(r["has_fan_b"])
    days_sorted = sorted(side_total)
    adoption_by_day = [
        {"day": d, "n_sides": side_total[d], "n_fan": side_fan[d],
         "fan_share": side_fan[d] / side_total[d] if side_total[d] else None}
        for d in days_sorted
    ]
    overall_sides = sum(side_total.values())
    overall_fan_sides = sum(side_fan.values())
    overall_fan_share = overall_fan_sides / overall_sides if overall_sides else None

    # Comparison vs 07-24 claim
    CLAIM_FAN_WR = 0.75
    CLAIM_NOFAN_WR = 0.56
    # The 07-24 claim's framing ("Fan holds 75% of mirror, vs 56% without") is ambiguous
    # about exactly what population it's WR-against; we compare our asymmetric Fan-side WR
    # against the two reference numbers directly.
    verdict_bits = []
    if n_asym < 150:
        verdict_bits.append(
            f"N_asym={n_asym} < 150 required threshold -> НЕДОСТАТОЧНО ДАННЫХ для WR-сравнения"
        )
    else:
        lo, hi = fan_ci
        in_range = lo <= CLAIM_FAN_WR <= hi
        verdict_bits.append(
            f"fresh asymmetric fan_wr={fan_wr:.3f} (95% CI [{lo:.3f},{hi:.3f}], N={n_asym}); "
            f"claim value 0.75 is {'INSIDE' if in_range else 'OUTSIDE'} the fresh 95% CI"
        )

    report = {
        "csv_path": str(csv_path),
        "total_mirror_games": total_games,
        "fan_vs_fan_games": fan_vs_fan,
        "nofan_vs_nofan_games": nofan_vs_nofan,
        "asymmetric_games_n": n_asym,
        "asymmetric_fan_side_wins": fan_wins,
        "asymmetric_fan_side_wr": fan_wr,
        "asymmetric_fan_side_wr_wilson95": fan_ci,
        "rating_controlled": {
            "delta_threshold": 50,
            "n_asym_with_both_ratings": n_asym_rated,
            "n_controlled": n_ctrl,
            "fan_side_wins": ctrl_fan_wins,
            "fan_side_wr": ctrl_wr,
            "fan_side_wr_wilson95": ctrl_ci,
        },
        "fan_adoption": {
            "overall_sides": overall_sides,
            "overall_fan_sides": overall_fan_sides,
            "overall_fan_share": overall_fan_share,
            "by_day": adoption_by_day,
            "caveat": "restricted to Grimmsnarl sides seen in mirror-vs-mirror games only",
        },
        "claim_20260724": {"fan_wr": CLAIM_FAN_WR, "nofan_wr": CLAIM_NOFAN_WR, "n": "unknown"},
        "verdict_notes": verdict_bits,
    }

    (out_dir / "b1_fan_wr_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
