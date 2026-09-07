#!/usr/bin/env python3
"""Aggregate out/all_versions_wr/archetypes.csv into matchup tables.

    .venv/bin/python scripts/all_versions_report.py            # everything
    .venv/bin/python scripts/all_versions_report.py --era      # split by era only
"""
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify_opponent_archetype import classify

ROOT = Path(__file__).resolve().parent.parent
ARCHE = ROOT / "out" / "all_versions_wr" / "archetypes.csv"

# Пилотные линии (эры) — по родословной агента, а не по дате.
ERA = [
    ("crustle_early", {"v001_random", "v002_lucario", "v006_crustle", "v007_crustle",
                       "v009a_crustle_resub", "v007_resub", "v010a_iono.zip",
                       "v012_iono_heur", ".zip"}),
    ("exit_cardemb", {"v015_exit_pilot", "v023_crustle_techD", "v024_cardemb_rl",
                      "v026_exit_gbdt", "v027_retreat_fix", "v028_mcts_fix",
                      "v029_pure_numpy_value", "v030_final"}),
    ("ppo_clones", {"v033_plan_d_ppo", "v034_romanrozen_clone", "v035_makthanithin",
                    "v037_plan_d_ppo_hidden", "v038_plan_d_ppo_poolfix", "v039_ft",
                    "v041_extpool", "v043_alakazam_mirror_fix"}),
    ("archaludon", {"v036_archaludon_patched", "v044_archaludon_abomasnow",
                    "v045_archaludon_judge_fix", "v046_final", "v046_judgefix",
                    "v047_deckbelief", "v048_deckbelief_judgefix",
                    "v049_bench_sticky_guard", "v050_p0p1"}),
    ("grimmsnarl", {"v018_grimmsnarl", "v061_grimmsnarl_heuristic",
                    "v062_grimmsnarl_playbook", "v063_forensic86", "v064_starmie_scaling",
                    "v065_defensive_swap", "v070_grimmsnarl_hybrid", "v072_mirror_gauntlet",
                    "v073_rockinn_damage", "v074_fullmetal", "v075_promo", "v076_final"}),
    ("scout", {"submission_search_v2", "scout_deep", "scout_deeper", "scout_evo",
               "scout_deeper_final_A", "scout_deeper_final_B", "scout_deeper_p0p1"}),
    ("bc2b", {"v080_bc2b", "v082_bc2b", "v083_bc2b", "v085_bc2b", "v086_bc2b",
              "v087_bc2b", "v088_bc2b", "v089_bc2b"}),
]
ERA_OF = {v: name for name, vs in ERA for v in vs}
ERA_ORDER = [name for name, _ in ERA]


def ts(s):
    """Seconds from an episode create_time string; 0 if it is unparseable."""
    s = (s or "").replace("T", " ").split(".")[0]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d * 100, (c + m) / d * 100)


def wl(rows):
    w = sum(1 for r in rows if float(r["our_reward"] or 0) > 0)
    l = sum(1 for r in rows if float(r["our_reward"] or 0) < 0)
    return w, l


def table(title, groups, min_n=1):
    print(f"\n### {title}")
    print(f"{'ключ':28} {'W':>5}{'L':>5} {'N':>5} {'WR%':>7}  95% CI")
    out = []
    for key, rows in groups.items():
        w, l = wl(rows)
        n = w + l
        if n < min_n:
            continue
        out.append((n, key, w, l, 100 * w / n if n else 0, wilson(w, n)))
    out.sort(reverse=True)
    for n, key, w, l, wr, ci in out:
        flag = "  ⚠шум" if n < 30 else ""
        print(f"{key:28.28} {w:5}{l:5} {n:5} {wr:7.1f}  [{ci[0]:4.0f},{ci[1]:4.0f}]{flag}")
    return out


def propagate(rows, index_rows):
    """Fill archetypes for games whose replay we never fetched, using the opponent
    TEAM's archetype observed in a fetched game (nearest in time).

    Justified because the archetype is a property of the opponent's decklist, and a
    team's list is mostly stable: over the whole history only ~7% of teams were ever
    seen on more than one archetype. Rows filled this way are marked source=team so
    the caller can drop them and check the conclusion on measured rows only.
    """
    obs = defaultdict(list)          # team_id -> [(create_time, archetype, key_cards)]
    for r in rows:
        if r.get("opp_team_id"):
            obs[r["opp_team_id"]].append(
                (r["create_time"], r["archetype"], r.get("opp_key_cards", "")))
    have = {r["episode_id"] for r in rows}
    out = [{**r, "source": "replay"} for r in rows]
    filled = 0
    for r in index_rows:
        if r["episode_id"] in have:
            continue
        cand = obs.get(r.get("opp_team_id"))
        if not cand:
            continue
        best = min(cand, key=lambda c: abs(ts(c[0]) - ts(r["create_time"])))
        out.append({**r, "archetype": best[1], "opp_key_cards": best[2],
                    "source": "team"})
        filled += 1
    return out, filled


def scan_fallback(rows, index_rows):
    """Third source of opponent archetypes, free: data/deck_scan/decks.jsonl holds
    60-card lists keyed by TEAM NAME, harvested earlier from public replays. Games
    against a team that appears there but whose own replay we never fetched get that
    list's archetype (marked source=scan).

    Weaker than a fetched replay — the scanned list is a snapshot from one point in
    time, so a team that later changed decks is labelled by its old list. Used only
    where nothing better exists.
    """
    scan = ROOT / "data" / "deck_scan" / "decks.jsonl"
    if not scan.exists():
        return rows, 0
    id2name = {}
    cards = ROOT / "data" / "EN_Card_Data.csv"
    if cards.exists():
        for c in csv.DictReader(open(cards)):
            key = c.get("Card ID") or c.get("id")
            nm = c.get("Name") or c.get("name")
            if key and nm:
                id2name[int(key)] = nm
    team_deck = {}
    for line in open(scan):
        d = json.loads(line)
        for t, deck in zip(d.get("teams", []), d.get("decks", [])):
            team_deck.setdefault(t, deck)
    have = {r["episode_id"] for r in rows}
    out = list(rows)
    filled = 0
    for r in index_rows:
        if r["episode_id"] in have:
            continue
        deck = team_deck.get(r.get("opp_team_name"))
        if not deck:
            continue
        names = [id2name.get(i, "") for i in deck]
        arche = classify(set(deck), names)
        if not arche:
            continue
        top = ", ".join(f"{n}x{c}" for n, c in Counter(names).most_common(6))
        out.append({**r, "archetype": arche, "opp_key_cards": top, "source": "scan"})
        filled += 1
    return out, filled


def main():
    rows = [r for r in csv.DictReader(open(ARCHE))
            if r["archetype"] not in ("FETCH_FAILED", "NO_DECK_DATA", "")]
    total = list(csv.DictReader(open(ARCHE)))
    index_rows = list(csv.DictReader(open(ARCHE.parent / "index.csv")))
    n_replay = len(rows)
    rows, filled = propagate(rows, index_rows)
    rows, scanned = scan_fallback(rows, index_rows)
    print(f"классифицировано {len(rows)} / {len(index_rows)} партий: "
          f"{n_replay} по своему реплею, {filled} по team_id того же оппонента, "
          f"{scanned} по локальному скану дек (скачано реплеев {len(total)})")

    by_arche = defaultdict(list)
    for r in rows:
        by_arche[r["archetype"]].append(r)
    table("WR по архетипу оппонента — ВСЯ ИСТОРИЯ (105 сабмитов)", by_arche)

    by_era = defaultdict(list)
    for r in rows:
        by_era[ERA_OF.get(r["version"], "other")].append(r)
    table("WR по эре пилота", by_era)

    # era × archetype
    print("\n### Матрица эра × архетип (WR%, N)")
    arches = [k for k, _ in sorted(by_arche.items(), key=lambda kv: -len(kv[1]))][:12]
    eras = [e for e in ERA_ORDER if e in by_era] + (["other"] if "other" in by_era else [])
    hdr = f"{'эра':16}" + "".join(f"{a[:11]:>13}" for a in arches)
    print(hdr)
    for e in eras:
        cells = []
        for a in arches:
            sub = [r for r in by_era[e] if r["archetype"] == a]
            w, l = wl(sub)
            n = w + l
            cells.append(f"{100*w/n:.0f}% ({n})" if n else "-")
        print(f"{e:16}" + "".join(f"{c:>13}" for c in cells))

    by_ver = defaultdict(list)
    for r in rows:
        by_ver[r["version"]].append(r)
    table("WR по версии (все копии сабмита слиты)", by_ver)


if __name__ == "__main__":
    main()
