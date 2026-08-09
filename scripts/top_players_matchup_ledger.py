#!/usr/bin/env python3
"""Матчап-леджер ТОП-игроков: те же метрики, что cross_version_matchup_ledger, но по чужим играм.

Вопрос, на который отвечает: если топы играют нашу же деку, как они выигрывают
матчапы, которые для нас структурно проигранные (Crustle-стена, abomasnow)?

Вход: out/top5_recon/<date>/index.csv (episodeId, team, side, result, opp_team)
      + out/top5_recon/<date>/episodes/<id>.json

Выход: out/top_players_ledger.csv (одна строка = одна игра топ-игрока) + сводки.

Usage:
    .venv/bin/python scripts/top_players_matchup_ledger.py [out/top5_recon/2026-07-23]
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify_opponent_archetype import find_initial_players, classify  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "out/top5_recon/2026-07-23"
CACHE = ROOT / "out/top_players_ledger.csv"


def build(src: Path):
    rows = list(csv.DictReader(open(src / "index.csv")))
    ep_dir = src / "episodes"
    out = []
    miss = 0
    for i, r in enumerate(rows):
        if i % 100 == 0:
            print(f"  ...{i}/{len(rows)}", file=sys.stderr, flush=True)
        path = ep_dir / f"{r['episodeId']}.json"
        if not path.exists():
            miss += 1
            continue
        try:
            replay = json.loads(path.read_text())
        except Exception:
            miss += 1
            continue
        players = find_initial_players(replay)
        if players is None:
            miss += 1
            continue
        side = int(r["side"])
        subj = players[side].get("deck", [])
        opp = players[1 - side].get("deck", [])
        if not subj or not opp:
            miss += 1
            continue
        out.append({
            "episode": r["episodeId"],
            "team": r["team"],
            "win": 1 if r["result"] == "W" else 0,
            "opp_team": r["opp_team"],
            "subj_arche": classify({c["id"] for c in subj},
                                   [c["name"] for c in subj]) or "UNKNOWN",
            "opp_arche": classify({c["id"] for c in opp},
                                  [c["name"] for c in opp]) or "UNKNOWN",
            "opp_has_crustle": int(any(c["id"] == 345 for c in opp)),
            "subj_deck": " ".join(str(c["id"]) for c in sorted(subj, key=lambda x: x["id"])),
        })
    print(f"[build] годных {len(out)}, пропущено {miss}", file=sys.stderr)
    with open(CACHE, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    return out


def wr(gs):
    if not gs:
        return "—"
    w = sum(g["win"] for g in gs)
    return f"{w}-{len(gs)-w} ({w/len(gs)*100:.0f}%)"


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = Path(pos[0]) if pos else DEFAULT_DIR
    if CACHE.exists() and "--rebuild" not in sys.argv:
        rows = [dict(r, win=int(r["win"]), opp_has_crustle=int(r["opp_has_crustle"]))
                for r in csv.DictReader(open(CACHE))]
        print(f"(кэш {CACHE}, {len(rows)} игр; --rebuild чтобы пересобрать)")
    else:
        rows = build(src)

    print("\n=== 1. Какую деку реально играют топы ===")
    for team, gs in sorted(defaultdict(list, {t: [r for r in rows if r["team"] == t]
                                              for t in {r["team"] for r in rows}}).items(),
                           key=lambda kv: -len(kv[1])):
        decks = Counter(g["subj_arche"] for g in gs)
        uniq = len({g["subj_deck"] for g in gs})
        print(f"  {team:<20} N={len(gs):<5} WR={wr(gs):<14} деки={dict(decks.most_common(3))}"
              f"  уник.60-карт.листов={uniq}")

    grim = [r for r in rows if r["subj_arche"] == "mirror_grimmsnarl"]
    print(f"\n=== 2. Только игры топов НАШЕЙ декой (Grimmsnarl): N={len(grim)}, WR={wr(grim)} ===")
    print(f"{'оппонент-архетип':<22}{'топы':>18}{'мы (леджер)':>18}")
    ours = {}
    lp = ROOT / "out/cross_version_ledger.csv"
    if lp.exists():
        our_rows = [dict(r, win=int(r["win"])) for r in csv.DictReader(open(lp))]
        for a in {r["arche"] for r in our_rows}:
            ours[a] = [r for r in our_rows if r["arche"] == a]
    by_opp = defaultdict(list)
    for g in grim:
        by_opp[g["opp_arche"]].append(g)
    for a, gs in sorted(by_opp.items(), key=lambda kv: -len(kv[1])):
        if len(gs) < 5:
            continue
        print(f"{a:<22}{wr(gs):>18}{wr(ours.get(a, [])):>18}")

    print("\n=== 3. Crustle-стена (карта 345 в деке оппонента) ===")
    for label, sel in (("все топ-игры", rows), ("топы нашей декой", grim)):
        w_ = [g for g in sel if g["opp_has_crustle"]]
        print(f"  {label:<20} встреч со стеной {len(w_):>4}/{len(sel)} "
              f"({len(w_)/len(sel)*100:.1f}% поля)   WR топов на стене: {wr(w_)}")
    if lp.exists():
        our_wall = [r for r in our_rows if "Crustle" in (r["cluster"] or "")]
        print(f"  {'мы (леджер)':<20} встреч со стеной {len(our_wall):>4}/{len(our_rows)} "
              f"({len(our_wall)/len(our_rows)*100:.1f}% поля)   наш WR на стене: {wr(our_wall)}")


if __name__ == "__main__":
    main()
