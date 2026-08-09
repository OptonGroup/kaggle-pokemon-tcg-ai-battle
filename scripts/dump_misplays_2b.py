#!/usr/bin/env python3
"""dump_misplays_2b.py — T1.4 п.3: человекочитаемый дамп промахов модели на TEST для ручного
разбора (play/retreat внутри group=main, весь group=c7). Промах = decision_level_eval с
model_match=False (predicted top-k set != true expert set), см. train_bc_2b.py.

НЕ импортирует featurize_corpus_2b.py (тот тянет cg_native — Linux-only, здесь macOS, см.
CLAUDE.md "Platform Notes") — переиспользует только ЧИСТО-Python константы/резолверы
(OPT_*, ZONE, card_at, resolve_opt_card_id), скопированные 1:1 отсюда:
    scripts/featurize_corpus_2b.py:105-107 (OPT_*), 126 (ZONE), 294-308 (card_at),
    483-518 (resolve_opt_card_id) — держать в синхроне, если те определения поменяются
    (маленький, редко трогаемый кусок; чисто для читаемости дампа, НЕ участвует в фичах/обучении).

Источники данных:
  out/corpus_2b/features/<group>.parquet — test split, meta-колонки + сырые X (для score)
  out/corpus_2b/models/<group>.txt       — обученный booster ПОСЛЕ финального ретрейна
  out/corpus_2b/obs_shards/<episode_id>.jsonl — сырой obs_dict, obs_ref = номер строки
  data/EN_Card_Data.csv                  — card_id -> человеческое имя

Запуск:
    .venv/bin/python scripts/dump_misplays_2b.py --group main --category play --n 60
    .venv/bin/python scripts/dump_misplays_2b.py --group main --category retreat --n 60
    .venv/bin/python scripts/dump_misplays_2b.py --group c7 --n 60
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "scripts"))
import train_bc_2b as tb  # noqa: E402

# ── скопировано из featurize_corpus_2b.py (см. докстринг выше) ──────────────────────────
OPT_NUMBER, OPT_YES, OPT_NO, OPT_CARD = 0, 1, 2, 3
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY = 7, 8, 9, 10
OPT_RETREAT, OPT_ATTACK, OPT_END = 12, 13, 14
AREA_HAND, AREA_ACTIVE, AREA_BENCH = 2, 4, 5
AREA_LOOKING = 12
ZONE = {1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench", 6: "prize"}
OPT_TYPE_NAMES = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 7: "PLAY", 8: "ATTACH",
                   9: "EVOLVE", 10: "ABILITY", 12: "RETREAT", 13: "ATTACK", 14: "END"}


def card_at(players, pi, area, idx):
    if area not in ZONE or not isinstance(idx, int):
        return None
    if not isinstance(pi, int) or not (0 <= pi < len(players)):
        return None
    zone = players[pi].get(ZONE[area])
    if not isinstance(zone, list) or not (0 <= idx < len(zone)):
        return None
    c = zone[idx]
    if not isinstance(c, dict):
        return None
    cid = c.get("id")
    return int(cid) if cid is not None else None


def _card_at_looking(looking, idx):
    if not isinstance(looking, list) or not isinstance(idx, int) or not (0 <= idx < len(looking)):
        return None
    c = looking[idx]
    if not isinstance(c, dict):
        return None
    cid = c.get("id")
    return int(cid) if cid is not None else None


def _card_at_sel_deck(sel_deck, idx):
    if not isinstance(sel_deck, list) or not isinstance(idx, int) or not (0 <= idx < len(sel_deck)):
        return None
    c = sel_deck[idx]
    if isinstance(c, dict):
        cid = c.get("id")
        return int(cid) if cid is not None else None
    if isinstance(c, int):
        return c
    return None


def resolve_opt_card_id(players, me, opt, looking=None, sel_deck=None):
    if not isinstance(opt, dict):
        return None
    if opt.get("area") == 1:
        cid = _card_at_sel_deck(sel_deck, opt.get("index"))
        if cid is not None:
            return cid
    if opt.get("cardId") is not None:
        try:
            return int(opt["cardId"])
        except (TypeError, ValueError):
            return None
    t = opt.get("type")
    if t == OPT_PLAY:
        return card_at(players, me, AREA_HAND, opt.get("index"))
    if t == OPT_ATTACH:
        return card_at(players, me, opt.get("inPlayArea"), opt.get("inPlayIndex"))
    if t == OPT_EVOLVE:
        return card_at(players, me, opt.get("area", AREA_HAND), opt.get("index"))
    if t == OPT_ABILITY:
        return card_at(players, me, opt.get("area"), opt.get("index"))
    if t == OPT_CARD:
        area = opt.get("area")
        if area == AREA_LOOKING:
            return _card_at_looking(looking, opt.get("index"))
        pi = opt.get("playerIndex")
        pi = int(pi) if isinstance(pi, int) else me
        return card_at(players, pi, area, opt.get("index"))
    return None
# ── конец скопированного блока ───────────────────────────────────────────────────────────


def load_card_names(path: Path) -> dict:
    names = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["Card ID"])
            except (ValueError, KeyError, TypeError):
                continue
            names[cid] = row.get("Card Name", "?")
    return names


def fmt_card(cid, names: dict) -> str:
    if cid is None:
        return "?"
    return f"{names.get(cid, '?')}({cid})"


def fmt_mon(mon: dict, names: dict) -> str:
    if not mon:
        return "-"
    cid = mon.get("id")
    hp = mon.get("hp")
    maxhp = mon.get("maxHp")
    n_en = len(mon.get("energies") or [])
    return f"{fmt_card(cid, names)} hp={hp}/{maxhp} energies={n_en}"


def fmt_opt(opt: dict, players, me: int, looking, sel_deck, names: dict) -> str:
    t = opt.get("type")
    tname = OPT_TYPE_NAMES.get(t, f"type={t}")
    cid = resolve_opt_card_id(players, me, opt, looking, sel_deck)
    extra = ""
    if t == OPT_ATTACK:
        extra = f" attackId={opt.get('attackId')}"
    if t == OPT_ATTACH:
        target = card_at(players, me, opt.get("inPlayArea"), opt.get("inPlayIndex"))
        extra = f" -> attach to {fmt_card(target, names)}"
    card_part = f" card={fmt_card(cid, names)}" if cid is not None else ""
    return f"[{tname}{card_part}{extra}]"


def load_obs(obs_dir: Path, episode_id: int, obs_ref: int) -> dict:
    path = obs_dir / f"{episode_id}.jsonl"
    with open(path) as f:
        for i, line in enumerate(f):
            if i == obs_ref:
                return json.loads(line)
    raise RuntimeError(f"obs_ref {obs_ref} not found in {path}")


def category_of_type(t) -> str:
    try:
        ti = int(t)
    except (TypeError, ValueError):
        return "other"
    return {OPT_ATTACK: "attack", OPT_ATTACH: "energy-attach", OPT_RETREAT: "retreat",
            OPT_PLAY: "play", OPT_ABILITY: "ability", OPT_END: "end"}.get(ti, "other")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True, choices=["main", "c7", "low", "mid", "easy", "rest"])
    ap.add_argument("--category", default=None, help="фильтр по категории true-опции (только для group=main): play/retreat/attack/ability/energy-attach/end")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features-dir", default=str(WORKTREE / "out" / "corpus_2b" / "features"))
    ap.add_argument("--models-dir", default=str(WORKTREE / "out" / "corpus_2b" / "models"))
    ap.add_argument("--obs-dir", default=str(WORKTREE / "out" / "corpus_2b" / "obs_shards"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    features_dir = Path(args.features_dir)
    spec = json.loads((features_dir / "feature_spec.json").read_text())
    feature_order = spec["feature_order"]
    names = load_card_names(WORKTREE / "data" / "EN_Card_Data.csv")

    df = tb.load_group_df(features_dir, args.group, feature_order)
    test = df[df["split"] == "test"].reset_index(drop=True)
    booster = lgb.Booster(model_file=str(Path(args.models_dir) / f"{args.group}.txt"))
    X_test = test[feature_order].to_numpy(dtype=np.float32)
    scores = booster.predict(X_test, num_iteration=booster.best_iteration or None)

    uid = test["decision_uid"].to_numpy()
    opt_idx = test["option_idx"].to_numpy()
    y = test["y"].to_numpy()
    min_count = test["min_count"].to_numpy()
    max_count = test["max_count"].to_numpy()
    opt_type = test["opt_type"].to_numpy()
    starts, ends = tb.contiguous_bounds(uid)

    misses = []
    for j in range(len(starts)):
        s, e = starts[j], ends[j]
        n_opt = e - s
        k = int(min_count[s])
        if k == 0:
            mx = int(max_count[s]) if max_count[s] == max_count[s] else 1
            k = mx if mx > 0 else 1
        k = max(0, min(k, n_opt))
        seg_opt = opt_idx[s:e]
        seg_y = y[s:e]
        true_idx = set(seg_opt[seg_y == 1].tolist())
        seg_scores = scores[s:e]
        order_by_score = np.argsort(-seg_scores, kind="stable")
        pred_idx = set(seg_opt[order_by_score[:k]].tolist()) if k > 0 else set()
        if pred_idx == true_idx:
            continue  # match, не промах
        true_rows = np.where(seg_y == 1)[0]
        cat = category_of_type(opt_type[s + true_rows[0]]) if len(true_rows) else "other"
        if args.category is not None and cat != args.category:
            continue
        misses.append({
            "decision_uid": uid[s], "start": s, "end": e, "k": k, "category": cat,
            "true_idx": true_idx, "pred_idx": pred_idx,
            "row_local_scores": {int(seg_opt[i]): float(seg_scores[i]) for i in range(n_opt)},
        })

    rng = np.random.default_rng(args.seed)
    if len(misses) > args.n:
        sel = rng.choice(len(misses), size=args.n, replace=False)
        sample = [misses[i] for i in sorted(sel.tolist())]
    else:
        sample = misses

    print(f"[dump_misplays] group={args.group} category={args.category} total_misses={len(misses)} "
          f"n_test_decisions={len(starts)} sampled={len(sample)}", file=sys.stderr)

    out_lines = []
    out_lines.append(f"=== group={args.group} category={args.category} total_misses={len(misses)}/{len(starts)} sampled={len(sample)} ===\n")
    meta_cols = ["episode_id", "agent_index", "turn_idx", "step_idx", "expert", "opp_archetype",
                 "subj_archetype", "game_won", "sel_context_name"]

    for m in sample:
        row0 = test.iloc[m["start"]]
        ep_id = int(row0["episode_id"])
        agent_idx = int(row0["agent_index"])
        obs_ref_str = m["decision_uid"].split(":")[-1]
        obs_ref = int(obs_ref_str)
        try:
            obs = load_obs(Path(args.obs_dir), ep_id, obs_ref)
        except Exception as ex:
            out_lines.append(f"--- {m['decision_uid']} FAILED to load obs: {ex}\n")
            continue

        cur = obs.get("current", {})
        players = cur.get("players", [])
        me = agent_idx
        opp = 1 - me
        sel = obs.get("select", {})
        looking = cur.get("looking")
        sel_deck = sel.get("deck")

        mine = players[me] if me < len(players) else {}
        theirs = players[opp] if opp < len(players) else {}

        header = (f"--- {m['decision_uid']}  cat={m['category']}  turn={cur.get('turn')} "
                  f"expert={row0['expert']}  opp_archetype={row0['opp_archetype']}  "
                  f"game_won={row0['game_won']}  sel_context={row0['sel_context_name']}  k={m['k']}")
        out_lines.append(header)
        out_lines.append(f"  own active: {[fmt_mon(mon, names) for mon in (mine.get('active') or [])]}")
        out_lines.append(f"  own bench: {[fmt_mon(mon, names) for mon in (mine.get('bench') or [])]}")
        own_hand = mine.get("hand")
        if own_hand is not None:
            out_lines.append(f"  own hand: {[fmt_card(c.get('id'), names) for c in own_hand]}")
        else:
            out_lines.append(f"  own hand: (hidden, count={mine.get('handCount')})")
        out_lines.append(f"  opp active: {[fmt_mon(mon, names) for mon in (theirs.get('active') or [])]}")
        out_lines.append(f"  opp bench: {[fmt_mon(mon, names) for mon in (theirs.get('bench') or [])]}")
        out_lines.append(f"  own prize left: {sum(1 for p in mine.get('prize', []) if p is not None) if isinstance(mine.get('prize'), list) else '?'}"
                          f"  opp prize left: {sum(1 for p in theirs.get('prize', []) if p is not None) if isinstance(theirs.get('prize'), list) else '?'}")

        options = sel.get("option") or []
        for i, opt in enumerate(options):
            marks = []
            if i in m["true_idx"]:
                marks.append("EXPERT")
            if i in m["pred_idx"]:
                marks.append("MODEL")
            mark_str = f" <-- {'+'.join(marks)}" if marks else ""
            sc = m["row_local_scores"].get(i)
            sc_str = f" score={sc:.3f}" if sc is not None else ""
            out_lines.append(f"    opt[{i}] {fmt_opt(opt, players, me, looking, sel_deck, names)}{sc_str}{mark_str}")
        out_lines.append("")

    text = "\n".join(out_lines)
    if args.out:
        Path(args.out).write_text(text)
        print(f"[dump_misplays] wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
