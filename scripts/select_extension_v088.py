#!/usr/bin/env python3
"""select_extension_v088.py — отбор эпизодов из пула v083_daily_full под v088.

Делает ДВА независимых отбора из одного скана пула (порядок важен: холдаут режется
ПЕРВЫМ, расширение — из остатка, поэтому пересечение невозможно по построению):

  1. holdout_field/  — БОЛЬШОЙ свежий холдаут поля (--holdout-episodes, равномерно по
     свободным эпизодам пула, seeded). Мотив (форензик 08-14, §D1): офлайн-прибор
     сейчас слеп на retreat — в замороженном test всего 108 retreat-решений,
     se≈4.8пп при заявленной цели +5пп, т.е. целевой эффект тонет в шуме измерения.
     ~4000 эпизодов дают ~2.5k retreat-решений (se≈1.2пп) и режутся по архетипам.
     Эти эпизоды НЕ входят ни в train, ни в extension — ни один кандидат их не видел.

  2. extension_dragapult/ — ВСЕ оставшиеся свободные эпизоды с opp_archetype==dragapult
     (K2: разрыв −21пп до эксперта при покрытии корпуса 11%, см. ANALYSIS §4).

Дедуп: (а) против merged-корпуса v086_targeted (train И замороженный test),
(б) между днями пула — эпизод пишется из ПЕРВОГО дня, где встретился (как в
select_targeted_extension_v086.py).

Двухпроходная схема: пас 1 — только (episode_id, opp_archetype, день) регексом
(2.2ГБ строк в память не влезут); пас 2 — выписывание отобранных строк.

Дальше по конвейеру:
  holdout:   featurize_corpus_2b.py --decisions holdout_field/decisions_extra.jsonl
             --obs-dir holdout_field/obs_shards --split holdout_field/split.json
  extension: merge_corpus_2b_daily_extension.py --base .../merged
             --extension .../extension_dragapult -> featurize -> concat parquet
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RE_EP = re.compile(rb'"episode_id":\s*(\d+)')
RE_AR = re.compile(rb'"opp_archetype":\s*"([^"]*)"')


def scan_pool(shards_root: Path, existing: set[int]) -> tuple[dict[int, str], dict[int, set[str]], dict[int, int]]:
    """Пас 1: eid -> день (первый), eid -> множество архетипов, eid -> число строк."""
    ep_day: dict[int, str] = {}
    ep_arch: dict[int, set[str]] = defaultdict(set)
    ep_rows: dict[int, int] = defaultdict(int)
    n_days = 0
    for day in sorted(shards_root.iterdir()):
        dec = day / "decisions_extra.jsonl"
        if not dec.is_file() or dec.stat().st_size == 0:
            continue
        n_days += 1
        with open(dec, "rb") as f:
            for line in f:
                m = RE_EP.search(line)
                if not m:
                    continue
                eid = int(m.group(1))
                if eid in existing:
                    continue
                if eid in ep_day and ep_day[eid] != day.name:
                    continue  # эпизод уже закреплён за более ранним днём
                ep_day.setdefault(eid, day.name)
                ma = RE_AR.search(line)
                ep_arch[eid].add(ma.group(1).decode() if ma else "none")
                ep_rows[eid] += 1
    print(f"[select_v088] дней просканировано: {n_days}; свободных эпизодов: {len(ep_day)}; "
          f"свободных решений: {sum(ep_rows.values())}", file=sys.stderr)
    return ep_day, ep_arch, ep_rows


def write_bundle(out_root: Path, name: str, eps: set[int], ep_day: dict[int, str],
                 shards_root: Path, as_test_split: bool) -> dict:
    """Пас 2 для одного набора: выписывает строки отобранных эпизодов + симлинки obs-шардов.
    as_test_split=True -> дополнительно split.json, где ВСЕ эпизоды помечены test (холдаут:
    featurize_corpus_2b.py читает split.json и штампует колонку split)."""
    d = out_root / name
    shards = d / "obs_shards"
    shards.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    written: set[int] = set()
    with open(d / "decisions_extra.jsonl", "w") as out_f:
        for day in sorted(shards_root.iterdir()):
            dec = day / "decisions_extra.jsonl"
            if not dec.is_file() or dec.stat().st_size == 0:
                continue
            with open(dec, "rb") as f:
                for line in f:
                    m = RE_EP.search(line)
                    if not m:
                        continue
                    eid = int(m.group(1))
                    if eid not in eps or ep_day[eid] != day.name:
                        continue
                    out_f.write(line.decode())
                    n_rows += 1
                    if eid not in written:
                        written.add(eid)
                        src = day / "obs_shards" / f"{eid}.jsonl"
                        if not src.is_file():
                            sys.exit(f"FATAL: нет obs-шарда для эпизода {eid}: {src}")
                        dest = shards / f"{eid}.jsonl"
                        if not (dest.exists() or dest.is_symlink()):
                            dest.symlink_to(src.resolve())
    if as_test_split:
        (d / "split.json").write_text(json.dumps(
            {"train_episode_ids": [], "test_episode_ids": sorted(written)}))
    missing = eps - written
    if missing:
        sys.exit(f"FATAL: {len(missing)} отобранных эпизодов не выписаны (напр. {sorted(missing)[:5]})")
    return {"episodes": len(written), "decisions": n_rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards-root", default=str(
        REPO / "out" / "corpus_2b" / "versions" / "v083_daily_full" / "shards"))
    ap.add_argument("--merged-decisions", default=str(
        REPO / "out" / "corpus_2b" / "versions" / "v086_targeted" / "merged" / "decisions.jsonl"))
    ap.add_argument("--out-root", default=str(
        REPO / "out" / "corpus_2b" / "versions" / "v088_dragapult"))
    ap.add_argument("--holdout-episodes", type=int, default=4000,
                    help="размер свежего холдаута поля в ЭПИЗОДАХ (равномерно из свободных)")
    ap.add_argument("--target-archetype", default="dragapult",
                    help="архетип целевого дообогащения (K2)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    shards_root = Path(args.shards_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    existing: set[int] = set()
    with open(args.merged_decisions, "rb") as f:
        for line in f:
            m = RE_EP.search(line)
            if m:
                existing.add(int(m.group(1)))
    print(f"[select_v088] merged-корпус: {len(existing)} эпизодов (дедуп-база)", file=sys.stderr)

    ep_day, ep_arch, ep_rows = scan_pool(shards_root, existing)

    free_eps = sorted(ep_day)
    rng = random.Random(args.seed)
    rng.shuffle(free_eps)
    n_hold = min(args.holdout_episodes, len(free_eps))
    hold_eps = set(free_eps[:n_hold])
    rest_eps = free_eps[n_hold:]

    tgt_eps = {e for e in rest_eps if args.target_archetype in ep_arch[e]}

    assert not (hold_eps & tgt_eps), "холдаут и расширение пересеклись — баг отбора"

    rep = {
        "seed": args.seed,
        "free_episodes_total": len(ep_day),
        "free_decisions_total": sum(ep_rows.values()),
        "holdout_field": write_bundle(out_root, "holdout_field", hold_eps, ep_day,
                                      shards_root, as_test_split=True),
        f"extension_{args.target_archetype}": write_bundle(
            out_root, f"extension_{args.target_archetype}", tgt_eps, ep_day,
            shards_root, as_test_split=False),
        "holdout_archetype_mix": {},
        "target_archetype": args.target_archetype,
    }
    mix: dict[str, int] = defaultdict(int)
    for e in hold_eps:
        for a in ep_arch[e]:
            mix[a] += 1
    rep["holdout_archetype_mix"] = dict(sorted(mix.items(), key=lambda kv: -kv[1]))

    (out_root / "selection_report_v088.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
