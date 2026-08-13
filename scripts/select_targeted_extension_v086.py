#!/usr/bin/env python3
"""select_targeted_extension_v086.py — ЦЕЛЕВОЕ дообогащение корпуса 2B (v086).

Контекст (13.08): live-разбивка v085 по 20-очковым корзинам показала обрыв WR на 900
(61% ниже / 8% выше, Fisher p=0.0021); убийцы бенда 900+ — alakazam и lucario. При этом
в merged-корпусе v083 (672k решений, РАВНОМЕРНЫЙ сабсемпл из пула 5.19M) lucario-игр
~1% (в полном пуле — 63k решений / 726 партий, WR экспертов 50%), а alakazam взят лишь
на ~15% доступного (эксперты его БЬЮТ: WR 58%). Уровень-фильтры по экспертам (k1000/
k1020/k1050) опровергнуты офлайн 13.08: объём важнее чистоты. Отсюда единственный
живой рычаг — ДОБАВИТЬ целевые срезы, ничего не удаляя.

Что делает:
  1. Сканирует все дневные шарды пула (decisions_extra.jsonl).
  2. Дедуп по episode_id против merged-корпуса (и внутри пула между днями).
  3. lucario: берёт ВСЕ эпизоды; отщепляет seeded ~10% в ОТДЕЛЬНЫЙ holdout-каталог
     (НЕ попадает ни в train, ни в merged split — честная офлайн-оценка "оба
     кандидата никогда не видели эти игры").
  4. alakazam: равномерный seeded сабсемпл эпизодов до --alakazam-target решений
     (равномерный, НЕ топ-скор — см. опровержение фильтров качества).
  5. Пишет extension/{decisions_extra.jsonl,obs_shards/} и
     holdout_lucario/{decisions_extra.jsonl,obs_shards/} (симлинки на шардовые obs).

Дальше по конвейеру: merge_corpus_2b_daily_extension.py --base .../merged
--extension .../extension -> featurize чанком -> concat parquet -> train_bc_2b_v086.py.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards-root", default=str(
        REPO / "out" / "corpus_2b" / "versions" / "v083_daily_full" / "shards"))
    ap.add_argument("--merged-decisions", default=str(
        REPO / "out" / "corpus_2b" / "versions" / "v083_daily_full" / "merged" / "decisions.jsonl"))
    ap.add_argument("--out-root", default=str(
        REPO / "out" / "corpus_2b" / "versions" / "v086_targeted"))
    ap.add_argument("--alakazam-target", type=int, default=150_000)
    ap.add_argument("--lucario-holdout-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    shards_root = Path(args.shards_root)
    out_root = Path(args.out_root)

    # эпизоды, уже присутствующие в merged-корпусе (train И замороженный test)
    existing: set[int] = set()
    with open(args.merged_decisions) as f:
        for line in f:
            existing.add(int(json.loads(line)["episode_id"]))
    print(f"[select] merged-корпус: {len(existing)} эпизодов (дедуп-база)", file=sys.stderr)

    # скан пула: собираем строки целевых архетипов, группируя по эпизоду
    by_ep: dict[int, list[str]] = defaultdict(list)
    ep_arche: dict[int, set] = defaultdict(set)
    ep_daydir: dict[int, Path] = {}
    n_days = 0
    for day_dir in sorted(shards_root.iterdir()):
        dec = day_dir / "decisions_extra.jsonl"
        if not dec.is_file():
            continue
        n_days += 1
        with open(dec) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                a = d.get("opp_archetype")
                if a not in ("lucario", "alakazam"):
                    continue
                eid = int(d["episode_id"])
                if eid in existing:
                    continue
                # эпизод, уже виденный в другом дне пула, пишем только один раз
                if eid in ep_daydir and ep_daydir[eid] != day_dir:
                    continue
                ep_daydir[eid] = day_dir
                by_ep[eid].append(line)
                ep_arche[eid].add(a)
    print(f"[select] просканировано дней: {n_days}; кандидатов эпизодов: {len(by_ep)}",
          file=sys.stderr)

    # ВАЖНО: эпизод может содержать стороны с разными opp (зеркало не в счёт — мы взяли
    # только строки lucario/alakazam). Классифицируем: lucario приоритетнее (дефицитный).
    lucario_eps = sorted(e for e, ar in ep_arche.items() if "lucario" in ar)
    alakazam_eps = sorted(e for e, ar in ep_arche.items() if "lucario" not in ar)

    rng = random.Random(args.seed)
    rng.shuffle(lucario_eps)
    n_hold = max(1, int(len(lucario_eps) * args.lucario_holdout_frac))
    hold_eps = sorted(lucario_eps[:n_hold])
    luc_train_eps = sorted(lucario_eps[n_hold:])

    rng.shuffle(alakazam_eps)
    ala_keep: list[int] = []
    n_dec = 0
    for eid in alakazam_eps:
        rows = by_ep[eid]
        if n_dec and n_dec + len(rows) > args.alakazam_target:
            continue
        ala_keep.append(eid)
        n_dec += len(rows)
        if n_dec >= args.alakazam_target:
            break
    ala_keep = sorted(ala_keep)

    def write_bundle(name: str, eps: list[int]) -> dict:
        d = out_root / name
        shards = d / "obs_shards"
        shards.mkdir(parents=True, exist_ok=True)
        n_rows = 0
        with open(d / "decisions_extra.jsonl", "w") as out_f:
            for eid in eps:
                for line in by_ep[eid]:
                    out_f.write(line + "\n")
                    n_rows += 1
                src = ep_daydir[eid] / "obs_shards" / f"{eid}.jsonl"
                if not src.is_file():
                    sys.exit(f"FATAL: нет obs-шарда для эпизода {eid}: {src}")
                dest = shards / f"{eid}.jsonl"
                if not (dest.exists() or dest.is_symlink()):
                    dest.symlink_to(src.resolve())
        return {"episodes": len(eps), "decisions": n_rows}

    rep = {
        "extension": write_bundle("extension", luc_train_eps + ala_keep),
        "holdout_lucario": write_bundle("holdout_lucario", hold_eps),
        "detail": {
            "lucario_train": {"episodes": len(luc_train_eps),
                              "decisions": sum(len(by_ep[e]) for e in luc_train_eps)},
            "lucario_holdout": {"episodes": len(hold_eps),
                                "decisions": sum(len(by_ep[e]) for e in hold_eps)},
            "alakazam_extra": {"episodes": len(ala_keep), "decisions": n_dec},
        },
        "seed": args.seed,
    }
    (out_root / "selection_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
