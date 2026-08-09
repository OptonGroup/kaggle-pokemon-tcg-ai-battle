#!/usr/bin/env python3
"""mirror_detect_recon.py — задание T2-B шаг 2: замер live-детекта _detect_matchup_raw
(submissions/v080_bc2b/main.py) на зеркальных партиях корпуса эталонов (out/corpus_2b).

Вопрос: если v081 планирует переключать модель на зеркальный вариант ПО _detect_matchup_raw,
насколько поздно/шумно это происходит? Специализация бесполезна, если детект долго молчит
("generic"/другое) в начале игры (первые ходы — самые частые в корпусе решения) или ложно
срабатывает в НЕ-зеркальных играх (тогда v081 применял бы зеркальную модель не туда).

Метод: воспроизводим _detect_matchup_raw РЕАЛЬНОГО main.py (импортируем модуль напрямую,
не копируем логику) последовательно по КАЖДОЙ игре корпуса (obs_shards/<episode_id>.jsonl,
порядок строк = хронологический, obs_ref decisions.jsonl на них ссылается 1:1), со сбросом
sticky-состояния (_raw_last_matchup=None) на границе игры — та же семантика, что в agent()
при obs.select is None.

Метрики:
  1) В ЗЕРКАЛЬНЫХ играх (opp_archetype==mirror_grimmsnarl): на каком по счёту решении/ходу
     впервые label=='mirror' (медиана/перцентили); доля решений ДО первого детекта (сколько
     решений в игре обслуживались бы generic-моделью, если бы переключение было по этому
     детекту); доля решений, где детект НЕ 'mirror' вообще (после первого детекта sticky
     должен держать 'mirror', но если line ids когда-либо пересекаются с ДРУГИМ приоритетным
     сетом раньше — детект может залипнуть на неверном ярлыке).
  2) В НЕ-зеркальных играх: доля решений, где детект ОШИБОЧНО говорит 'mirror' (ложное
     срабатывание -> v081 применил бы зеркальную модель не в том матчапе).

Запуск:
    .venv/bin/python scripts/mirror_detect_recon.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parent.parent
CORPUS = WORKTREE / "out" / "corpus_2b"
SUB_DIR = WORKTREE / "submissions" / "v080_bc2b"


def load_main_module():
    sys.path.insert(0, str(SUB_DIR))
    spec = importlib.util.spec_from_file_location("v080_main_for_recon", str(SUB_DIR / "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_decisions_by_episode() -> dict:
    by_ep = defaultdict(list)
    with open(CORPUS / "decisions.jsonl") as f:
        for line in f:
            d = json.loads(line)
            by_ep[d["episode_id"]].append(d)
    return by_ep


def main() -> int:
    mod = load_main_module()
    detect = mod._detect_matchup_raw

    by_ep = load_decisions_by_episode()
    print(f"[recon] {len(by_ep)} эпизодов, {sum(len(v) for v in by_ep.values())} решений", file=sys.stderr)

    mirror_first_detect_ordinal = []   # 1-based ordinal решения в игре, когда впервые label=='mirror'
    mirror_first_detect_turn = []      # turn_idx этого решения
    mirror_never_detected = 0
    mirror_total_decisions = 0
    mirror_wrong_label_decisions = 0   # решения в зеркальных играх, где label != 'mirror'
    mirror_wrong_label_after_first = 0  # то же, но ПОСЛЕ первого детекта (залип на неверном)
    mirror_episodes = 0

    nonmirror_total_decisions = 0
    nonmirror_false_positive_decisions = 0  # label=='mirror', но истина != mirror_grimmsnarl
    nonmirror_episodes_with_fp = 0
    nonmirror_episodes = 0
    fp_by_archetype = defaultdict(int)
    fp_total_by_archetype = defaultdict(int)

    per_episode_rows = []

    for ep_id, rows in by_ep.items():
        # obs_ref sorted ascending == хронологический порядок (проверено на выборке;
        # decisions.jsonl пишется построчно при обходе игры, см. featurize_corpus_2b.py)
        rows_sorted = sorted(rows, key=lambda r: r["obs_ref"])
        obs_path = CORPUS / "obs_shards" / f"{ep_id}.jsonl"
        if not obs_path.exists():
            continue
        with open(obs_path) as f:
            obs_lines = f.readlines()

        mod._raw_last_matchup = None  # сброс sticky-состояния на границе игры (как в agent())
        truth = rows_sorted[0]["opp_archetype"]
        is_mirror = truth == "mirror_grimmsnarl"

        first_detect_ord = None
        n_wrong = 0
        n_wrong_after = 0
        n_fp = 0
        for ordinal, row in enumerate(rows_sorted, start=1):
            obs_ref = row["obs_ref"]
            if obs_ref >= len(obs_lines):
                continue
            obs = json.loads(obs_lines[obs_ref])
            label = detect(obs)
            if is_mirror:
                if label == "mirror":
                    if first_detect_ord is None:
                        first_detect_ord = ordinal
                else:
                    n_wrong += 1
                    if first_detect_ord is not None:
                        n_wrong_after += 1
            else:
                if label == "mirror":
                    n_fp += 1
                    fp_by_archetype[truth] += 1
                fp_total_by_archetype[truth] += 1

        if is_mirror:
            mirror_episodes += 1
            mirror_total_decisions += len(rows_sorted)
            mirror_wrong_label_decisions += n_wrong
            mirror_wrong_label_after_first += n_wrong_after
            if first_detect_ord is None:
                mirror_never_detected += 1
            else:
                mirror_first_detect_ordinal.append(first_detect_ord)
                mirror_first_detect_turn.append(rows_sorted[first_detect_ord - 1]["turn_idx"])
            per_episode_rows.append({
                "episode_id": ep_id, "expert": rows_sorted[0]["expert"],
                "n_decisions": len(rows_sorted), "first_detect_ordinal": first_detect_ord,
                "first_detect_turn": (rows_sorted[first_detect_ord - 1]["turn_idx"]
                                       if first_detect_ord else None),
                "n_wrong_total": n_wrong, "n_wrong_after_first": n_wrong_after,
            })
        else:
            nonmirror_episodes += 1
            nonmirror_total_decisions += len(rows_sorted)
            nonmirror_false_positive_decisions += n_fp
            if n_fp > 0:
                nonmirror_episodes_with_fp += 1

    def pct(a, q):
        return float(np.percentile(a, q)) if a else None

    report = {
        "mirror": {
            "n_episodes": mirror_episodes,
            "n_decisions": mirror_total_decisions,
            "n_never_detected_episodes": mirror_never_detected,
            "first_detect_ordinal_stats": {
                "n": len(mirror_first_detect_ordinal),
                "mean": round(float(np.mean(mirror_first_detect_ordinal)), 2) if mirror_first_detect_ordinal else None,
                "median": pct(mirror_first_detect_ordinal, 50),
                "p10": pct(mirror_first_detect_ordinal, 10),
                "p90": pct(mirror_first_detect_ordinal, 90),
                "max": max(mirror_first_detect_ordinal) if mirror_first_detect_ordinal else None,
            },
            "first_detect_turn_idx_stats": {
                "mean": round(float(np.mean(mirror_first_detect_turn)), 2) if mirror_first_detect_turn else None,
                "median": pct(mirror_first_detect_turn, 50),
                "p90": pct(mirror_first_detect_turn, 90),
                "max": max(mirror_first_detect_turn) if mirror_first_detect_turn else None,
            },
            "decisions_before_first_detect_or_never":
                mirror_total_decisions - sum(
                    r["n_decisions"] - (r["first_detect_ordinal"] - 1 if r["first_detect_ordinal"] else 0)
                    for r in per_episode_rows
                ),
            "wrong_label_rate_overall": round(mirror_wrong_label_decisions / mirror_total_decisions, 4)
                if mirror_total_decisions else None,
            "wrong_label_rate_after_first_detect": round(
                mirror_wrong_label_after_first / max(1, mirror_total_decisions - mirror_never_detected), 4),
            "n_wrong_after_first_detect": mirror_wrong_label_after_first,
        },
        "nonmirror": {
            "n_episodes": nonmirror_episodes,
            "n_decisions": nonmirror_total_decisions,
            "n_episodes_with_false_positive": nonmirror_episodes_with_fp,
            "false_positive_rate_decisions": round(
                nonmirror_false_positive_decisions / nonmirror_total_decisions, 4)
                if nonmirror_total_decisions else None,
            "false_positive_by_archetype": {
                k: {"fp_decisions": fp_by_archetype.get(k, 0), "total_decisions": v,
                    "rate": round(fp_by_archetype.get(k, 0) / v, 4) if v else None}
                for k, v in sorted(fp_total_by_archetype.items(), key=lambda kv: -kv[1])
            },
        },
    }

    out_path = CORPUS / "mirror_detect_recon.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    rows_path = CORPUS / "mirror_detect_recon_episodes.jsonl"
    with open(rows_path, "w") as f:
        for r in per_episode_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[recon] -> {out_path}\n[recon] -> {rows_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
