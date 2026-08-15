"""build_value_dataset_gs.py -- value-model dataset builder (PLAN_v084_SEARCH.md §3ter item 2).

Собирает p(win) датасет для будущей leaf-eval value-модели (search_gs.py v2) из
`out/corpus_2b/versions/v083_daily_full/merged/{decisions.jsonl,obs_shards/}` (merge_report.json
подтверждает эту структуру: 672316 решений / 6331 obs_shards-эпизода, base+extension слиты).

Метод (в точности постановка задачи):
  1. Стрим decisions.jsonl, отбираем ПЕРВОЕ MAIN-решение каждой пары
     (episode_id, agent_index, turn_idx) -- один снимок доски на ход на сторону.
  2. Кап ~400-500k снимков РАВНОМЕРНЫМ прореживанием (стабильный stride по отсортированному
     списку ключей), НЕ "первые N" -- на случай, если этот скрипт когда-нибудь запустят на
     более крупном срезе корпуса. На фактическом merged-срезе (см. отчёт в конце запуска)
     доступно МЕНЬШЕ, чем кап (44942 снимка) -- кап в этом случае не бьёт вообще, идёт
     ВЕСЬ пул без прореживания (см. отчёт сессии).
  3. Для каждого отобранного снимка тянем obs из obs_shards/{episode_id}.jsonl[obs_ref]
     (obs_ref -- построчный индекс В ЭТОМ файле, проверено на примере эпизода -- 1 obs-строка
     на 1 decision-запись, тот же порядок).
  4. Фичи -- ЕДИНСТВЕННЫЙ источник истины: value_features_gs.value_features(obs, agent_index)
     ИЗ submissions/v084_bc2b_search/ (импорт, не копия/переопределение).
  5. Сплит train/test -- zlib.crc32(str(episode_id).encode()) % 5 == 0 -> test (НЕ встроенный
     hash(), который рандомизирован по PYTHONHASHSEED между процессами/запусками -- см. проектную
     память feedback_measurement_and_ritual_discipline о дисциплине измерения: сплит должен быть
     воспроизводим байт-в-байт при перезапуске).
  6. Пишем матрицу в out/value_gs/dataset.npz (+ out/value_gs/meta.json).

Ничего не мутирует за пределами out/value_gs/ и __pycache__. Использует numpy (офлайн-скрипт,
.venv позволяет; сама модель/сабмит остаются чистым Python -- см. постановку задачи).
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_DIR = os.path.join(ROOT, "submissions", "v084_bc2b_search")
MERGED_DIR = os.path.join(ROOT, "out", "corpus_2b", "versions", "v083_daily_full", "merged")
DECISIONS_PATH = os.path.join(MERGED_DIR, "decisions.jsonl")
OBS_SHARD_DIR = os.path.join(MERGED_DIR, "obs_shards")
OUT_DIR = os.path.join(ROOT, "out", "value_gs")

CAP = 500_000  # верхняя граница из постановки задачи (~400-500k); фактический пул может быть меньше
TEST_MOD = 5  # crc32(episode_id) % TEST_MOD == 0 -> test

# submissions/v084_bc2b_search -- единственный источник истины для value_features (контракт
# задачи: "импортирует value_features_gs ИЗ директории сабмита").
sys.path.insert(0, SUB_DIR)
import value_features_gs as vfg  # noqa: E402


def collect_first_main_records():
    """Pass 1: стрим decisions.jsonl, ПЕРВОЕ MAIN-решение на (episode_id, agent_index, turn_idx).
    Возвращает list[tuple] в порядке появления в файле (порядок неважен для последующего
    равномерного прореживания -- он делается по отсортированному ключу, см. cap_uniform)."""
    seen = set()
    records = []
    n_lines = 0
    n_main = 0
    with open(DECISIONS_PATH) as f:
        for line in f:
            n_lines += 1
            d = json.loads(line)
            if d.get("sel_context_name") != "MAIN":
                continue
            n_main += 1
            key = (d["episode_id"], d["agent_index"], d["turn_idx"])
            if key in seen:
                continue
            seen.add(key)
            records.append((
                d["episode_id"], d["agent_index"], d["turn_idx"], d["obs_ref"],
                bool(d["game_won"]), d.get("deck_key"), d.get("opp_archetype"),
            ))
    return records, n_lines, n_main


def cap_uniform(records, cap):
    """Равномерное прореживание до `cap` записей -- НЕ первые N. Сортируем по стабильному ключу
    (episode_id, agent_index, turn_idx) для воспроизводимости, затем берём каждый k-й элемент
    (k = ceil(n/cap)), гарантируя покрытие всего диапазона эпизодов/ходов, а не только начала
    файла (которое в decisions.jsonl соответствует более РАННИМ датам скана -- см.
    build_corpus_2b_daily_extension.py -- сэмпл по первым N был бы смещён по времени)."""
    n = len(records)
    if n <= cap:
        return records, False
    records_sorted = sorted(records, key=lambda r: (r[0], r[2], r[1]))
    stride = n / cap
    idx = 0.0
    picked = []
    seen_i = set()
    while len(picked) < cap and int(idx) < n:
        i = int(idx)
        if i not in seen_i:
            picked.append(records_sorted[i])
            seen_i.add(i)
        idx += stride
    return picked, True


def build_matrix(records):
    """Pass 2: группируем по episode_id, читаем ровно нужные строки obs_shards/{eid}.jsonl,
    вызываем value_features_gs.value_features(obs, agent_index) на каждой."""
    by_ep = collections.defaultdict(list)
    for r in records:
        by_ep[r[0]].append(r)

    n_features = vfg.N_FEATURES
    n = len(records)
    X = np.empty((n, n_features), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    episode_id_arr = np.empty(n, dtype=np.int64)
    agent_index_arr = np.empty(n, dtype=np.int8)
    turn_idx_arr = np.empty(n, dtype=np.int32)
    deck_key_arr = np.empty(n, dtype=object)
    opp_archetype_arr = np.empty(n, dtype=object)
    is_test_arr = np.empty(n, dtype=bool)

    row = 0
    n_missing_shard = 0
    n_missing_line = 0

    for ep_id, recs in by_ep.items():
        needed = {r[3]: r for r in recs}
        path = os.path.join(OBS_SHARD_DIR, f"{ep_id}.jsonl")
        try:
            f = open(path)
        except FileNotFoundError:
            n_missing_shard += len(recs)
            continue
        found = set()
        with f:
            for i, line in enumerate(f):
                rec = needed.get(i)
                if rec is None:
                    continue
                found.add(i)
                obs = json.loads(line)
                _, agent_index, turn_idx, _obs_ref, game_won, deck_key, opp_archetype = rec
                feats = vfg.value_features(obs, agent_index)
                X[row, :] = feats
                y[row] = 1.0 if game_won else 0.0
                episode_id_arr[row] = ep_id
                agent_index_arr[row] = agent_index
                turn_idx_arr[row] = turn_idx
                deck_key_arr[row] = deck_key
                opp_archetype_arr[row] = opp_archetype
                is_test_arr[row] = (zlib.crc32(str(ep_id).encode()) % TEST_MOD == 0)
                row += 1
        n_missing_line += len(needed) - len(found)

    if row != n:
        # some obs lines were missing (shard shorter than expected) -- trim arrays to actual count
        X = X[:row]
        y = y[:row]
        episode_id_arr = episode_id_arr[:row]
        agent_index_arr = agent_index_arr[:row]
        turn_idx_arr = turn_idx_arr[:row]
        deck_key_arr = deck_key_arr[:row]
        opp_archetype_arr = opp_archetype_arr[:row]
        is_test_arr = is_test_arr[:row]

    return {
        "X": X, "y": y, "episode_id": episode_id_arr, "agent_index": agent_index_arr,
        "turn_idx": turn_idx_arr, "deck_key": deck_key_arr, "opp_archetype": opp_archetype_arr,
        "is_test": is_test_arr,
    }, n_missing_shard, n_missing_line


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    records, n_lines, n_main = collect_first_main_records()
    n_pool = len(records)
    records, was_capped = cap_uniform(records, CAP)
    n_used = len(records)

    data, n_missing_shard, n_missing_line = build_matrix(records)
    n_rows = data["X"].shape[0]

    np.savez_compressed(
        os.path.join(OUT_DIR, "dataset.npz"),
        X=data["X"], y=data["y"], episode_id=data["episode_id"],
        agent_index=data["agent_index"], turn_idx=data["turn_idx"],
        deck_key=data["deck_key"].astype(str), opp_archetype=data["opp_archetype"].astype(str),
        is_test=data["is_test"],
    )

    n_test = int(data["is_test"].sum())
    n_train = n_rows - n_test
    win_rate = float(data["y"].mean()) if n_rows else float("nan")
    win_rate_train = float(data["y"][~data["is_test"]].mean()) if n_train else float("nan")
    win_rate_test = float(data["y"][data["is_test"]].mean()) if n_test else float("nan")

    turn_vals = data["turn_idx"]
    turn_hist = {
        "min": int(turn_vals.min()) if n_rows else None,
        "max": int(turn_vals.max()) if n_rows else None,
        "p10": float(np.percentile(turn_vals, 10)) if n_rows else None,
        "p50": float(np.percentile(turn_vals, 50)) if n_rows else None,
        "p90": float(np.percentile(turn_vals, 90)) if n_rows else None,
    }

    archetype_counts = collections.Counter(data["opp_archetype"].tolist())
    deck_key_counts = collections.Counter(data["deck_key"].tolist())
    n_episodes = len(set(data["episode_id"].tolist()))
    n_episodes_test = len(set(data["episode_id"][data["is_test"]].tolist()))
    n_episodes_train = len(set(data["episode_id"][~data["is_test"]].tolist()))

    meta = {
        "source_decisions": os.path.relpath(DECISIONS_PATH, ROOT),
        "source_obs_shards": os.path.relpath(OBS_SHARD_DIR, ROOT),
        "feature_module": os.path.relpath(os.path.join(SUB_DIR, "value_features_gs.py"), ROOT),
        "n_features": vfg.N_FEATURES,
        "feature_order": vfg.FEATURE_ORDER,
        "n_lines_decisions_jsonl": n_lines,
        "n_main_decisions": n_main,
        "n_pool_first_main_snapshots": n_pool,
        "cap": CAP,
        "was_capped": was_capped,
        "n_used_after_cap": n_used,
        "n_missing_shard_records": n_missing_shard,
        "n_missing_line_records": n_missing_line,
        "n_rows_final": n_rows,
        "n_train": n_train,
        "n_test": n_test,
        "n_episodes_total": n_episodes,
        "n_episodes_train": n_episodes_train,
        "n_episodes_test": n_episodes_test,
        "test_split_rule": f"zlib.crc32(str(episode_id).encode()) % {TEST_MOD} == 0",
        "win_rate_all": win_rate,
        "win_rate_train": win_rate_train,
        "win_rate_test": win_rate_test,
        "turn_idx_histogram": turn_hist,
        "opp_archetype_counts": dict(archetype_counts.most_common()),
        "deck_key_counts": dict(deck_key_counts.most_common()),
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
