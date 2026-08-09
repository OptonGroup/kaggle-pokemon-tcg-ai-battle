#!/usr/bin/env python3
"""profile_r1a_endturn_cost.py — п.6 ТЗ: сколько РЕАЛЬНО дороже end-of-turn развёртка
на прод-параметре depth=3 (policy.json, не тронут). Берём несколько живых
MAIN-снимков середины игры, гоняем search_choice() напрямую (без agent()/episode-
бюджета) с ОГРОМНЫМ budget_ms (чтобы не срезать по времени — иначе ratio занижен)
и МАЛЫМ фиксированным samples (иначе прод (samples=64) будет минуты на снимок),
меряем wall time OFF vs ON на ОДНОМ и том же снимке — это чистое отношение
"стоимость одного сэмпла с endturn=True / endturn=False" при depth=3, не зависящее
от budget_ms/samples (масштабируется линейно).

Запуск:
    .venv/bin/python scripts/profile_r1a_endturn_cost.py
"""
from __future__ import annotations

import copy
import importlib.util
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
import cg_native, cg_native.api, cg_native.game, cg_native.sim, cg_native.utils  # noqa: E402
sys.modules.setdefault("cg", cg_native)
sys.modules.setdefault("cg.api", cg_native.api)
sys.modules.setdefault("cg.game", cg_native.game)
sys.modules.setdefault("cg.sim", cg_native.sim)
sys.modules.setdefault("cg.utils", cg_native.utils)

CAND_DIR = os.path.join(ROOT, "submissions", "scout_r1a_endturn")
OPP_DECK = os.path.join(ROOT, "decks", "meta", "Dragapult.csv")
CTX_MAIN = 0
SAMPLES_FOR_PROFILE = 3     # мало, но >1 -- за счёт budget_ms=huge гарантированно все samples доиграют
BUDGET_MS_HUGE = 600000     # не режем по времени -- честное сравнение стоимости
N_SNAPSHOTS_WANTED = 8
MIN_TURN = int(os.environ.get("PROFILE_MIN_TURN", "4"))


def _load_deck(path):
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip()]


def _load_module(sub_dir, tag):
    main_path = os.path.join(sub_dir, "main.py")
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    mod_name = f"_profile_{tag}_{id(object())}"
    cwd0 = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(mod_name, main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd0)
    return mod


def greedy(obs):
    sel = obs.get("select")
    if sel is None:
        return []
    opts = sel.get("option") or []
    n = len(opts)
    if n == 0:
        return []
    for ot in (13, 10, 9, 7, 8, 12, 14):
        idxs = [i for i in range(n) if opts[i].get("type") == ot]
        if idxs:
            return [idxs[0]]
    lo, hi = sel.get("minCount") or 0, min(sel.get("maxCount") or 1, n)
    return list(range(max(lo, min(1, hi))))


def collect_snapshots(n_wanted: int) -> list:
    """Собираем через ГРИДИ-агента с обеих сторон (быстро, без search) -- нам нужны
    только реалистичные MAIN-состояния середины игры, не решения кандидата."""
    d0 = _load_deck(os.path.join(CAND_DIR, "deck.csv"))
    d1 = _load_deck(OPP_DECK)
    random.seed(555)
    obs_dict, _ = battle_start(d0, d1)
    snaps = []
    moves = 0
    try:
        while obs_dict is not None and moves < 3000 and len(snaps) < n_wanted:
            cur = obs_dict.get("current") or {}
            if cur.get("result", -1) != -1:
                break
            sel = obs_dict.get("select")
            if sel is None:
                break
            you = cur.get("yourIndex", 0)
            if (you == 0 and sel.get("context") == CTX_MAIN
                    and len(sel.get("option") or []) >= 4 and (cur.get("turn") or 0) >= MIN_TURN):
                snaps.append(copy.deepcopy(obs_dict))
            obs_dict = battle_select(greedy(obs_dict))
            moves += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass
    return snaps


def time_search_choice(mod, obs_dict, endturn: bool, samples: int, budget_ms: int, depth) -> dict:
    os.environ["SCOUT_R1A_ENDTURN"] = "1" if endturn else "0"
    saved_samples = mod.SEARCH_CFG.get("samples")
    saved_budget = mod.SEARCH_CFG.get("budget_ms")
    saved_depth = mod.SEARCH_CFG.get("depth")
    mod.SEARCH_CFG["samples"] = samples
    mod.SEARCH_CFG["budget_ms"] = budget_ms
    if depth is not None:
        mod.SEARCH_CFG["depth"] = depth
    try:
        random.seed(11)
        t0 = time.perf_counter()
        picked = mod.search_choice(copy.deepcopy(obs_dict))
        dt = time.perf_counter() - t0
        vc0 = mod._SPENT.get("value_calls", 0)
        return dict(dt=dt, picked=picked, value_calls=vc0)
    finally:
        mod.SEARCH_CFG["samples"] = saved_samples
        mod.SEARCH_CFG["budget_ms"] = saved_budget
        mod.SEARCH_CFG["depth"] = saved_depth


def main() -> int:
    print(f"Собираем {N_SNAPSHOTS_WANTED} снимков середины игры (греди vs греди, быстро)...")
    snaps = collect_snapshots(N_SNAPSHOTS_WANTED)
    print(f"Снимков: {len(snaps)}")
    if not snaps:
        print("НЕТ снимков.")
        return 1

    depth_prod = None  # None => берём depth из policy.json как есть (=3 в проде)
    print(f"\nprod depth=3 (policy.json), samples={SAMPLES_FOR_PROFILE} (фикс, budget_ms={BUDGET_MS_HUGE} "
          f"-- не режем по времени, честное сравнение)\n")
    ratios = []
    for i, snap in enumerate(snaps):
        mod_off = _load_module(CAND_DIR, f"off{i}")
        mod_on = _load_module(CAND_DIR, f"on{i}")
        r_off = time_search_choice(mod_off, snap, False, SAMPLES_FOR_PROFILE, BUDGET_MS_HUGE, depth_prod)
        r_on = time_search_choice(mod_on, snap, True, SAMPLES_FOR_PROFILE, BUDGET_MS_HUGE, depth_prod)
        ratio = r_on["dt"] / r_off["dt"] if r_off["dt"] > 0 else float("nan")
        ratios.append(ratio)
        n_opt = len(snap["select"]["option"])
        turn = snap["current"].get("turn")
        print(f"  snap {i} turn={turn} n_opt={n_opt}: "
              f"OFF dt={r_off['dt']:.2f}s value_calls={r_off['value_calls']} picked={r_off['picked']}  |  "
              f"ON dt={r_on['dt']:.2f}s value_calls={r_on['value_calls']} picked={r_on['picked']}  "
              f"=> ratio ON/OFF = {ratio:.2f}x")

    ok_ratios = [r for r in ratios if r == r]  # drop nan
    if ok_ratios:
        mean_ratio = sum(ok_ratios) / len(ok_ratios)
        print(f"\nСредний ratio ON/OFF (depth=3, samples={SAMPLES_FOR_PROFILE} фикс, budget не давит): "
              f"{mean_ratio:.2f}x  (min={min(ok_ratios):.2f}x max={max(ok_ratios):.2f}x)")
        if mean_ratio > 5.0:
            print(f"=> >5x порог п.6 ТЗ ПРЕВЫШЕН ({mean_ratio:.2f}x) -- по умолчанию НЕ включать, "
                  f"нужен вариант с ограничением (см. отчёт).")
        else:
            print(f"=> <=5x порог п.6 ТЗ НЕ превышен ({mean_ratio:.2f}x) на этой выборке снимков.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
