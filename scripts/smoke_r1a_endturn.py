#!/usr/bin/env python3
"""smoke_r1a_endturn.py — живость + стоимость R1(a) (SCOUT_R1A_ENDTURN) на реальных
партиях (cg_native — валидный macOS-стенд, см. project_macos_stand_silently_disables_search
в памяти проекта). НЕ мера WR: N здесь мало (десятки), это ТОЛЬКО smoke —
0 crash/illegal, END-при-доступной-атаке % (OFF vs ON), время/эпизод (OFF vs ON),
число вызовов value-модели/эпизод, средняя стоимость на playout()-шаг.

Партии играются в отдельных процессах (ProcessPoolExecutor), кандидат =
submissions/scout_r1a_endturn, оппонент — greedy meta-агент (детерминированная
эвристика без собственного stderr-трейса, чтобы [scout]-лог в захваченном stderr
принадлежал ТОЛЬКО кандидату).

Опционально ускоряет прод-конфиг для трактабельности через --samples/--budget-ms/
--episode-soft-s/--episode-hard-s (патчит SEARCH_CFG/EPISODE_*_S уже ПОСЛЕ загрузки
модуля кандидата, в файле submissions/scout_r1a_endturn/policy.json НИЧЕГО не меняется).

Запуск:
    .venv/bin/python scripts/smoke_r1a_endturn.py --n 16 --opp meta:Dragapult \
        --samples 8 --budget-ms 8000 --episode-soft-s 40 --episode-hard-s 60 --workers 6
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import multiprocessing
import os
import re
import statistics
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

CANDIDATE_DIR = os.path.join(ROOT, "submissions", "scout_r1a_endturn")
MAX_MOVES_PER_GAME = 2000

_SUMMARY_RE = re.compile(
    r"\[scout\] episode done: decisions=(\d+) main=(\d+) "
    r"end_at_atk=(\d+)/(\d+) \(([\d.]+)%\) "
    r"value_calls=(\d+) search_time=([\d.]+)s "
    r"total_time=([\d.]+)s s/h=(\d+)/(\d+)"
)


def _load_deck(path):
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip()]


def _load_module(sub_dir: str, tag: str):
    main_path = os.path.join(sub_dir, "main.py")
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    mod_name = f"_smoke_r1a_{tag}_{os.getpid()}_{id(object())}"
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


_GREEDY_PRIO = (13, 10, 9, 7, 8, 12, 14)  # ATTACK, ABILITY, EVOLVE, PLAY, ATTACH, RETREAT, END


def greedy_meta_agent(obs_dict):
    """Простой greedy-оппонент без собственного stderr-трейса (см. докстринг)."""
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return []
    opts = sel.get("option") or []
    n = len(opts)
    if n == 0:
        return []
    for ot in _GREEDY_PRIO:
        idxs = [i for i in range(n) if opts[i].get("type") == ot]
        if idxs:
            return [idxs[0]]
    lo, hi = sel.get("minCount") or 0, min(sel.get("maxCount") or 1, n)
    return list(range(max(lo, min(1, hi))))


def _play_one(task: dict) -> dict:
    from cg_native.game import battle_start, battle_finish, battle_select
    import cg_native, cg_native.api, cg_native.game, cg_native.sim, cg_native.utils
    sys.modules.setdefault("cg", cg_native)
    sys.modules.setdefault("cg.api", cg_native.api)
    sys.modules.setdefault("cg.game", cg_native.game)
    sys.modules.setdefault("cg.sim", cg_native.sim)
    sys.modules.setdefault("cg.utils", cg_native.utils)

    endturn = task["endturn"]
    our_first = task["our_first"]
    opp_spec = task["opp_spec"]
    game_idx = task["game_idx"]
    seed = task["seed"]
    overrides = task["overrides"]

    os.environ["SCOUT_R1A_ENDTURN"] = "1" if endturn else "0"

    import random
    random.seed(seed)

    cand_mod = _load_module(CANDIDATE_DIR, "cand")
    if overrides.get("samples") is not None:
        cand_mod.SEARCH_CFG["samples"] = overrides["samples"]
    if overrides.get("budget_ms") is not None:
        cand_mod.SEARCH_CFG["budget_ms"] = overrides["budget_ms"]
    if overrides.get("episode_soft_s") is not None:
        cand_mod.EPISODE_SOFT_S = overrides["episode_soft_s"]
    if overrides.get("episode_hard_s") is not None:
        cand_mod.EPISODE_HARD_S = overrides["episode_hard_s"]
    cand_agent = cand_mod.agent
    cand_deck = cand_mod.read_deck()

    assert opp_spec.startswith("meta:"), f"bad opp_spec: {opp_spec}"
    opp_deck = _load_deck(os.path.join(ROOT, "decks", "meta", f"{opp_spec[5:]}.csv"))
    opp_agent = greedy_meta_agent

    if our_first:
        deck0, deck1 = list(cand_deck), list(opp_deck)
        agent0, agent1 = cand_agent, opp_agent
    else:
        deck0, deck1 = list(opp_deck), list(cand_deck)
        agent0, agent1 = opp_agent, cand_agent

    our_side = 0 if our_first else 1
    stderr_buf = io.StringIO()
    t0 = time.perf_counter()
    move_count = 0
    n_illegal = 0
    n_exceptions = 0
    winner = None
    with contextlib.redirect_stderr(stderr_buf):
        obs_dict, _start = battle_start(deck0, deck1)
        try:
            while obs_dict is not None and move_count < MAX_MOVES_PER_GAME:
                cur = obs_dict.get("current") or {}
                if cur.get("result", -1) != -1:
                    winner = cur.get("result")
                    break
                sel = obs_dict.get("select")
                if sel is None:
                    break
                you = cur.get("yourIndex", 0)
                fn = agent0 if you == 0 else agent1
                try:
                    choice = fn(obs_dict)
                except Exception:
                    n_exceptions += 1
                    break
                if not isinstance(choice, list):
                    n_illegal += 1
                    break
                obs_dict = battle_select(choice)
                move_count += 1
        finally:
            try:
                battle_finish()
            except Exception:
                pass
        # main.py агрегирует и печатает episode-summary (п.4 ТЗ: end_at_atk/value_calls/
        # search_time) ТОЛЬКО в момент запроса колоды СЛЕДУЮЩЕГО эпизода (sel is None) —
        # это уже поведение эталона (см. submissions/scout_deeper_p0p1: _SPENT сбрасывается
        # там же). Один процесс = одна партия здесь, следующего эпизода не будет, поэтому
        # форсируем flush явным вызовом agent({"select": None}) сразу после игры.
        try:
            cand_agent({"select": None})
        except Exception:
            pass
    elapsed = time.perf_counter() - t0
    log_text = stderr_buf.getvalue()

    m = _SUMMARY_RE.search(log_text)
    summary = None
    if m:
        summary = dict(
            decisions=int(m.group(1)), main=int(m.group(2)),
            end_at_atk=int(m.group(3)), main2=int(m.group(4)),
            end_at_atk_pct=float(m.group(5)), value_calls=int(m.group(6)),
            search_time_s=float(m.group(7)), total_time_s=float(m.group(8)),
            search_decisions=int(m.group(9)), heur_decisions=int(m.group(10)),
        )
    n_search_lines = log_text.count(" src=search ")
    n_decision_lines = log_text.count("[scout] d")

    return dict(
        game_idx=game_idx, endturn=endturn, our_first=our_first, seed=seed,
        moves=move_count, wallclock_s=elapsed, winner=winner,
        our_win=(winner == our_side) if winner is not None else None,
        n_illegal=n_illegal, n_exceptions=n_exceptions,
        n_search_lines=n_search_lines, n_decision_lines=n_decision_lines,
        summary=summary,
    )


def run_condition(endturn: bool, n_games: int, opp_spec: str, base_seed: int,
                   workers: int, overrides: dict) -> list:
    ctx = multiprocessing.get_context("spawn")
    queue = []
    for i in range(n_games):
        seed = base_seed + (zlib.crc32(f"r1a|{i}".encode()) % 1_000_000)
        queue.append(dict(endturn=endturn, our_first=(i % 2 == 0), opp_spec=opp_spec,
                           game_idx=i, seed=seed, overrides=overrides))
    results = []
    q_iter = iter(queue)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = {}
        for _ in range(workers):
            try:
                t = next(q_iter)
            except StopIteration:
                break
            futures[ex.submit(_play_one, t)] = t
        while futures:
            done, _pending = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                del futures[fut]
                r = fut.result()
                results.append(r)
                print(f"  [{'ON ' if endturn else 'OFF'}] game {r['game_idx']:2d} "
                      f"moves={r['moves']:4d} wall={r['wallclock_s']:6.1f}s "
                      f"win={r['our_win']} illegal={r['n_illegal']} exc={r['n_exceptions']} "
                      f"summary={r['summary']}")
                try:
                    t = next(q_iter)
                    futures[ex.submit(_play_one, t)] = t
                except StopIteration:
                    pass
    print(f"  condition endturn={endturn}: {len(results)} games in {time.time() - t0:.1f}s wall")
    return results


def aggregate(label: str, results: list) -> dict:
    n = len(results)
    n_illegal = sum(r["n_illegal"] for r in results)
    n_exc = sum(r["n_exceptions"] for r in results)
    n_search_lines = sum(r["n_search_lines"] for r in results)
    wins = sum(1 for r in results if r["our_win"] is True)
    decided = sum(1 for r in results if r["our_win"] is not None)
    walls = [r["wallclock_s"] for r in results]
    summaries = [r["summary"] for r in results if r["summary"]]
    end_pct = [s["end_at_atk_pct"] for s in summaries]
    value_calls = [s["value_calls"] for s in summaries]
    search_time = [s["search_time_s"] for s in summaries]
    total_time = [s["total_time_s"] for s in summaries]

    def mean(xs):
        return statistics.mean(xs) if xs else float("nan")

    out = dict(
        label=label, n=n, n_illegal=n_illegal, n_exceptions=n_exc,
        n_search_lines=n_search_lines, wins=wins, decided=decided,
        wr=(wins / decided if decided else float("nan")),
        wallclock_mean_s=mean(walls), wallclock_median_s=(statistics.median(walls) if walls else float("nan")),
        wallclock_max_s=(max(walls) if walls else float("nan")),
        episodes_with_summary=len(summaries),
        end_at_atk_pct_mean=mean(end_pct),
        value_calls_mean=mean(value_calls),
        search_time_s_mean=mean(search_time),
        total_time_s_mean=mean(total_time),
    )
    return out


def print_report(off_agg: dict, on_agg: dict) -> None:
    print("\n=== SMOKE REPORT (R1(a) SCOUT_R1A_ENDTURN) ===")
    print(f"{'metric':<28} {'OFF':>14} {'ON':>14} {'ON/OFF':>10}")
    rows = [
        ("games", "n", "{:d}"),
        ("illegal/exceptions", None, None),
        ("search-decisions total", "n_search_lines", "{:d}"),
        ("wallclock mean s/game", "wallclock_mean_s", "{:.1f}"),
        ("wallclock median s/game", "wallclock_median_s", "{:.1f}"),
        ("wallclock max s/game", "wallclock_max_s", "{:.1f}"),
        ("end_at_atk % (mean/ep)", "end_at_atk_pct_mean", "{:.1f}"),
        ("value_calls (mean/ep)", "value_calls_mean", "{:.1f}"),
        ("search_time s (mean/ep)", "search_time_s_mean", "{:.1f}"),
        ("total_time s (mean/ep)", "total_time_s_mean", "{:.1f}"),
    ]
    for label, key, fmt in rows:
        if key is None:
            print(f"{label:<28} {off_agg['n_illegal']}/{off_agg['n_exceptions']:>3} "
                  f"{' ':>9} {on_agg['n_illegal']}/{on_agg['n_exceptions']:>3}")
            continue
        a, b = off_agg[key], on_agg[key]
        ratio = (b / a) if a and a == a and a != 0 else float("nan")
        print(f"{label:<28} {fmt.format(a):>14} {fmt.format(b):>14} {ratio:>9.2f}x")
    print(f"\nWR (smoke, N={off_agg['n']}/{on_agg['n']} — НЕ доказательство, только liveness): "
          f"OFF {off_agg['wr']:.2f}  ON {on_agg['wr']:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="игр НА условие (OFF и ON отдельно)")
    ap.add_argument("--opp", default="meta:Dragapult")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--base-seed", type=int, default=90000)
    ap.add_argument("--samples", type=int, default=None, help="override SEARCH_CFG.samples (не трогает policy.json)")
    ap.add_argument("--budget-ms", type=int, default=None, help="override SEARCH_CFG.budget_ms")
    ap.add_argument("--episode-soft-s", type=float, default=None)
    ap.add_argument("--episode-hard-s", type=float, default=None)
    ap.add_argument("--out", default=None, help="путь для сырых результатов JSON")
    args = ap.parse_args()

    overrides = dict(samples=args.samples, budget_ms=args.budget_ms,
                      episode_soft_s=args.episode_soft_s, episode_hard_s=args.episode_hard_s)
    print(f"overrides (харнесс-сторона, policy.json не тронут): {overrides}")

    print("\n--- OFF (SCOUT_R1A_ENDTURN=0) ---")
    off_results = run_condition(False, args.n, args.opp, args.base_seed, args.workers, overrides)
    print("\n--- ON (SCOUT_R1A_ENDTURN=1) ---")
    on_results = run_condition(True, args.n, args.opp, args.base_seed, args.workers, overrides)

    off_agg = aggregate("OFF", off_results)
    on_agg = aggregate("ON", on_results)
    print_report(off_agg, on_agg)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(overrides=overrides, off=off_results, on=on_results,
                            off_agg=off_agg, on_agg=on_agg), f, indent=2)
        print(f"\nсырые данные: {args.out}")

    total_illegal = off_agg["n_illegal"] + on_agg["n_illegal"]
    total_exc = off_agg["n_exceptions"] + on_agg["n_exceptions"]
    if total_illegal or total_exc:
        print(f"\nFAIL: illegal={total_illegal} exceptions={total_exc}")
        return 1
    print("\nPASS: 0 illegal / 0 exceptions на обоих условиях.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
