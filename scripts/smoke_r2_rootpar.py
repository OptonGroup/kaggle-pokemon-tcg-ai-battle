#!/usr/bin/env python3
"""smoke_r2_rootpar.py — минимальная живость submissions/scout_r2_rootpar (R2:
root-parallel + бюджет) на РЕАЛЬНЫХ партиях cg_native. НЕ мера WR (партий мало) —
только: 0 исключений/нелегальных ходов, при флаге ВЫКЛ трейс структурно совпадает с
эталоном (ни одной строки [r2-rootpar]), при флаге ВКЛ (K>1) видно исполнение
root-parallel ветки в трейсе.

Один процесс, последовательно (без ProcessPoolExecutor) — по требованию задачи
("проверка живости... один процесс") и из-за жёсткого бюджета ресурсов, пока рядом
идёт долгий гейт. Конфиг поиска урезан через ENV-оверрайды (B) ДО импорта модуля
кандидата (policy.json НЕ трогаем) — только ради трактабельности смоука, не имеет
отношения к тому, что реально измеряет root-parallel.

Запуск:
    nice -n 15 .venv/bin/python scripts/smoke_r2_rootpar.py
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import random
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

CANDIDATE_DIR = os.path.join(ROOT, "submissions", "scout_r2_rootpar")
OPP_DECK_PATH = os.path.join(ROOT, "decks", "meta", "Dragapult.csv")
MAX_MOVES_PER_GAME = 600

_SUMMARY_RE = re.compile(
    r"\[scout\] d(\d+) t(\S+) ctx(\d+) n=(\d+) src=(\w+) ([\d.]+)ms "
    r"spent=([\d.]+)s s/h=(\d+)/(\d+)")


def _load_deck(path):
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip()]


def _load_module(sub_dir: str, tag: str):
    main_path = os.path.join(sub_dir, "main.py")
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    mod_name = f"_smoke_r2_{tag}_{id(object())}"
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


def play_one(cand_agent, cand_deck, our_first: bool, seed: int) -> dict:
    from cg_native.game import battle_start, battle_finish, battle_select

    opp_deck = _load_deck(OPP_DECK_PATH)
    random.seed(seed)

    if our_first:
        deck0, deck1 = list(cand_deck), list(opp_deck)
        agent0, agent1 = cand_agent, greedy_meta_agent
    else:
        deck0, deck1 = list(opp_deck), list(cand_deck)
        agent0, agent1 = greedy_meta_agent, cand_agent
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
                except Exception as exc:
                    n_exceptions += 1
                    print(f"    !! agent exception: {type(exc).__name__}: {exc}", file=sys.__stderr__)
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
        try:
            cand_agent({"select": None})  # форсируем flush episode-лога кандидата
        except Exception:
            pass
    elapsed = time.perf_counter() - t0
    log_text = stderr_buf.getvalue()

    n_exc_log = log_text.count("EXC ")
    n_search_lines = log_text.count(" src=search ")
    n_heur_lines = log_text.count(" src=heur ")
    n_decision_lines = log_text.count("[scout] d")
    n_rootpar_lines = log_text.count("[r2-rootpar]")

    return dict(
        our_first=our_first, seed=seed, moves=move_count, wallclock_s=elapsed,
        winner=winner, our_win=(winner == our_side) if winner is not None else None,
        n_illegal=n_illegal, n_exceptions=n_exceptions, n_exc_log=n_exc_log,
        n_search_lines=n_search_lines, n_heur_lines=n_heur_lines,
        n_decision_lines=n_decision_lines, n_rootpar_lines=n_rootpar_lines,
        log_text=log_text,
    )


def run_condition(label: str, rootpar_env: str | None, n_games: int, base_seed: int,
                   overrides: dict) -> list:
    print(f"\n--- {label} (SCOUT_R2_ROOTPAR={rootpar_env!r}) ---")
    if rootpar_env is None:
        os.environ.pop("SCOUT_R2_ROOTPAR", None)
    else:
        os.environ["SCOUT_R2_ROOTPAR"] = rootpar_env
    for k, v in overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)

    # Свежий инстанс модуля ПОСЛЕ выставления ENV (оверрайды (B) читаются при импорте;
    # ROOTPAR (A) читается заново на каждый вызов, но свежий модуль всё равно нужен,
    # чтобы _SPENT/эпизодный бюджет не тёк между условиями).
    mod = _load_module(CANDIDATE_DIR, label)
    cand_deck = mod.read_deck()

    results = []
    for i in range(n_games):
        seed = base_seed + i
        r = play_one(mod.agent, cand_deck, our_first=(i % 2 == 0), seed=seed)
        results.append(r)
        print(f"  game {i} our_first={i % 2 == 0} moves={r['moves']:4d} "
              f"wall={r['wallclock_s']:6.2f}s win={r['our_win']} "
              f"illegal={r['n_illegal']} exc={r['n_exceptions']}(agent)/{r['n_exc_log']}(log) "
              f"decisions={r['n_decision_lines']} search={r['n_search_lines']} "
              f"heur={r['n_heur_lines']} rootpar_lines={r['n_rootpar_lines']}")
    return results


def main() -> int:
    base = dict(SCOUT_R2_SAMPLES=3, SCOUT_R2_BUDGET_MS=1500,
                SCOUT_R2_EP_SOFT=60, SCOUT_R2_EP_HARD=90)
    print(f"overrides (только ради скорости смоука, policy.json не тронут): {base}")

    off_results = run_condition("OFF", None, n_games=2, base_seed=71001, overrides=base)
    on_results = run_condition("ON(K=3)", "3", n_games=2, base_seed=71101, overrides=base)

    all_results = off_results + on_results
    total_illegal = sum(r["n_illegal"] for r in all_results)
    total_exc = sum(r["n_exceptions"] for r in all_results) + sum(r["n_exc_log"] for r in all_results)
    off_rootpar_lines = sum(r["n_rootpar_lines"] for r in off_results)
    on_rootpar_lines = sum(r["n_rootpar_lines"] for r in on_results)
    on_search_lines = sum(r["n_search_lines"] for r in on_results)

    print("\n=== SMOKE REPORT (R2 root-parallel) ===")
    print(f"games total: {len(all_results)} (OFF={len(off_results)} ON={len(on_results)})")
    print(f"illegal moves total: {total_illegal}")
    print(f"exceptions total (agent-level + EXC-в-логе): {total_exc}")
    print(f"[r2-rootpar] строк при ВЫКЛЮЧЕННОМ флаге: {off_rootpar_lines} (ожидание: 0)")
    print(f"[r2-rootpar] строк при ВКЛЮЧЁННОМ флаге (K=3): {on_rootpar_lines} "
          f"из search-decisions={on_search_lines} (ожидание: >0, если поиск вообще срабатывал)")
    for label, rs in (("OFF", off_results), ("ON", on_results)):
        walls = [r["wallclock_s"] for r in rs]
        moves = [r["moves"] for r in rs]
        print(f"{label}: wallclock/game = {walls}, moves/game = {moves}")

    ok = (total_illegal == 0 and total_exc == 0 and off_rootpar_lines == 0
          and (on_rootpar_lines > 0 or on_search_lines == 0))
    print("\n" + ("PASS" if ok else "FAIL") +
          ": 0 illegal, 0 exceptions, OFF без [r2-rootpar], ON с [r2-rootpar] "
          "(если вообще были search-решения).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
