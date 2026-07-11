"""pimc_crustle_smoke.py — S1 смок: PIMCAgent, перепараметрированный на Crustle
(вместо мёртвого Iono), пилотирует одну партию нативным движком и выдаёт
только легальные ходы.

Запуск:
    .venv/bin/python scripts/pimc_crustle_smoke.py

Что проверяем (S1, см. PLAN.md §S):
  1. crustle_prior(obs_dict) — переиспользует ЧЕМПИОНСКИЙ IL-агент v009a
     (LightGBM ranker + features_v009a.py) как rollout/decision prior.
  2. PIMCAgent(deck=CRUSTLE_DECK, prior_agent=make_prior_decide(crustle_prior),
     opponent_model="auto") с МАЛЕНЬКИМ бюджетом (K=2 мира, budget=0.3s) —
     смок должен быть быстрым, не глубоким.
  3. Партия Crustle-mirror через src/cg_native (нативный движок, macOS arm64,
     БЕЗ Docker). Каждый выбор агента валидируется на легальность ПЕРЕД
     передачей в battle_select.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
from cg_native.api import to_observation_class  # noqa: E402

from exit.pimc_agent import (  # noqa: E402
    PIMCAgent,
    CRUSTLE_DECK_APPROX,
)


# ---------------------------------------------------------------------------
# 1. crustle_prior — переиспользуем чемпионский IL-агент v009a как есть.
#    submissions/v009a_crustle/main.py уже реализует ровно тот контракт,
#    который нужен: agent(obs_dict) -> list[int], deck-select на первом
#    вызове (sel is None), IL-ranker + trivial fallback, никогда не падает.
# ---------------------------------------------------------------------------

def _load_v009a_agent():
    v009a_dir = os.path.join(ROOT, "submissions", "v009a_crustle")
    v009a_main = os.path.join(v009a_dir, "main.py")
    assert os.path.exists(v009a_main), f"не найден {v009a_main}"

    # features_v009a.py / il_crustle.txt / deck.csv / EN_Card_Data.csv лежат
    # рядом с main.py и резолвятся через _find_file(__file__-относительно),
    # поэтому нужно и добавить папку в sys.path (для `import features_v009a`),
    # и выставить mod.__file__ на реальный путь (для _DIR внутри main.py).
    if v009a_dir not in sys.path:
        sys.path.insert(0, v009a_dir)

    spec = importlib.util.spec_from_file_location("_v009a_crustle_main", v009a_main)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = v009a_main
    spec.loader.exec_module(mod)
    return mod


print("[1/4] Загружаю чемпионский IL-агент v009a (crustle_prior)...", flush=True)
_v009a_mod = _load_v009a_agent()
crustle_prior = _v009a_mod.agent  # Callable[[dict], list[int]]
print(f"      OK, my_deck len={len(_v009a_mod.my_deck)}, ranker={'loaded' if _v009a_mod._ranker else 'MISSING'}",
      flush=True)


# ---------------------------------------------------------------------------
# 2. PIMCAgent, перепараметрированный на Crustle с маленьким бюджетом
#    (смок — не гейт: цель доказать легальность, не силу игры).
# ---------------------------------------------------------------------------

from exit.pimc_agent import _make_prior_decide_wrapper  # noqa: E402


def make_prior_decide(prior_agent_fn):
    """Совместимый хелпер (в файле нет отдельной публичной make_prior_decide —
    _make_prior_decide_wrapper делает ровно то же самое: оборачивает
    Callable[[dict], list[int]] в _decide-совместимый Callable[[Observation], list[int]]).
    """
    return _make_prior_decide_wrapper(prior_agent_fn)


CRUSTLE_DECK = list(CRUSTLE_DECK_APPROX)

print("[2/4] Инстанцирую PIMCAgent(deck=Crustle, prior=crustle_prior, "
      "opponent_model='auto', budget=0.3, n_worlds_min=2)...", flush=True)
pimc = PIMCAgent(
    budget=0.3,          # маленький бюджет ради скорости смока (K~2 мира)
    rollout_depth=40,    # короткий rollout — тоже ради скорости
    n_worlds_min=2,      # K=2 детерминизации минимум на кандидата
    max_candidates=6,
    prior_agent=crustle_prior,
    deck=CRUSTLE_DECK,
    opponent_model="auto",
)
print("      OK", flush=True)


# ---------------------------------------------------------------------------
# 3. Одна партия Crustle-mirror нативно, с валидацией легальности каждого хода.
# ---------------------------------------------------------------------------

MAX_TURNS = 25

print(f"[3/4] Запускаю партию (Crustle mirror, нативный движок), до {MAX_TURNS} ходов...", flush=True)

obs_dict, start = battle_start(list(CRUSTLE_DECK), list(CRUSTLE_DECK))
assert obs_dict is not None, f"battle_start failed: errorType={start.errorType}"

decision_times: list[float] = []
n_illegal = 0
n_exceptions = 0
result = None

try:
    for turn in range(MAX_TURNS):
        obs = to_observation_class(obs_dict)
        if obs.current and obs.current.result != -1:
            result = obs.current.result
            print(f"  партия завершилась на ходу {turn}: result={result}", flush=True)
            break

        sel = obs_dict.get("select")

        t0 = time.perf_counter()
        try:
            choice = pimc.agent(obs_dict)
        except Exception as e:
            n_exceptions += 1
            print(f"  ХОД {turn}: ИСКЛЮЧЕНИЕ в pimc.agent(): {e!r}", flush=True)
            raise
        dt = time.perf_counter() - t0

        # --- Валидация легальности ---
        if sel is None:
            # deck-select: choice должен быть списком из 60 легальных card id
            legal = (isinstance(choice, list) and len(choice) == 60
                     and all(isinstance(c, int) for c in choice))
            n_options_desc = "deck-select (60 card ids)"
        else:
            opts = sel.get("option") or []
            n_opts = len(opts)
            min_c = int(sel.get("minCount") or 1)
            max_c = int(sel.get("maxCount") or 1)
            legal = (
                isinstance(choice, list)
                and all(isinstance(i, int) and 0 <= i < n_opts for i in choice)
                and min_c <= len(choice) <= max_c
                and len(set(choice)) == len(choice)
            )
            n_options_desc = f"n_opts={n_opts} min={min_c} max={max_c} ctx={sel.get('context')}"

        decision_times.append(dt)
        status = "OK" if legal else "ILLEGAL"
        if not legal:
            n_illegal += 1
        print(f"  ход {turn:2d}: {dt*1000:7.1f}ms  {status:8s}  choice={choice}  {n_options_desc}",
              flush=True)

        if not legal:
            print(f"  !!! НЕЛЕГАЛЬНЫЙ ВЫБОР на ходу {turn}, останавливаюсь", flush=True)
            break

        obs_dict = battle_select(choice)
    else:
        print(f"  партия НЕ завершилась за {MAX_TURNS} ходов (это ок для смока)", flush=True)
finally:
    battle_finish()


# ---------------------------------------------------------------------------
# 4. Отчёт: тайминги + доля решений через поиск vs fallback.
# ---------------------------------------------------------------------------

print("\n[4/4] ИТОГ", flush=True)
print(f"  ходов сыграно      : {len(decision_times)}")
print(f"  исключений         : {n_exceptions}")
print(f"  нелегальных выборов: {n_illegal}")
if decision_times:
    sorted_t = sorted(decision_times)
    med = sorted_t[len(sorted_t) // 2]
    print(f"  время/решение (мс) : медиана={med*1000:.1f}  макс={max(decision_times)*1000:.1f}  "
          f"мин={min(decision_times)*1000:.1f}  сумма={sum(decision_times):.2f}s")

stats = pimc.stats
print(f"  pimc.stats         : {stats}")
if stats:
    n_pimc = stats.get("decisions_pimc", 0)
    n_heur = stats.get("decisions_heuristic", 0)
    n_triv = stats.get("decisions_trivial", 0)
    total = n_pimc + n_heur + n_triv
    if total:
        print(f"  через ПОИСК (pimc)      : {n_pimc}/{total} ({100*n_pimc/total:.0f}%)")
        print(f"  через prior/heuristic   : {n_heur}/{total} ({100*n_heur/total:.0f}%)")
        print(f"  через trivial fallback  : {n_triv}/{total} ({100*n_triv/total:.0f}%)")

ok = (n_exceptions == 0 and n_illegal == 0 and len(decision_times) > 0)
print(f"\n{'СМОК ПРОЙДЕН' if ok else 'СМОК НЕ ПРОЙДЕН'}: exceptions={n_exceptions} illegal={n_illegal} "
      f"turns={len(decision_times)}")
sys.exit(0 if ok else 1)
