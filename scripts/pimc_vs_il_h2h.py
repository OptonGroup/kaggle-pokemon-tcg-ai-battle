"""pimc_vs_il_h2h.py — S4a H2H-смок: PIMC(Crustle, S3 bank-manager) vs
плоский чемпионский IL v009a, Crustle-mirror, нативный движок (macOS arm64,
src/cg_native/), сторона чередуется.

Это ПЕРВАЯ проверка всей ставки на поиск (PLAN.md §S, S4a): если PIMC не
обыгрывает СВОЙ ЖЕ rollout-prior (v009a), поиск не добавляет ничего — это
повтор провала старого 1-ply поиска (поиск ≈ его собственная rollout-policy).
Дешёвый ранний сигнал (N игр), НЕ полный 300-игровой S4-гейт.

Игрок A = PIMCAgent(deck=Crustle, prior_agent=v009a.agent, opponent_model="auto")
          с адаптивным S3 bank-manager (см. src/exit/pimc_agent.py).
Игрок B = чистый v009a.agent (тот же submissions/v009a_crustle/main.py), без поиска.
Обе стороны играют Crustle-дек (mirror) — изолируем вклад ПОИСКА поверх prior.

Запуск:
    .venv/bin/python scripts/pimc_vs_il_h2h.py [--games N] [--seed BASE_SEED]

Печатает промежуточный счёт каждые ~5 игр, финальный отчёт: winrate PIMC,
Wilson 95% LB, медиану/макс времени на решение (PIMC), средний/макс расход
банка (self._bank_spent, см. bank_stats) на партию — критично: НИ ОДНА
партия не должна подойти к 600s (даже к резервным 300s=BANK_TOTAL*SAFETY).
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
from cg_native.api import to_observation_class  # noqa: E402

from exit.pimc_agent import PIMCAgent, CRUSTLE_DECK_APPROX  # noqa: E402

CRUSTLE_DECK = list(CRUSTLE_DECK_APPROX)

MAX_MOVES_PER_GAME = 2000  # safety cap (реальные партии много короче) — избегаем runaway loop


# ---------------------------------------------------------------------------
# Wilson 95% CI (тот же формула что src/sim_harness.wilson_ci — переиспользуем
# формулу, не импортируем sim_harness напрямую: он хардкодит `from cg.game
# import ...` = Linux-only libcg.so, что упадёт на macOS ARM).
# ---------------------------------------------------------------------------

def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Загрузка чемпионского IL-агента v009a (переиспользуем 1:1, как в
# scripts/pimc_crustle_smoke.py — НЕ реимплементируем IL).
# ---------------------------------------------------------------------------

def _load_v009a_agent():
    v009a_dir = os.path.join(ROOT, "submissions", "v009a_crustle")
    v009a_main = os.path.join(v009a_dir, "main.py")
    assert os.path.exists(v009a_main), f"не найден {v009a_main}"
    if v009a_dir not in sys.path:
        sys.path.insert(0, v009a_dir)
    spec = importlib.util.spec_from_file_location("_v009a_crustle_main_h2h", v009a_main)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = v009a_main
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Валидация легальности выбора (тот же контракт что и pimc_crustle_smoke.py).
# ---------------------------------------------------------------------------

def _is_legal(choice, sel: dict | None) -> bool:
    if sel is None:
        return isinstance(choice, list) and len(choice) == 60 and all(isinstance(c, int) for c in choice)
    opts = sel.get("option") or []
    n_opts = len(opts)
    min_c = int(sel.get("minCount") if sel.get("minCount") is not None else 1)
    max_c = int(sel.get("maxCount") if sel.get("maxCount") is not None else 1)
    return (
        isinstance(choice, list)
        and all(isinstance(i, int) and 0 <= i < n_opts for i in choice)
        and min_c <= len(choice) <= max_c
        and len(set(choice)) == len(choice)
    )


# ---------------------------------------------------------------------------
# Одна партия. pimc_first=True -> PIMC играет player0 (ходит первым по
# порядку деков в battle_start; фактический coin-flip первого хода —
# внутренняя логика движка, не контролируется отсюда, но deck0/1 назначение
# чередуем для полной симметрии раннера).
# ---------------------------------------------------------------------------

def run_one_game(pimc_first: bool, seed: int, il_agent_fn) -> dict:
    random.seed(seed)

    pimc = PIMCAgent(
        deck=CRUSTLE_DECK,
        prior_agent=il_agent_fn,
        # S4a — ЧЕСТНАЯ изоляция вклада поиска: в зеркале оппонент ТОЧНО Crustle,
        # поэтому модель оппонента должна быть "crustle", а не "auto" (иначе PIMC
        # планирует против мета-поля Alakazam/Archaludon и это ломает mirror-тест).
        opponent_model="crustle",
    )

    if pimc_first:
        agent0, agent1 = pimc.agent, il_agent_fn
    else:
        agent0, agent1 = il_agent_fn, pimc.agent

    obs_dict, start = battle_start(list(CRUSTLE_DECK), list(CRUSTLE_DECK))
    if obs_dict is None:
        raise RuntimeError(f"battle_start failed: errorType={start.errorType}")

    move_count = 0
    pimc_decision_times: list[float] = []
    n_illegal = 0
    n_exceptions = 0
    winner = None

    try:
        while move_count < MAX_MOVES_PER_GAME:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                winner = obs.current.result  # 0/1 winner, 2 = draw
                break

            your_index = obs.current.yourIndex if obs.current else 0
            is_pimc_turn = (your_index == 0) == pimc_first
            agent_fn = agent0 if your_index == 0 else agent1

            sel = obs_dict.get("select")
            t0 = time.perf_counter()
            try:
                choice = agent_fn(obs_dict)
            except Exception as e:
                n_exceptions += 1
                print(f"    !!! ИСКЛЮЧЕНИЕ (side={your_index}, pimc_turn={is_pimc_turn}): {e!r}", flush=True)
                raise
            dt = time.perf_counter() - t0
            if is_pimc_turn:
                pimc_decision_times.append(dt)

            if not _is_legal(choice, sel):
                n_illegal += 1
                print(f"    !!! НЕЛЕГАЛЬНЫЙ ВЫБОР (side={your_index}, pimc_turn={is_pimc_turn}): "
                      f"choice={choice} sel={sel}", flush=True)
                break

            obs_dict = battle_select(choice)
            move_count += 1
    finally:
        battle_finish()

    pimc_side = 0 if pimc_first else 1
    if winner is None:
        outcome = "aborted"
    elif winner == pimc_side:
        outcome = "pimc_win"
    elif winner == 2:
        outcome = "draw"
    else:
        outcome = "il_win"

    bank = pimc.bank_stats
    return dict(
        outcome=outcome,
        winner=winner,
        moves=move_count,
        pimc_decision_times=pimc_decision_times,
        bank_spent=bank["spent_sec"],
        wall_clock=bank["wall_clock_sec"],
        decisions_made=bank["decisions_made"],
        n_search=bank["n_search_decisions"],
        n_cheap=bank["n_cheap_decisions"],
        n_illegal=n_illegal,
        n_exceptions=n_exceptions,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n_games = args.games
    base_seed = args.seed

    print(f"[1/2] Загружаю чемпионский IL-агент v009a (общий для prior И для B-игрока)...", flush=True)
    v009a_mod = _load_v009a_agent()
    il_agent = v009a_mod.agent
    print(f"      OK, my_deck len={len(v009a_mod.my_deck)}, ranker={'loaded' if v009a_mod._ranker else 'MISSING'}",
          flush=True)

    print(f"[2/2] H2H: PIMC(Crustle, S3 bank-manager) vs v009a IL (Crustle mirror), "
          f"N={n_games} игр (сторона чередуется, base_seed={base_seed})\n", flush=True)

    pimc_wins = 0
    il_wins = 0
    draws = 0
    aborted = 0
    all_pimc_times: list[float] = []
    bank_spent_per_game: list[float] = []
    wall_clock_per_game: list[float] = []
    moves_per_game: list[int] = []
    total_illegal = 0
    total_exceptions = 0

    t_run_start = time.time()

    for i in range(n_games):
        pimc_first = (i % 2 == 0)
        seed = base_seed + i
        t0 = time.time()
        res = run_one_game(pimc_first, seed, il_agent)
        dt_game = time.time() - t0

        if res["outcome"] == "pimc_win":
            pimc_wins += 1
        elif res["outcome"] == "il_win":
            il_wins += 1
        elif res["outcome"] == "draw":
            draws += 1
        else:
            aborted += 1

        all_pimc_times.extend(res["pimc_decision_times"])
        bank_spent_per_game.append(res["bank_spent"])
        wall_clock_per_game.append(res["wall_clock"])
        moves_per_game.append(res["moves"])
        total_illegal += res["n_illegal"]
        total_exceptions += res["n_exceptions"]

        side = "P0(first)" if pimc_first else "P1(second)"
        import exit.pimc_agent as _pa  # noqa
        _pcalls, _pexc = _pa._PRIOR_CALL_COUNT, _pa._PRIOR_EXC_COUNT
        _prate = (_pexc / _pcalls * 100.0) if _pcalls else 0.0
        print(f"  игра {i+1:3d}/{n_games}  PIMC={side}  outcome={res['outcome']:9s}  "
              f"moves={res['moves']:4d}  decisions={res['decisions_made']:3d} "
              f"(search={res['n_search']}, cheap={res['n_cheap']})  "
              f"bank_spent={res['bank_spent']:6.1f}s  wall={dt_game:6.1f}s  "
              f"illegal={res['n_illegal']}  exc={res['n_exceptions']}  "
              f"rollout_prior_exc={_prate:.1f}%({_pexc}/{_pcalls})", flush=True)

        if (i + 1) % 5 == 0 or (i + 1) == n_games:
            decisive = pimc_wins + il_wins
            wr = pimc_wins / decisive if decisive else float("nan")
            lo, hi = wilson_ci(pimc_wins, decisive) if decisive else (float("nan"), float("nan"))
            print(f"  --- промежуточно после {i+1} игр: PIMC {pimc_wins}-{il_wins} "
                  f"(draws={draws}, aborted={aborted})  winrate={wr:.3f}  "
                  f"Wilson95%=[{lo:.3f},{hi:.3f}] ---\n", flush=True)

    t_run_total = time.time() - t_run_start

    print("\n" + "=" * 70)
    print("ИТОГ S4a H2H: PIMC(Crustle, S3) vs v009a IL flat (Crustle mirror)")
    print("=" * 70)
    print(f"  игр сыграно        : {n_games}  (base_seed={base_seed}, суммарное время рана={t_run_total:.1f}s)")
    print(f"  PIMC wins          : {pimc_wins}")
    print(f"  IL   wins          : {il_wins}")
    print(f"  draws              : {draws}")
    print(f"  aborted (>{MAX_MOVES_PER_GAME} moves): {aborted}")

    decisive = pimc_wins + il_wins
    if decisive:
        wr = pimc_wins / decisive
        lo, hi = wilson_ci(pimc_wins, decisive)
        print(f"  winrate PIMC (excl. draws): {wr:.4f}   Wilson 95% CI: [{lo:.4f}, {hi:.4f}]")
    else:
        wr, lo, hi = float("nan"), float("nan"), float("nan")
        print("  winrate PIMC: N/A (0 decisive games)")

    if all_pimc_times:
        st = sorted(all_pimc_times)
        med = st[len(st) // 2]
        print(f"\n  время/решение PIMC (мс): медиана={med*1000:.1f}  макс={max(st)*1000:.1f}  "
              f"мин={min(st)*1000:.1f}  n={len(st)}")
    if bank_spent_per_game:
        print(f"  bank_spent (PIMC._bank_spent) за партию: "
              f"среднее={statistics.mean(bank_spent_per_game):.1f}s  "
              f"медиана={statistics.median(bank_spent_per_game):.1f}s  "
              f"макс={max(bank_spent_per_game):.1f}s  (лимит=600s, резерв SAFETY=300s)")
        over_reserve = sum(1 for b in bank_spent_per_game if b > 300.0)
        over_bank = sum(1 for b in bank_spent_per_game if b > 600.0)
        print(f"  партий с bank_spent > 300s (резерв): {over_reserve}/{n_games}")
        print(f"  партий с bank_spent > 600s (ТАЙМАУТ): {over_bank}/{n_games}  "
              f"{'<-- КРИТИЧНО, ловим таймаут!' if over_bank else '(OK, ни одна не подошла к лимиту)'}")
    if wall_clock_per_game:
        print(f"  wall-clock (полное время партии, для справки): "
              f"среднее={statistics.mean(wall_clock_per_game):.1f}s  макс={max(wall_clock_per_game):.1f}s")
    if moves_per_game:
        print(f"  ходов/партию: медиана={statistics.median(moves_per_game):.0f}  "
              f"среднее={statistics.mean(moves_per_game):.1f}  макс={max(moves_per_game)}")

    print(f"\n  нелегальных выборов (всего): {total_illegal}")
    print(f"  исключений (всего)         : {total_exceptions}")

    # Диагностика деградации rollout-политики: доля вызовов IL-prior, упавших в
    # исключение (→ молчаливый откат на тупую эвристику внутри роллаута).
    import exit.pimc_agent as _pa  # noqa
    _calls, _exc = _pa._PRIOR_CALL_COUNT, _pa._PRIOR_EXC_COUNT
    _rate = (_exc / _calls * 100.0) if _calls else 0.0
    print(f"  rollout-prior: вызовов={_calls}  исключений={_exc}  ({_rate:.1f}% → откат на тупую эвристику)")

    print("\n--- Вердикт по гипотезе (PLAN.md §S4a) ---")
    if decisive:
        if lo > 0.55 or wr >= 0.55:
            print(f"  winrate={wr:.3f} (LB={lo:.3f}) >= ~0.55 -> гипотеза ЖИВА, зелёный свет на полный S4 (300g).")
        elif wr >= 0.45:
            print(f"  winrate={wr:.3f} (LB={lo:.3f}) ~ 0.50 -> поиск НЕ превосходит свой prior заметно; "
                  f"крутить K/глубину/бюджет до полного гейта.")
        else:
            print(f"  winrate={wr:.3f} (LB={lo:.3f}) < ~0.45 -> ТРЕВОГА, поиск ВРЕДИТ.")
    else:
        print("  недостаточно решительных партий для вердикта.")

    ok = (total_exceptions == 0 and total_illegal == 0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
