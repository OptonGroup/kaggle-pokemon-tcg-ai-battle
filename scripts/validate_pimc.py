"""Скрипт для валидации PIMC агента против v012_ref и v009a_crustle.

Запуск в Docker:
  docker run --rm -v $(pwd):/workspace -e PYTHONPATH=/workspace/src \
    pokemon-tcg-sim:latest python /workspace/scripts/validate_pimc.py

Или напрямую:
  python scripts/validate_pimc.py --games 20

Аргументы:
  --games N      Количество партий на матч (по умолчанию 20)
  --budget SEC   Бюджет PIMC на ход в секундах (по умолчанию 3.0)
  --quick        Быстрый тест: 4 партии, budget=1.0 (для проверки без ошибок)
  --vs12         Только против v012_ref
  --vs09         Только против v009a
"""
import sys
import os
import argparse
import time

# Пути
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import importlib.util

def load_agent_from_file(path: str, module_name: str):
    """Загружает агент из файла (имитирует Kaggle exec без __file__)."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    # Kaggle не устанавливает __file__ — эмулируем
    mod.__file__ = path
    spec.loader.exec_module(mod)
    return mod.agent


def load_deck_from_csv(path: str) -> list[int]:
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.append(int(line))
                except ValueError:
                    pass
    return ids


def run_match(agent0, agent1, deck0, deck1, n_games: int, label: str) -> dict:
    from simulate import run_tournament
    print(f"\n=== {label} ({n_games} партий) ===")
    t0 = time.time()
    result = run_tournament(agent0, agent1, deck0, deck1, n_games)
    elapsed = time.time() - t0
    print(f"  Wins agent0: {result['wins_0']}")
    print(f"  Wins agent1: {result['wins_1']}")
    print(f"  Draws: {result['draws']}")
    print(f"  Winrate agent0: {result['winrate_0']:.3f}")
    print(f"  Время: {elapsed:.1f}s ({elapsed/n_games:.1f}s/партия)")
    return result


def measure_rollout_speed():
    """Замер скорости rollout движка."""
    print("\n=== Замер скорости rollout ===")
    from cg.game import battle_start, battle_finish, battle_select
    from cg.api import to_observation_class, search_begin, search_step, search_end

    # Iono дека
    DECK = [
        265, 265, 265,
        268, 268, 268,
        269, 269, 269,
        270, 270, 270,
        271, 271, 271,
        1233, 1233, 1233, 1233,
        1227, 1227, 1227, 1227,
        1086, 1086, 1086,
        1121, 1121, 1121,
        1254, 1254, 1254,
        1097, 1097,
        1152, 1152,
        1110,
        1118,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4,
    ]

    import random

    # Запускаем реальную партию и считаем rollouts через search API
    obs_dict, start = battle_start(DECK, DECK)
    if obs_dict is None:
        print("  ОШИБКА: не удалось запустить партию")
        return

    try:
        obs = to_observation_class(obs_dict)

        # Пропускаем setup (deck selection, setup active)
        steps = 0
        while obs.select is None or obs.current is None or obs.current.turn < 2:
            if obs.select is None:
                # Deck selection — возвращаем деку
                sel = list(DECK)
            else:
                n = len(obs.select.option)
                min_c = obs.select.minCount
                max_c = min(obs.select.maxCount, n)
                if n == 0:
                    sel = []
                else:
                    count = max(1, min_c)
                    sel = random.sample(range(n), min(count, n))
            obs_dict = battle_select(sel)
            obs = to_observation_class(obs_dict)
            steps += 1
            if steps > 100:
                break

        if obs.current and obs.select:
            n_rollouts = 0
            t_start = time.time()
            t_limit = 5.0  # 5 секунд замер

            while time.time() - t_start < t_limit:
                my_idx = obs.current.yourIndex
                opp_idx = 1 - my_idx
                my = obs.current.players[my_idx]
                opp = obs.current.players[opp_idx]

                # Минимальные параметры для search_begin
                your_deck = list(DECK)[:my.deckCount]
                while len(your_deck) < my.deckCount:
                    your_deck.append(4)

                your_prize = [p.id if p is not None else 4 for p in my.prize]
                opp_deck = list(DECK)[:opp.deckCount]
                while len(opp_deck) < opp.deckCount:
                    opp_deck.append(4)

                opp_prize = [p.id if p is not None else 4 for p in opp.prize]
                opp_hand = [DECK[i % len(DECK)] for i in range(opp.handCount)]

                opp_active = []
                if opp.active and opp.active[0] is None:
                    opp_active = [265]

                try:
                    root = search_begin(
                        obs,
                        your_deck=your_deck,
                        your_prize=your_prize,
                        opponent_deck=opp_deck,
                        opponent_prize=opp_prize,
                        opponent_hand=opp_hand,
                        opponent_active=opp_active,
                        manual_coin=False,
                    )
                    # Случайный rollout
                    cur = root
                    steps_r = 0
                    while steps_r < 200:
                        if cur.observation.current and cur.observation.current.result != -1:
                            break
                        sel_r = cur.observation.select
                        if sel_r is None:
                            break
                        n_r = len(sel_r.option)
                        if n_r == 0:
                            break
                        choice = [random.randint(0, n_r-1)]
                        try:
                            cur = search_step(cur.searchId, choice)
                        except Exception:
                            break
                        steps_r += 1

                    search_end()
                    n_rollouts += 1

                except Exception as e:
                    break

            elapsed = time.time() - t_start
            if elapsed > 0:
                rps = n_rollouts / elapsed
                print(f"  Rollouts за {elapsed:.1f}с: {n_rollouts}")
                print(f"  Скорость: {rps:.1f} rollout/сек")
                print(f"  При бюджете 3с на ход: ~{int(rps * 3)} rollout/ход")

    finally:
        battle_finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--quick", action="store_true", help="Быстрый тест (4 партии, budget=1.0)")
    parser.add_argument("--vs12", action="store_true")
    parser.add_argument("--vs09", action="store_true")
    parser.add_argument("--speed-only", action="store_true", help="Только замер скорости")
    parser.add_argument("--mode", choices=["pimc", "hybrid", "heuristic", "guided"], default="guided",
                        help="Режим: guided=PIMC с guided rollout (v012), pimc=чистый PIMC, "
                             "hybrid=PIMC+эвристика, heuristic=только эвристика")
    args = parser.parse_args()

    if args.quick:
        args.games = 4
        args.budget = 1.0

    # Загружаем агента по режиму
    if args.mode == "guided":
        import pilot.pimc_guided as pimc_mod
        pimc_mod.TIME_BUDGET_SEC = args.budget
        pimc_agent = pimc_mod.agent
        agent_label = f"PIMC Guided v012 (budget={args.budget}s)"
    elif args.mode == "hybrid":
        import pilot.pimc_hybrid as pimc_mod
        pimc_mod.TIME_BUDGET_SEC = args.budget
        pimc_agent = pimc_mod.agent
        agent_label = f"PIMC Hybrid (budget={args.budget}s)"
    elif args.mode == "pimc":
        import pilot.pimc as pimc_mod
        pimc_mod.TIME_BUDGET_SEC = args.budget
        pimc_agent = pimc_mod.agent
        agent_label = f"PIMC Pure (budget={args.budget}s)"
    else:  # heuristic
        import pilot.pimc_hybrid as pimc_mod
        pimc_mod.PIMC_ENABLED = False
        pimc_agent = pimc_mod.agent
        agent_label = "Heuristic only (v012)"

    print(f"Агент: {agent_label}")

    # Замер скорости
    measure_rollout_speed()

    if args.speed_only:
        return

    run_vs12 = args.vs12 or (not args.vs12 and not args.vs09)
    run_vs09 = args.vs09 or (not args.vs12 and not args.vs09)

    pimc_deck = load_deck_from_csv(os.path.join(REPO, "submissions/v014_pimc_pilot/deck.csv"))

    results = {}

    # --- PIMC vs v012_ref ---
    if run_vs12:
        v12_path = os.path.join(REPO, "submissions/v012_ref/main.py")
        v12_agent = load_agent_from_file(v12_path, "v012_ref_main")
        v12_deck = load_deck_from_csv(os.path.join(REPO, "submissions/v012_ref/deck.csv"))

        r = run_match(
            pimc_agent, v12_agent, pimc_deck, v12_deck,
            args.games, f"{agent_label} vs v012_ref"
        )
        results["agent_vs_v12"] = r

    # --- Агент vs v009a_crustle ---
    if run_vs09:
        v09_path = os.path.join(REPO, "submissions/v009a_crustle/main.py")
        if os.path.exists(v09_path):
            sys.path.insert(0, os.path.join(REPO, "submissions/v009a_crustle"))
            v09_agent = load_agent_from_file(v09_path, "v009a_main")
            v09_deck = load_deck_from_csv(os.path.join(REPO, "submissions/v009a_crustle/deck.csv"))

            r = run_match(
                pimc_agent, v09_agent, pimc_deck, v09_deck,
                args.games, f"{agent_label} vs v009a_crustle"
            )
            results["agent_vs_v09"] = r
        else:
            print(f"\nv009a_crustle не найден: {v09_path}")

    # Итоговый отчёт
    print("\n" + "="*50)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("="*50)
    for key, r in results.items():
        label = f"vs v012_ref" if "v12" in key else f"vs v009a"
        print(f"  {agent_label} {label}: WR={r['winrate_0']:.3f} ({r['wins_0']}W-{r['wins_1']}L-{r['draws']}D)")

    # Статистика
    stats = pimc_mod.get_stats()
    print(f"\nСтатистика агента:")
    for k, v in stats.items():
        if not isinstance(v, list):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
