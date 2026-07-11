"""WP2 — Харнесс массовых прогонов агент × пул оппонентов.

Запускает N партий агент-кандидат × каждый оппонент из пула через multiprocessing
(НЕ threading — движок не thread-safe, threading даёт SIGSEGV).

Возвращает:
  - winrates: dict[opponent_name → float] — винрейт кандидата vs каждого оппонента
  - trajectories: list[Trajectory] — (obs_dict, chosen, reward, player_id) по каждому решению
  - games_per_sec: float — пропускная способность

Принципы:
  - Каждый worker-процесс работает независимо (нет shared state между процессами).
  - Агенты передаются как ПУТИ к main.py + deck.csv, а не как объекты Python
    (чтобы избежать pickle-проблем с замыканиями и ctypes-объектами движка).
  - Только multiprocessing.Pool или concurrent.futures.ProcessPoolExecutor.
    threading.Thread — ЗАПРЕЩЕНО.
  - Движок НЕ импортируется на уровне модуля — только внутри worker-функций.
  - Каждый worker-процесс загружает движок ровно один раз при старте (import cg).

Пример использования (в Docker или Kaggle notebook):
  from src.exit.opponents import list_opponents, Agent
  from src.exit.harness import run_games

  pool = list_opponents(names=["crustle_il", "iono_heur"])
  candidate = Agent(name="my_pimc", deck=MY_DECK, policy=lambda _: [],
                    main_py_path="/workspace/submissions/v012_ref/main.py")
  result = run_games(candidate, pool, n=200, workers=4)
  print(result.winrates)           # {"crustle_il": 0.61, "iono_heur": 0.55}
  print(result.games_per_sec)      # ~2.5
  print(len(result.trajectories))  # ~24000 записей (200 партий × ~120 шагов)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .opponents import Agent

# Максимальное число шагов в одной партии (защита от зависания)
MAX_GAME_STEPS: int = 2000

# Минимально допустимое число workers
MIN_WORKERS: int = 1

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Типы данных
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """Одна пара (obs, action) из реальной партии.

    Используется для BC-обучения в WP4.

    Attributes:
        obs_dict:      Необработанный obs_dict (dict) из движка.
        chosen:        list[int] — выбранные индексы опций агентом-кандидатом.
        n_options:     Количество доступных опций (len(obs.select.option)).
        final_reward:  Финальная награда с точки зрения кандидата: +1 победа, 0 ничья, -1 поражение.
        player_id:     Индекс кандидата в этой партии (0 или 1).
        game_id:       Порядковый номер партии (для отладки).
        opponent_name: Имя оппонента (для фильтрации по оппоненту).
    """
    obs_dict: dict
    chosen: list[int]
    n_options: int
    final_reward: float
    player_id: int
    game_id: int = 0
    opponent_name: str = ""


@dataclass
class GameResult:
    """Результат одной партии.

    Attributes:
        winner:          0 или 1 (индекс победителя), -1 = ничья.
        n_steps:         Число шагов (battle_select-вызовов).
        wall_time_s:     Время партии в секундах.
        opponent_name:   Имя оппонента.
        game_id:         Порядковый номер.
        cand_player_idx: Индекс кандидата в этой партии (0 или 1). Нужен для
                         правильного подсчёта побед в _aggregate_results.
        trajectories:    Траектории кандидата из этой партии (если collect_trajectories=True).
    """
    winner: int
    n_steps: int
    wall_time_s: float
    opponent_name: str = ""
    game_id: int = 0
    cand_player_idx: int = 0
    trajectories: list[Trajectory] = field(default_factory=list)


@dataclass
class HarnessResult:
    """Итоговый результат прогона run_games().

    Attributes:
        winrates:         {opponent_name → winrate_candidate} (исключая ничьи).
        game_results:     Все GameResult-ы для детальной аналитики.
        trajectories:     Объединённые траектории кандидата из всех партий.
        games_per_sec:    Средняя пропускная способность.
        wilson_lo:        Wilson 95% lower bound на общий winrate (по всем оппонентам).
        wilson_hi:        Wilson 95% upper bound.
        total_games:      Суммарное число партий.
        total_wins:       Суммарное число побед кандидата.
        trajectory_path:  Путь к файлу с сохранёнными траекториями (JSONL), "" если не сохранялись.
    """
    winrates: dict[str, float] = field(default_factory=dict)
    game_results: list[GameResult] = field(default_factory=list)
    trajectories: list[Trajectory] = field(default_factory=list)
    games_per_sec: float = 0.0
    wilson_lo: float = 0.0
    wilson_hi: float = 1.0
    total_games: int = 0
    total_wins: int = 0
    trajectory_path: str = ""


# ---------------------------------------------------------------------------
# Worker-функция (выполняется в отдельном процессе)
# ---------------------------------------------------------------------------

@dataclass
class _WorkerArgs:
    """Аргументы для worker-процесса.

    Все поля сериализуемы через pickle (нет ctypes, нет lambda).
    None в *_main_py означает inline random policy (агент без файла).
    """
    candidate_main_py: str | None    # путь к main.py кандидата; None = inline random
    candidate_deck: list[int]        # 60 card IDs кандидата
    opponent_main_py: str | None     # путь к main.py оппонента; None = inline random
    opponent_deck: list[int]         # 60 card IDs оппонента
    opponent_name: str               # имя оппонента (для логов)
    n_games: int                     # число партий в этом chunk
    game_id_offset: int              # для уникальных game_id и чередования сторон
    seed: int | None                 # random seed
    collect_trajectories: bool       # собирать ли траектории


def _worker_fn(args: _WorkerArgs) -> list[GameResult]:
    """Выполняется в отдельном worker-процессе.

    Загружает движок, агентов, прогоняет n_games партий, возвращает GameResult-ы.

    ВАЖНО: импортируем cg.* здесь — не на уровне модуля (Linux-only libcg.so).
    """
    import random
    import sys

    # --- Настройка путей ---
    src_dir = os.path.join(_REPO, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # --- Ленивый импорт движка (Linux x86_64 only) ---
    try:
        from cg.game import battle_start, battle_finish, battle_select
        from cg.api import to_observation_class
    except OSError as exc:
        raise RuntimeError(
            "Не удалось загрузить libcg.so. "
            "Запускайте в Docker linux/amd64 или Kaggle notebook. "
            f"Ошибка: {exc}"
        ) from exc

    # --- Загрузка агентов ---
    from exit.opponents import _load_agent_module, _make_inline_random_policy

    if args.candidate_main_py is not None:
        cand_mod = _load_agent_module(args.candidate_main_py, "_cand_wp2")
        candidate_agent = cand_mod.agent
    else:
        candidate_agent = _make_inline_random_policy(args.candidate_deck)

    if args.opponent_main_py is not None:
        opp_mod = _load_agent_module(args.opponent_main_py, "_opp_wp2")
        opponent_agent = opp_mod.agent
    else:
        opponent_agent = _make_inline_random_policy(args.opponent_deck)

    results: list[GameResult] = []

    for i in range(args.n_games):
        gid = args.game_id_offset + i

        if args.seed is not None:
            random.seed(args.seed + gid)

        # Чередование: чётный gid → кандидат идёт первым (player 0)
        cand_is_p0 = (gid % 2 == 0)
        if cand_is_p0:
            deck0, deck1 = args.candidate_deck, args.opponent_deck
            agent0, agent1 = candidate_agent, opponent_agent
            cand_player_idx = 0
        else:
            deck0, deck1 = args.opponent_deck, args.candidate_deck
            agent0, agent1 = opponent_agent, candidate_agent
            cand_player_idx = 1

        # --- Старт партии ---
        t0 = time.perf_counter()
        try:
            obs_dict, start_data = battle_start(deck0, deck1)
        except Exception:
            # Не удалось начать партию — засчитываем поражение кандидата
            results.append(GameResult(
                winner=1 - cand_player_idx,
                n_steps=0,
                wall_time_s=0.0,
                opponent_name=args.opponent_name,
                game_id=gid,
                cand_player_idx=cand_player_idx,
            ))
            continue

        if obs_dict is None:
            # Движок отклонил деку
            results.append(GameResult(
                winner=1 - cand_player_idx,
                n_steps=0,
                wall_time_s=0.0,
                opponent_name=args.opponent_name,
                game_id=gid,
                cand_player_idx=cand_player_idx,
            ))
            try:
                battle_finish()
            except Exception:
                pass
            continue

        # --- Игровой цикл ---
        step = 0
        winner = -1
        cand_step_data: list[tuple[dict, list[int], int]] = []

        try:
            while True:
                obs = to_observation_class(obs_dict)

                # Проверка завершения партии
                if obs.current and obs.current.result != -1:
                    winner = obs.current.result
                    break

                # Защита от зависания
                if step >= MAX_GAME_STEPS:
                    winner = -1  # ничья по лимиту шагов
                    break

                your_index = obs.current.yourIndex if obs.current else 0
                agent_fn = agent0 if your_index == 0 else agent1
                is_cand_turn = (your_index == cand_player_idx)

                # Снимок obs_dict ДО вызова агента — для траектории
                if args.collect_trajectories and is_cand_turn:
                    obs_snapshot = obs_dict  # новый dict придёт от battle_select
                    n_opts = 0
                    if obs.select is not None and obs.select.option is not None:
                        n_opts = len(obs.select.option)
                else:
                    obs_snapshot = None
                    n_opts = 0

                # Вызов агента с защитой от исключений
                try:
                    chosen = agent_fn(obs_dict)
                except Exception:
                    # Фолбэк: выбираем минимальное число первых опций
                    if obs.select is not None and obs.select.option is not None:
                        min_c = int(obs.select.minCount or 1)
                        chosen = list(range(min(min_c, len(obs.select.option))))
                    else:
                        chosen = []

                # Сохраняем шаг траектории кандидата
                if args.collect_trajectories and is_cand_turn and obs_snapshot is not None:
                    cand_step_data.append((obs_snapshot, list(chosen), n_opts))

                obs_dict = battle_select(chosen)
                step += 1

        finally:
            try:
                battle_finish()
            except Exception:
                pass

        wall_time = time.perf_counter() - t0

        # Награда с точки зрения кандидата
        if winner == cand_player_idx:
            final_reward = 1.0
        elif winner == -1:
            final_reward = 0.0
        else:
            final_reward = -1.0

        # Формируем траектории
        trajs: list[Trajectory] = []
        if args.collect_trajectories:
            for obs_d, ch, n_opts in cand_step_data:
                trajs.append(Trajectory(
                    obs_dict=obs_d,
                    chosen=ch,
                    n_options=n_opts,
                    final_reward=final_reward,
                    player_id=cand_player_idx,
                    game_id=gid,
                    opponent_name=args.opponent_name,
                ))

        results.append(GameResult(
            winner=winner,
            n_steps=step,
            wall_time_s=wall_time,
            opponent_name=args.opponent_name,
            game_id=gid,
            cand_player_idx=cand_player_idx,
            trajectories=trajs,
        ))

    return results


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------

def run_games(
    candidate: "Agent",
    opponents: list["Agent"],
    n: int = 200,
    workers: int = 4,
    seed: int | None = 42,
    collect_trajectories: bool = True,
) -> HarnessResult:
    """Запускает N партий candidate × каждый оппонент через multiprocessing.

    Args:
        candidate:            Агент-кандидат (тестируемый агент). Обязателен
                              атрибут ``main_py_path`` (str) — путь к main.py.
                              Если агент in-memory (PIMC/Pilot), предварительно
                              сохраните через ``pilot.export_to_submission()``.
        opponents:            Список оппонентов из list_opponents(). Оппоненты
                              с ``main_py_path=None`` используют inline random policy.
        n:                    Число партий на каждого оппонента.
        workers:              Число параллельных процессов (НЕ потоков!).
        seed:                 Base seed (каждая партия получает seed+game_id).
        collect_trajectories: Собирать ли (obs, action, reward) для BC-обучения.
                              При True траектории также сохраняются в JSONL-файл.

    Returns:
        HarnessResult с winrates, trajectories, games_per_sec, wilson_lo/hi.

    Raises:
        ValueError: если candidate.main_py_path не задан.
        RuntimeError: если libcg.so недоступна (не-Linux платформа) — бросается
                      внутри worker при первом вызове.

    Пример:
        result = run_games(candidate, pool, n=200, workers=4)
        assert result.winrates["crustle_il"] > 0.55, "Не проходим гейт vs Crustle"

    ВАЖНО:
        - Запускать ТОЛЬКО в Docker или Kaggle notebook (Linux x86_64 для libcg.so).
        - Использовать multiprocessing, не threading (threading → SIGSEGV в libcg).
    """
    import math
    import concurrent.futures

    cand_main_py = getattr(candidate, "main_py_path", None)
    if cand_main_py is None:
        raise ValueError(
            "candidate.main_py_path не задан. "
            "Сохраните агента в файл через pilot.export_to_submission() "
            "или задайте Agent.main_py_path вручную."
        )

    all_game_results: list[GameResult] = []
    t_start = time.perf_counter()

    for opp in opponents:
        opp_main_py = getattr(opp, "main_py_path", None)
        # opp_main_py может быть None → worker использует inline random

        # Разбиваем n партий на chunks по числу workers
        chunk_size = max(1, math.ceil(n / workers))
        chunks: list[_WorkerArgs] = []
        game_id_offset = 0
        for _ in range(workers):
            remaining = n - game_id_offset
            if remaining <= 0:
                break
            this_n = min(chunk_size, remaining)
            chunks.append(_WorkerArgs(
                candidate_main_py=cand_main_py,
                candidate_deck=list(candidate.deck),
                opponent_main_py=opp_main_py,
                opponent_deck=list(opp.deck),
                opponent_name=opp.name,
                n_games=this_n,
                game_id_offset=game_id_offset,
                seed=seed,
                collect_trajectories=collect_trajectories,
            ))
            game_id_offset += this_n

        # ProcessPoolExecutor — НЕ ThreadPoolExecutor (движок не thread-safe)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(MIN_WORKERS, len(chunks))) as ex:
            futures = [ex.submit(_worker_fn, ch) for ch in chunks]
            for fut in concurrent.futures.as_completed(futures):
                all_game_results.extend(fut.result())

    t_total = time.perf_counter() - t_start
    return _aggregate_results(all_game_results, opponents, t_total, collect_trajectories)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI для доли wins/n."""
    import math
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _aggregate_results(
    game_results: list[GameResult],
    opponents: "list[Agent]",
    total_time_s: float,
    collect_trajectories: bool,
) -> HarnessResult:
    """Агрегирует список GameResult-ов в HarnessResult.

    Подсчитывает wins/losses/draws по каждому оппоненту,
    вычисляет Wilson CI на совокупном винрейте,
    сохраняет траектории в JSONL-файл.
    """
    from collections import defaultdict

    opp_wins: dict[str, int] = defaultdict(int)
    opp_losses: dict[str, int] = defaultdict(int)
    opp_draws: dict[str, int] = defaultdict(int)
    all_trajectories: list[Trajectory] = []

    for r in game_results:
        cand_idx = r.cand_player_idx
        if r.winner == cand_idx:
            opp_wins[r.opponent_name] += 1
        elif r.winner == -1:
            opp_draws[r.opponent_name] += 1
        else:
            opp_losses[r.opponent_name] += 1

        if collect_trajectories:
            all_trajectories.extend(r.trajectories)

    # Винрейт per-оппонент (исключая ничьи)
    all_opp_names = set(opp_wins) | set(opp_losses) | set(opp_draws)
    winrates: dict[str, float] = {}
    for opp_name in all_opp_names:
        wins = opp_wins[opp_name]
        losses = opp_losses[opp_name]
        decisive = wins + losses
        winrates[opp_name] = wins / decisive if decisive > 0 else 0.5

    # Совокупная статистика
    total_wins = sum(opp_wins.values())
    total_losses = sum(opp_losses.values())
    total_games = len(game_results)
    decisive_total = total_wins + total_losses

    lo, hi = _wilson_ci(total_wins, decisive_total)
    games_per_sec = total_games / total_time_s if total_time_s > 0 else 0.0

    # Сохраняем траектории в JSONL
    traj_path = ""
    if collect_trajectories and all_trajectories:
        try:
            traj_path = _save_trajectories(all_trajectories)
        except Exception as exc:
            import warnings
            warnings.warn(f"Не удалось сохранить траектории: {exc}")

    return HarnessResult(
        winrates=winrates,
        game_results=game_results,
        trajectories=all_trajectories,
        games_per_sec=games_per_sec,
        wilson_lo=lo,
        wilson_hi=hi,
        total_games=total_games,
        total_wins=total_wins,
        trajectory_path=traj_path,
    )


def _save_trajectories(
    trajectories: list[Trajectory],
    out_dir: str | None = None,
) -> str:
    """Сохраняет траектории в JSONL-файл.

    Формат: одна запись на строку в виде JSON-объекта:
    {"obs_dict": {...}, "chosen": [...], "n_options": N,
     "final_reward": R, "player_id": P, "game_id": G, "opponent_name": "..."}

    Args:
        trajectories: Список Trajectory-объектов.
        out_dir:      Директория для файла (по умолчанию results/ в корне репо).

    Returns:
        Абсолютный путь к созданному JSONL-файлу.
    """
    import json
    from datetime import datetime

    if out_dir is None:
        out_dir = os.path.join(_REPO, "results")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"trajectories_{ts}.jsonl")

    with open(path, "w", encoding="utf-8") as f:
        for t in trajectories:
            record: dict = {
                "obs_dict": t.obs_dict,
                "chosen": t.chosen,
                "n_options": t.n_options,
                "final_reward": t.final_reward,
                "player_id": t.player_id,
                "game_id": t.game_id,
                "opponent_name": t.opponent_name,
            }
            try:
                line = json.dumps(record, ensure_ascii=False)
            except (TypeError, ValueError):
                # obs_dict содержит не-сериализуемые объекты — сохраняем без него
                record["obs_dict"] = None
                line = json.dumps(record, ensure_ascii=False)
            f.write(line + "\n")

    return path


# ---------------------------------------------------------------------------
# CLI entry-point (запуск через scripts/run_field.sh)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys as _sys

    # Добавляем src/ в path для импортов
    _src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src_dir not in _sys.path:
        _sys.path.insert(0, _src_dir)

    from exit.opponents import list_opponents, _load_deck_csv, Agent  # noqa: E402

    parser = argparse.ArgumentParser(
        description="WP2 — поле прогонов кандидата vs все оппоненты"
    )
    parser.add_argument(
        "--candidate", required=True,
        help="Версия сабмита (e.g. v012_ref) или абсолютный путь к main.py",
    )
    parser.add_argument("--n", type=int, default=100, help="Партий на оппонента")
    parser.add_argument("--workers", type=int, default=4, help="Параллельных процессов")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-trajectories", action="store_true",
        help="Не собирать траектории (быстрее, меньше памяти)",
    )
    parser.add_argument(
        "--opponents", nargs="*",
        help="Имена оппонентов через пробел (по умолчанию все доступные)",
    )
    cli = parser.parse_args()

    # Определяем пути к кандидату
    if os.path.isabs(cli.candidate) or cli.candidate.endswith(".py"):
        cand_main_py = cli.candidate
        cand_dir = os.path.dirname(cand_main_py)
    else:
        # Версия сабмита: submissions/<version>/main.py
        cand_dir = os.path.join(_REPO, "submissions", cli.candidate)
        cand_main_py = os.path.join(cand_dir, "main.py")

    cand_deck_csv = os.path.join(cand_dir, "deck.csv")

    if not os.path.exists(cand_main_py):
        print(f"ОШИБКА: main.py кандидата не найден: {cand_main_py}", file=_sys.stderr)
        _sys.exit(1)
    if not os.path.exists(cand_deck_csv):
        print(f"ОШИБКА: deck.csv кандидата не найден: {cand_deck_csv}", file=_sys.stderr)
        _sys.exit(1)

    cand_deck = _load_deck_csv(cand_deck_csv)
    candidate = Agent(
        name=cli.candidate,
        deck=cand_deck,
        policy=lambda obs: [],  # placeholder — не используется в worker
        main_py_path=cand_main_py,
    )

    pool = list_opponents(names=cli.opponents or None, skip_errors=True)
    if not pool:
        print("ОШИБКА: нет доступных оппонентов", file=_sys.stderr)
        _sys.exit(1)

    print(f"Кандидат : {cli.candidate}")
    print(f"  main.py: {cand_main_py}")
    print(f"  deck   : {len(cand_deck)} карт")
    print(f"Оппоненты: {[o.name for o in pool]}")
    print(f"Партий на оппонента: {cli.n}")
    print(f"Workers : {cli.workers}")
    print(f"Траектории: {'нет' if cli.no_trajectories else 'да'}")
    print()

    result = run_games(
        candidate=candidate,
        opponents=pool,
        n=cli.n,
        workers=cli.workers,
        seed=cli.seed,
        collect_trajectories=not cli.no_trajectories,
    )

    print("=" * 50)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 50)
    print(f"Итого партий : {result.total_games}")
    print(f"Побед        : {result.total_wins}")
    print(f"Скорость     : {result.games_per_sec:.2f} партий/сек")
    overall_wr = result.total_wins / result.total_games if result.total_games else 0.0
    print(
        f"Общий WR     : {overall_wr:.3f} "
        f"(Wilson 95%: [{result.wilson_lo:.3f}, {result.wilson_hi:.3f}])"
    )
    print()
    print("Per-opponent:")
    for opp_name, wr in sorted(result.winrates.items(), key=lambda x: x[1], reverse=True):
        print(f"  {opp_name:24s}: {wr:.3f}")
    if result.trajectory_path:
        print()
        print(f"Траектории сохранены : {result.trajectory_path}")
        print(f"Записей              : {len(result.trajectories)}")
