"""WP5 — Драйвер expert-iteration против ЗАМОРОЖЕННОГО поля.

Цикл "AlphaZero в миниатюре" для неполной информации. НЕ голый self-play:
каждая итерация оценивается ВИНРЕЙТОМ ПРОТИВ ФИКСИРОВАННОГО ПУЛА оппонентов
(WP1), а не против себя.

Схема одной итерации:
  1. Собрать эксперта: PIMCAgent(prior_agent=current_pilot)   [WP3 + WP4]
  2. Прогнать эксперта против ЗАМОРОЖЕННОГО поля               [WP2 harness]
       → winrates + trajectories
  3. Отфильтровать траектории где ПОБИЛИ поле (final_reward>=min_reward)
       filter_mode="win"  : quality > volume (урок v008)
       filter_mode="dagger": переразметка PIMC-экспертом (заглушка с TODO)
  4. Дообучить pilot на отфильтрованных траекториях            [WP4 train]
  5. Переоценить winrate нового pilot против того же поля       [WP2 harness]
  6. Если winrate вырос и прошёл гейт → сохранить как champion; иначе откатить.
  7. Повторить.

СЕВЕРНАЯ ЗВЕЗДА: aggregate winrate против фиксированного поля.
Поле НЕ меняется между итерациями (иначе метрика не сравнима).

ГЕЙТ (из PLAN.md Guardrails, NON-NEGOTIABLE):
  Кандидат промоутится только если winrate >= 55% над >= 300 партий И Wilson_lo > 50%.

Пример использования (в Docker):
  from src.exit.opponents import list_opponents
  from src.exit.pilot import Pilot
  from src.exit.exit_loop import run_exit_loop

  pool = list_opponents(names=["crustle_il", "iono_heur", "lucario_heur", "random"])
  init_pilot = Pilot().load("models/il_iono_a.txt")
  best = run_exit_loop(init_pilot, pool, n_iters=5, games_per_opp=300, workers=4)

Запуск в Docker:
  ./scripts/run_exit.sh [n_iters] [games_per_opp] [workers] [pimc_budget] [filter_mode]
  ./scripts/run_exit.sh 1 20 2 1.0 win    # smoke-тест
  ./scripts/run_exit.sh 5 300 4 3.0 win   # production
"""

from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .opponents import Agent
    from .pilot import Pilot
    from .harness import Trajectory, HarnessResult

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Гейт-пороги (из PLAN.md Guardrails)
GATE_WINRATE_MIN: float = 0.55
GATE_WILSON_LO_MIN: float = 0.50
GATE_MIN_GAMES: int = 300

# Ранняя остановка если winrate не растёт PLATEAU_PATIENCE итераций подряд
PLATEAU_PATIENCE: int = 3


# ---------------------------------------------------------------------------
# Типы данных
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    """Результат одной итерации ExIt-цикла.

    Attributes:
        iteration:      Номер итерации (0-based).
        winrate:        Aggregate winrate нового pilot против всего поля.
        winrate_before: Winrate до дообучения (эксперт с текущим prior).
        wilson_lo:      Wilson 95% lower bound на aggregate winrate нового pilot.
        per_opponent:   {opponent_name → winrate} нового pilot.
        n_trajectories: Число отфильтрованных траекторий для обучения.
        promoted:       Прошёл ли гейт и промоутнут ли pilot.
        pilot_path:     Путь к сохранённому pilot (если promoted).
        train_metrics:  Метрики обучения (top1_val и т.д.).
    """
    iteration: int
    winrate: float
    winrate_before: float
    wilson_lo: float
    per_opponent: dict[str, float] = field(default_factory=dict)
    n_trajectories: int = 0
    promoted: bool = False
    pilot_path: str = ""
    train_metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Гейт
# ---------------------------------------------------------------------------

def passes_gate(winrate: float, wilson_lo: float, n_games: int) -> bool:
    """Проверяет прохождение submission-гейта (PLAN.md Guardrails).

    Args:
        winrate:   Aggregate winrate против поля (исключая ничьи).
        wilson_lo: Wilson 95% lower bound.
        n_games:   Число сыгранных партий.

    Returns:
        True если winrate>=55% И wilson_lo>50% И n>=300.
    """
    return (
        winrate >= GATE_WINRATE_MIN
        and wilson_lo > GATE_WILSON_LO_MIN
        and n_games >= GATE_MIN_GAMES
    )


# ---------------------------------------------------------------------------
# Фильтры траекторий (два режима: win-filter и dagger)
# ---------------------------------------------------------------------------

def filter_trajectories(
    trajectories: "list[Trajectory]",
    filter_mode: str = "win",
    min_reward: float = 1.0,
    pimc_budget: float = 3.0,
    deck: list[int] | None = None,
) -> "list[Trajectory]":
    """Фильтрует/переразмечает траектории для BC-обучения пилота.

    Args:
        trajectories: Сырые траектории из run_games().
        filter_mode:
          "win"    — берём только выигранные (final_reward >= min_reward).
                     Качество > объём (урок v008). Быстро, надёжно.
          "dagger" — переразметка PIMC-экспертом на посещённых состояниях.
                     Реализован как заглушка с чётким интерфейсом (TODO).
        min_reward:  Порог фильтрации (только для "win").
        pimc_budget: Бюджет PIMC на ход в секундах (только для "dagger").
        deck:        Дек для PIMCAgent при переразметке (только для "dagger").

    Returns:
        Отфильтрованный список Trajectory-совместимых объектов.

    Raises:
        NotImplementedError: для "dagger" (пока заглушка).
        ValueError: если filter_mode неизвестен.
    """
    if filter_mode == "win":
        return _filter_win(trajectories, min_reward)
    elif filter_mode == "dagger":
        return _filter_dagger(trajectories, pimc_budget=pimc_budget, deck=deck)
    else:
        raise ValueError(
            f"filter_mode должен быть 'win' или 'dagger', получен {filter_mode!r}"
        )


def _filter_win(
    trajectories: "list[Trajectory]",
    min_reward: float = 1.0,
) -> "list[Trajectory]":
    """win-filter: оставляем только партии с final_reward >= min_reward.

    CRITICAL (quality > volume — урок v008): больше данных ≠ лучше.
    Только победные траектории несут надёжный сигнал действия.
    """
    return [t for t in trajectories if getattr(t, "final_reward", 0.0) >= min_reward]


def _filter_dagger(
    trajectories: "list[Trajectory]",
    pimc_budget: float = 3.0,
    deck: list[int] | None = None,
) -> "list[Trajectory]":
    """DAgger-фильтр: переразметка PIMC-экспертом (ЗАГЛУШКА — интерфейс готов, тело TODO).

    Алгоритм (когда будет реализован):
      1. Для каждого (obs_dict, ...) из trajectories:
         a. Создать PIMCAgent(budget=pimc_budget, deck=deck, opponent_model="auto").
         b. Вызвать pimc.agent(obs_dict) → expert_action.
         c. Создать новый Trajectory(obs_dict=obs_dict, chosen=expert_action,
                                     final_reward=1.0, ...).
      2. Вернуть переразмеченный набор.

    TODO[dagger]:
      - Запускать PIMC только для MAIN-контекста (context=0).
      - Батчевать obs_dict и использовать multiprocessing (движок Linux only).
      - Кэшировать идентичные состояния.

    Стоимость: O(|trajectories| × pimc_budget).
      При 300 партий × 120 решений × 3s = ~30 часов CPU → только Kaggle/кластер.
    Рекомендация: сначала win-filter (доказан в v007/v009a), dagger — позже.
    """
    raise NotImplementedError(
        "DAgger-режим: переразметка PIMC-экспертом не реализована.\n"
        "Используйте filter_mode='win' (качество > объём — урок v008).\n"
        "TODO: PIMCAgent.agent(obs_dict) → expert_action → Trajectory(chosen=expert_action).\n"
        "Интерфейс готов — достаточно заполнить тело и добавить multiprocessing.\n"
        "Стоимость: O(|traj| × pimc_budget). Только на Kaggle/кластере."
    )


# ---------------------------------------------------------------------------
# Генерация PIMC main.py для харнесса
# ---------------------------------------------------------------------------

def _generate_pimc_main_py(
    deck_list: list[int],
    budget: float,
    feature_module: str = "features_iono",
    has_model: bool = True,
) -> str:
    """Генерирует main.py который запускает PIMCAgent с BC-prior (LightGBM).

    Этот файл выполняется в worker-процессах харнесса через _load_agent_module().
    Требует Linux + libcg.so — только в Docker. На macOS упадёт с ImportError (ожидаемо).

    Args:
        deck_list:      60 card IDs.
        budget:         Бюджет PIMC в секундах (float).
        feature_module: Имя модуля фич (без .py) для импорта build_feature_matrix.
        has_model:      True если в директории будет model.txt (BC-prior активен).

    Returns:
        Строка с кодом main.py.
    """
    deck_repr = repr(deck_list)
    return f'''\
"""
PIMC-эксперт с BC-prior pilot — сгенерировано src/exit/exit_loop.py.
Запускается worker-процессами харнесса (harness.py) в Docker/Linux.
"""
from __future__ import annotations
import os, sys

# ── __file__ guard (Kaggle exec() без __file__) ───────────────────────────────
_KAGGLE_AGENT_DIR = "/kaggle_simulations/agent"
if os.path.isdir(_KAGGLE_AGENT_DIR):
    _DIR = _KAGGLE_AGENT_DIR
else:
    try:
        _DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _DIR = os.getcwd()
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# src/ в PATH: Docker (PYTHONPATH=/workspace/src) + fallback
for _docker_src in ("/workspace/src",):
    if os.path.isdir(_docker_src) and _docker_src not in sys.path:
        sys.path.insert(0, _docker_src)

import numpy as np

# ── Дек ──────────────────────────────────────────────────────────────────────
my_deck: list[int] = {deck_repr}
assert len(my_deck) == 60, f"deck: {{len(my_deck)}} карт (ожидалось 60)"


def _find_file(name: str) -> str:
    """Ищет файл в _DIR, CWD и /kaggle_simulations/agent."""
    for p in [os.path.join(_DIR, name), name, os.path.join("/kaggle_simulations/agent", name)]:
        if os.path.exists(p):
            return p
    return os.path.join(_DIR, name)


# ── BC-prior (LightGBM + фичи) ───────────────────────────────────────────────
_has_model = {has_model!r}
_ranker = None
_features_ok = False

if _has_model:
    try:
        import lightgbm as lgb
        _ranker = lgb.Booster(model_file=_find_file("model.txt"))
    except Exception as _e:
        print(f"[pimc_main] WARNING: LightGBM не загружен: {{_e}}", file=sys.stderr)

    try:
        from {feature_module} import build_feature_matrix
        _features_ok = True
    except Exception as _e:
        print(f"[pimc_main] WARNING: фичи не загружены: {{_e}}", file=sys.stderr)


def _pilot_decide(obs_dict: dict) -> list[int]:
    """BC-prior: LightGBM → ранжирование опций."""
    if _ranker is None or not _features_ok:
        raise RuntimeError("BC-prior недоступен")
    sel = obs_dict.get("select") or {{}}
    opts = sel.get("option") or []
    n_opts = len(opts)
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)
    if n_opts == 0:
        return []
    X = build_feature_matrix(obs_dict)
    if X.shape[0] != n_opts:
        raise ValueError(f"shape mismatch: {{X.shape[0]}} != {{n_opts}}")
    scores = _ranker.predict(X.astype(np.float32))
    order = [int(i) for i in np.argsort(scores)[::-1] if 0 <= i < n_opts]
    k = max(1, min_count)
    k = min(k, max_count, n_opts)
    return order[:k]


# ── PIMCAgent (ленивая инициализация при первом вызове) ──────────────────────
_pimc = None


def _get_pimc():
    global _pimc
    if _pimc is None:
        from exit.pimc_agent import PIMCAgent
        prior = _pilot_decide if (_ranker is not None and _features_ok) else None
        _pimc = PIMCAgent(
            budget={budget!r},
            deck=my_deck,
            prior_agent=prior,
            opponent_model="auto",
        )
    return _pimc


# ── Публичная точка входа ─────────────────────────────────────────────────────

def agent(obs_dict: dict) -> list[int]:
    """Never-throw entry point. PIMC + BC-prior → действие."""
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return my_deck

    n_options = len(sel.get("option") or [])
    min_count = int(sel.get("minCount") or 1)
    max_count = int(sel.get("maxCount") or 1)
    if n_options == 0:
        return []

    try:
        return _get_pimc().agent(obs_dict)
    except Exception:
        pass

    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))
'''


def _export_pimc_to_dir(
    pilot: "Pilot",
    out_dir: str,
    budget: float,
    feature_module: str = "features_iono.py",
) -> str:
    """Экспортирует PIMC-агента (с BC-prior) в временную директорию для харнесса.

    Использует export_to_submission как базу (копирует cg/, model.txt, deck.csv,
    features) и заменяет main.py на PIMC-версию.
    Если pilot.model is None — копирует только cg/ + deck.csv (BC-prior будет None,
    PIMC использует v012 heuristic как rollout policy).

    Args:
        pilot:          Обученный Pilot (model может быть None — тогда нет BC-prior).
        out_dir:        Директория для экспорта (обычно tempdir).
        budget:         Бюджет PIMC в секундах.
        feature_module: Файл фич из src/il/ (например "features_iono.py").

    Returns:
        Абсолютный путь к созданному main.py.
    """
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    feat_module = feature_module.removesuffix(".py")
    has_model = (pilot.model is not None)

    if has_model:
        # Полный экспорт через export_to_submission
        from .pilot import export_to_submission
        export_to_submission(
            pilot=pilot,
            deck=pilot.deck,
            out_dir=out_dir,
            feature_module=feature_module,
        )
    else:
        # Минимальный экспорт: cg/ + deck.csv (нет модели → PIMC без BC-prior)
        repo = Path(_REPO)
        cg_src = repo / "src" / "cg"
        if not cg_src.is_dir():
            raise FileNotFoundError(f"src/cg/ не найдена: {cg_src}")
        cg_dst = out / "cg"
        if cg_dst.exists():
            shutil.rmtree(str(cg_dst))
        shutil.copytree(str(cg_src), str(cg_dst))
        with open(out / "deck.csv", "w") as f:
            for cid in pilot.deck:
                f.write(f"{cid}\n")

    # Заменяем main.py на PIMC-версию (export_to_submission генерирует plain pilot main.py)
    pimc_main_text = _generate_pimc_main_py(
        deck_list=pilot.deck,
        budget=budget,
        feature_module=feat_module,
        has_model=has_model,
    )
    main_path = out / "main.py"
    main_path.write_text(pimc_main_text, encoding="utf-8")
    return str(main_path)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _compute_overall_winrate(harness_result: "HarnessResult") -> float:
    """Вычисляет aggregate winrate кандидата (исключая ничьи)."""
    decisive = sum(1 for r in harness_result.game_results if r.winner != -1)
    return harness_result.total_wins / decisive if decisive > 0 else 0.5


def _log_iteration(log_path: str, result: IterationResult) -> None:
    """Дописывает строку в exit_log.csv (append-only). Создаёт заголовок если файла нет."""
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0

    row = {
        "iteration":       result.iteration,
        "winrate":         round(result.winrate, 4),
        "winrate_before":  round(result.winrate_before, 4),
        "wilson_lo":       round(result.wilson_lo, 4),
        "n_trajectories":  result.n_trajectories,
        "promoted":        int(result.promoted),
        "pilot_path":      result.pilot_path,
        "per_opponent":    json.dumps(
            {k: round(v, 4) for k, v in result.per_opponent.items()},
            ensure_ascii=False,
        ),
        "train_metrics":   json.dumps(result.train_metrics, ensure_ascii=False),
    }

    fieldnames = list(row.keys())
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Одна итерация
# ---------------------------------------------------------------------------

def run_iteration(
    pilot: "Pilot",
    pool: "list[Agent]",
    iteration: int,
    games_per_opp: int = 300,
    workers: int = 4,
    pimc_budget: float = 3.0,
    min_reward: float = 1.0,
    out_dir: str = "models",
    filter_mode: str = "win",
    feature_module: str = "features_iono.py",
) -> "tuple[Pilot, IterationResult]":
    """Один шаг ExIt-цикла: эксперт играет поле → фильтр → обучение → переоценка.

    Args:
        pilot:         Текущий policy-prior (WP4 Pilot).
        pool:          ЗАМОРОЖЕННЫЙ пул оппонентов (WP1). НЕ меняется между итерациями.
        iteration:     Номер итерации (0-based, для логов и чекпойнтов).
        games_per_opp: Партий против каждого оппонента (>= 300 для гейта).
        workers:       Процессов multiprocessing (НЕ потоков! libcg SIGSEGV).
        pimc_budget:   Бюджет PIMC-эксперта на ход, секунды.
        min_reward:    Порог фильтрации траекторий (1.0 = только победы).
        out_dir:       Директория для сохранения чекпойнтов (pilot_iterN.txt).
        filter_mode:   "win" | "dagger". "dagger" — NotImplementedError (заглушка).
        feature_module: Файл фич из src/il/ (например "features_v009a.py" для
                        Crustle или "features_iono.py" для Iono — ДОЛЖЕН совпадать
                        с feature_fn, на котором обучен/дообучается pilot). Раньше
                        было захардкожено на "features_iono.py" в обоих местах
                        экспорта (PIMC-эксперт + переоценка) — баг для не-Iono деков.

    Returns:
        (new_pilot, IterationResult).
        new_pilot == pilot если гейт не пройден (anti-regression откат).

    Примечания:
        - PIMC-агент и новый pilot сначала экспортируются во временные директории
          (для harness.run_games который требует main_py_path).
        - Переоценка (шаг 5) — plain pilot (без PIMC), чтобы измерять чистый прирост IL.
        - Anti-regression: не промоутим если winrate нового pilot ниже эксперта.
    """
    from .harness import run_games
    from .opponents import Agent
    from .pilot import export_to_submission

    os.makedirs(out_dir, exist_ok=True)

    # Временные директории для экспорта (убираем в finally)
    tmp_pimc_dir = tempfile.mkdtemp(prefix=f"exit_pimc_iter{iteration}_")
    tmp_pilot_dir = tempfile.mkdtemp(prefix=f"exit_eval_iter{iteration}_")

    # Значения по умолчанию (будут перезаписаны)
    winrate_before: float = 0.5
    winrate_after: float = 0.5
    wilson_lo_after: float = 0.0
    total_games_after: int = 0
    per_opp_after: dict[str, float] = {}
    n_filtered: int = 0
    train_metrics: dict = {}
    promoted: bool = False
    pilot_path: str = ""
    new_pilot: "Pilot" = pilot

    try:
        # ── Шаг 1: Экспортируем PIMC-эксперта ──────────────────────────────
        print(f"\n[ExIt iter={iteration}] Шаг 1: Экспорт PIMC-эксперта (budget={pimc_budget}s)...")
        pimc_main_py = _export_pimc_to_dir(
            pilot=pilot,
            out_dir=tmp_pimc_dir,
            budget=pimc_budget,
            feature_module=feature_module,
        )

        pimc_agent = Agent(
            name=f"pimc_expert_iter{iteration}",
            deck=list(pilot.deck),
            policy=lambda obs: [],  # placeholder — harness использует main_py_path
            main_py_path=pimc_main_py,
        )

        # ── Шаг 2: Прогон эксперта против ЗАМОРОЖЕННОГО поля ───────────────
        opp_names = [o.name for o in pool]
        print(
            f"[ExIt iter={iteration}] Шаг 2: Эксперт vs {len(pool)} оппонентов "
            f"× {games_per_opp} партий = {len(pool) * games_per_opp} итого"
        )
        print(f"  Оппоненты: {opp_names}")

        t0 = time.perf_counter()
        expert_result = run_games(
            candidate=pimc_agent,
            opponents=pool,
            n=games_per_opp,
            workers=workers,
            collect_trajectories=True,
        )
        t_games = time.perf_counter() - t0

        winrate_before = _compute_overall_winrate(expert_result)
        print(
            f"[ExIt iter={iteration}] Эксперт: WR={winrate_before:.3f} "
            f"wilson=[{expert_result.wilson_lo:.3f},{expert_result.wilson_hi:.3f}] "
            f"wins={expert_result.total_wins}/{expert_result.total_games} "
            f"({expert_result.games_per_sec:.2f} партий/сек, {t_games:.0f}s)"
        )
        for opp_name, wr in sorted(expert_result.winrates.items(), key=lambda x: x[1]):
            print(f"  {opp_name:28s}: {wr:.3f}")

        # ── Шаг 3: Фильтрация траекторий ────────────────────────────────────
        raw_trajs = expert_result.trajectories
        print(
            f"[ExIt iter={iteration}] Шаг 3: Фильтр траекторий "
            f"(mode={filter_mode}, min_reward={min_reward}). "
            f"Всего: {len(raw_trajs)}"
        )

        filtered_trajs = filter_trajectories(
            raw_trajs,
            filter_mode=filter_mode,
            min_reward=min_reward,
            pimc_budget=pimc_budget,
            deck=list(pilot.deck),
        )
        n_filtered = len(filtered_trajs)
        print(f"[ExIt iter={iteration}] Отфильтровано: {n_filtered}")

        if n_filtered < 10:
            print(
                f"[ExIt iter={iteration}] ПРЕДУПРЕЖДЕНИЕ: слишком мало обучающих траекторий "
                f"({n_filtered} < 10). Пропускаем обучение — возвращаем старый pilot."
            )
            return pilot, IterationResult(
                iteration=iteration,
                winrate=winrate_before,
                winrate_before=winrate_before,
                wilson_lo=expert_result.wilson_lo,
                per_opponent=dict(expert_result.winrates),
                n_trajectories=n_filtered,
                promoted=False,
                pilot_path="",
                train_metrics={"skipped": "not enough trajectories"},
            )

        # ── Шаг 4: Обучаем нового пилота ────────────────────────────────────
        print(f"[ExIt iter={iteration}] Шаг 4: Обучение нового пилота...")

        try:
            new_pilot_candidate = copy.deepcopy(pilot)
        except Exception:
            # deepcopy может упасть на LightGBM Booster — создаём свежий pilot
            from .pilot import Pilot
            new_pilot_candidate = Pilot(
                feature_fn=pilot.feature_fn,
                model_type=pilot.model_type,
                deck=list(pilot.deck),
            )

        try:
            train_metrics = new_pilot_candidate.train(
                filtered_trajs,
                min_reward=min_reward,
            )
            print(f"[ExIt iter={iteration}] Train metrics: {train_metrics}")
        except Exception as exc:
            print(f"[ExIt iter={iteration}] ОШИБКА обучения: {exc}")
            return pilot, IterationResult(
                iteration=iteration,
                winrate=winrate_before,
                winrate_before=winrate_before,
                wilson_lo=expert_result.wilson_lo,
                per_opponent=dict(expert_result.winrates),
                n_trajectories=n_filtered,
                promoted=False,
                pilot_path="",
                train_metrics={"error": str(exc)},
            )

        # ── Шаг 5: Переоцениваем нового пилота ───────────────────────────────
        print(f"[ExIt iter={iteration}] Шаг 5: Переоценка нового пилота (без PIMC)...")

        export_to_submission(
            pilot=new_pilot_candidate,
            deck=new_pilot_candidate.deck,
            out_dir=tmp_pilot_dir,
            feature_module=feature_module,
        )
        new_pilot_main_py = os.path.join(tmp_pilot_dir, "main.py")

        eval_agent = Agent(
            name=f"pilot_iter{iteration}",
            deck=list(new_pilot_candidate.deck),
            policy=lambda obs: [],  # placeholder
            main_py_path=new_pilot_main_py,
        )

        eval_result = run_games(
            candidate=eval_agent,
            opponents=pool,
            n=games_per_opp,
            workers=workers,
            collect_trajectories=False,  # переоценка — без сбора данных
        )

        winrate_after = _compute_overall_winrate(eval_result)
        wilson_lo_after = eval_result.wilson_lo
        total_games_after = eval_result.total_games
        per_opp_after = dict(eval_result.winrates)

        print(
            f"[ExIt iter={iteration}] Новый пилот: WR={winrate_after:.3f} "
            f"wilson=[{eval_result.wilson_lo:.3f},{eval_result.wilson_hi:.3f}] "
            f"wins={eval_result.total_wins}/{eval_result.total_games}"
        )
        for opp_name, wr in sorted(per_opp_after.items(), key=lambda x: x[1]):
            print(f"  {opp_name:28s}: {wr:.3f}")

        # ── Шаг 6: Гейт + anti-regression ────────────────────────────────────
        gate_ok = passes_gate(winrate_after, wilson_lo_after, total_games_after)
        improved = winrate_after >= winrate_before  # anti-regression

        if gate_ok and improved:
            pilot_path = os.path.join(out_dir, f"pilot_iter{iteration}.txt")
            new_pilot_candidate.save(pilot_path)
            # Перезаписываем "best" чекпойнт
            best_path = os.path.join(out_dir, "pilot_best.txt")
            new_pilot_candidate.save(best_path)
            promoted = True
            new_pilot = new_pilot_candidate
            print(
                f"[ExIt iter={iteration}] PROMOTED:"
                f" WR {winrate_before:.3f} -> {winrate_after:.3f},"
                f" wilson_lo={wilson_lo_after:.3f} > 0.50,"
                f" games={total_games_after}"
            )
        else:
            reasons: list[str] = []
            if not gate_ok:
                if winrate_after < GATE_WINRATE_MIN:
                    reasons.append(f"WR {winrate_after:.3f} < {GATE_WINRATE_MIN}")
                if wilson_lo_after <= GATE_WILSON_LO_MIN:
                    reasons.append(f"wilson_lo {wilson_lo_after:.3f} <= {GATE_WILSON_LO_MIN}")
                if total_games_after < GATE_MIN_GAMES:
                    reasons.append(f"games {total_games_after} < {GATE_MIN_GAMES}")
            if not improved:
                reasons.append(
                    f"anti-regression: {winrate_after:.3f} < {winrate_before:.3f}"
                )
            print(f"[ExIt iter={iteration}] NOT PROMOTED: {'; '.join(reasons)}")
            new_pilot = pilot  # откат к старому

    finally:
        # Убираем временные директории
        for d in (tmp_pimc_dir, tmp_pilot_dir):
            try:
                shutil.rmtree(d)
            except Exception:
                pass

    return new_pilot, IterationResult(
        iteration=iteration,
        winrate=winrate_after,
        winrate_before=winrate_before,
        wilson_lo=wilson_lo_after,
        per_opponent=per_opp_after,
        n_trajectories=n_filtered,
        promoted=promoted,
        pilot_path=pilot_path,
        train_metrics=train_metrics,
    )


# ---------------------------------------------------------------------------
# Полный цикл
# ---------------------------------------------------------------------------

def run_exit_loop(
    init_pilot: "Pilot",
    pool: "list[Agent]",
    n_iters: int = 5,
    games_per_opp: int = 300,
    workers: int = 4,
    pimc_budget: float = 3.0,
    log_path: str = "results/exit_log.csv",
    out_dir: str = "models",
    filter_mode: str = "win",
    min_reward: float = 1.0,
    feature_module: str = "features_iono.py",
) -> "Pilot":
    """Полный expert-iteration цикл.

    СЕВЕРНАЯ ЗВЕЗДА каждой итерации = ВИНРЕЙТ ПРОТИВ ФИКСИРОВАННОГО ПОЛЯ (pool).
    НЕ против своей копии.

    Args:
        init_pilot:    Стартовый policy-prior (напр. загруженная IL модель).
        pool:          ЗАМОРОЖЕННЫЙ пул оппонентов (WP1). НЕ меняется!
        n_iters:       Число итераций.
        games_per_opp: Партий против каждого оппонента за итерацию (>= 300 для гейта).
        workers:       Процессов multiprocessing (НЕ потоков!).
        pimc_budget:   Бюджет PIMC-эксперта на ход, сек.
        log_path:      CSV лог (iter, winrate, wilson_lo, promoted, per_opponent, ...).
        out_dir:       Директория для чекпойнтов (pilot_iterN.txt, pilot_best.txt).
        filter_mode:   "win" | "dagger". Режим фильтрации траекторий.
        min_reward:    Порог для win-filter (1.0 = только победы).

    Returns:
        Лучший pilot по aggregate winrate против поля за все итерации.

    Ранняя остановка:
        Если winrate не улучшается PLATEAU_PATIENCE (=3) итераций подряд.

    Лог:
        results/exit_log.csv — строка на каждую итерацию:
        iteration, winrate, winrate_before, wilson_lo, n_trajectories,
        promoted, pilot_path, per_opponent (JSON), train_metrics (JSON).
    """
    # Абсолютные пути
    abs_log = log_path if os.path.isabs(log_path) else os.path.join(_REPO, log_path)
    abs_out = out_dir if os.path.isabs(out_dir) else os.path.join(_REPO, out_dir)

    os.makedirs(abs_out, exist_ok=True)
    os.makedirs(os.path.dirname(abs_log) or ".", exist_ok=True)

    best_pilot = init_pilot
    best_winrate: float = 0.0
    plateau_count: int = 0

    print("=" * 70)
    print("[ExIt] EXPERT-ITERATION ЦИКЛ")
    print(f"  Итераций      : {n_iters}")
    print(f"  Оппонентов    : {len(pool)} ({[o.name for o in pool]})")
    print(f"  Партий/оппон  : {games_per_opp}")
    print(f"  Партий итого  : {len(pool) * games_per_opp} (× 2 при переоценке)")
    print(f"  Workers       : {workers}")
    print(f"  PIMC budget   : {pimc_budget}s")
    print(f"  Filter mode   : {filter_mode}")
    print(f"  Min reward    : {min_reward}")
    print(f"  Лог           : {abs_log}")
    print(f"  Чекпойнты     : {abs_out}/pilot_iter*.txt")
    print("=" * 70)

    for i in range(n_iters):
        print(f"\n{'#' * 70}")
        print(f"# ExIt: ИТЕРАЦИЯ {i + 1}/{n_iters}  (plateau={plateau_count}/{PLATEAU_PATIENCE})")
        print(f"{'#' * 70}")

        t_iter_start = time.perf_counter()

        try:
            new_pilot, result = run_iteration(
                pilot=best_pilot,
                pool=pool,
                iteration=i,
                games_per_opp=games_per_opp,
                workers=workers,
                pimc_budget=pimc_budget,
                min_reward=min_reward,
                out_dir=abs_out,
                filter_mode=filter_mode,
                feature_module=feature_module,
            )
        except Exception as exc:
            print(f"[ExIt iter={i}] КРИТИЧЕСКАЯ ОШИБКА итерации: {exc}")
            # Не прерываем весь цикл — продолжаем с тем же pilot
            result = IterationResult(
                iteration=i,
                winrate=best_winrate,
                winrate_before=best_winrate,
                wilson_lo=0.0,
                promoted=False,
                train_metrics={"critical_error": str(exc)},
            )
            new_pilot = best_pilot
            plateau_count += 1

        t_iter_total = time.perf_counter() - t_iter_start

        # Логируем
        try:
            _log_iteration(abs_log, result)
        except Exception as exc:
            print(f"[ExIt iter={i}] WARN: не удалось записать лог: {exc}")

        print(
            f"\n[ExIt iter={i}] Итерация завершена за {t_iter_total:.0f}s. "
            f"WR: {result.winrate_before:.3f} -> {result.winrate:.3f} "
            f"({'PROMOTED' if result.promoted else 'not promoted'})"
        )

        # Отслеживаем лучший pilot
        if result.promoted:
            if result.winrate > best_winrate:
                best_winrate = result.winrate
                best_pilot = new_pilot
                plateau_count = 0
                print(f"[ExIt] Новый BEST pilot: WR={best_winrate:.3f} (iter={i})")
            else:
                # Промоутился, но не лучший overall — всё равно обновляем (нет регрессии)
                best_pilot = new_pilot
                plateau_count += 1
        else:
            plateau_count += 1

        # Ранняя остановка на плато
        if plateau_count >= PLATEAU_PATIENCE:
            print(
                f"\n[ExIt] РАННЯЯ ОСТАНОВКА: нет улучшения "
                f"{plateau_count}/{PLATEAU_PATIENCE} итераций подряд. "
                f"Лучший WR против поля: {best_winrate:.3f}"
            )
            break

    print(f"\n{'=' * 70}")
    print(f"[ExIt] Цикл завершён. Лучший WR против замороженного поля: {best_winrate:.3f}")
    print(f"[ExIt] Чекпойнт: {abs_out}/pilot_best.txt")
    print(f"[ExIt] Лог     : {abs_log}")
    print("=" * 70)

    return best_pilot


# ---------------------------------------------------------------------------
# CLI entry-point (запускается через scripts/run_exit.sh в Docker)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys as _sys

    # Добавляем src/ в path для импортов без установленного пакета
    _src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src_dir not in _sys.path:
        _sys.path.insert(0, _src_dir)

    parser = argparse.ArgumentParser(
        description="WP5 — ExIt-цикл против замороженного поля оппонентов",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-iters",       type=int,   default=5,
                        help="Число итераций ExIt")
    parser.add_argument("--games-per-opp", type=int,   default=300,
                        help="Партий против каждого оппонента")
    parser.add_argument("--workers",       type=int,   default=4,
                        help="Параллельных процессов (НЕ потоков!)")
    parser.add_argument("--pimc-budget",   type=float, default=3.0,
                        help="Бюджет PIMC на ход, секунды")
    parser.add_argument("--filter-mode",   default="win", choices=["win", "dagger"],
                        help="Режим фильтрации траекторий")
    parser.add_argument("--min-reward",    type=float, default=1.0,
                        help="Мин. final_reward для win-filter")
    parser.add_argument("--init-model",    default=None,
                        help="Путь к стартовой модели (.txt LightGBM)")
    parser.add_argument("--log-path",      default="results/exit_log.csv",
                        help="Путь к CSV-логу")
    parser.add_argument("--out-dir",       default="models",
                        help="Директория для чекпойнтов pilot_iter*.txt")
    parser.add_argument("--opponents",     nargs="*",  default=None,
                        help="Имена оппонентов (по умолчанию все доступные)")
    parser.add_argument("--feature-module", default="features_iono.py",
                        help="Файл фич из src/il/ (должен совпадать с decкой/моделью pilot)")
    cli = parser.parse_args()

    # ── Загрузка пула оппонентов ─────────────────────────────────────────────
    from exit.opponents import list_opponents
    from exit.pilot import Pilot

    pool = list_opponents(names=cli.opponents or None, skip_errors=True)
    if not pool:
        print("ОШИБКА: нет доступных оппонентов", file=_sys.stderr)
        _sys.exit(1)
    print(f"Пул: {[o.name for o in pool]}")

    # ── Инициализация стартового pilot ──────────────────────────────────────
    pilot = Pilot()

    # Порядок поиска стартовой модели:
    # 1. --init-model (явный путь)
    # 2. models/il_iono_a.txt (стандартный IL-чекпойнт Iono)
    # 3. Без модели (PIMC будет использовать v012 heuristic как prior)
    _model_candidates = []
    if cli.init_model:
        _model_candidates.append(cli.init_model)
    for _mname in ("il_iono_a.txt", "pilot_best.txt"):
        _p = os.path.join(_REPO, "models", _mname)
        if _p not in _model_candidates:
            _model_candidates.append(_p)

    _loaded = False
    for _mp in _model_candidates:
        if os.path.exists(_mp):
            print(f"Загружаем стартовую модель: {_mp}")
            pilot.load(_mp)
            _loaded = True
            break

    if not _loaded:
        print(
            "ПРЕДУПРЕЖДЕНИЕ: стартовая модель не найдена. "
            "Первая итерация будет использовать v012 heuristic как PIMC prior.",
            file=_sys.stderr,
        )

    # ── Запуск ExIt-цикла ────────────────────────────────────────────────────
    best = run_exit_loop(
        init_pilot=pilot,
        pool=pool,
        n_iters=cli.n_iters,
        games_per_opp=cli.games_per_opp,
        workers=cli.workers,
        pimc_budget=cli.pimc_budget,
        log_path=cli.log_path,
        out_dir=cli.out_dir,
        filter_mode=cli.filter_mode,
        min_reward=cli.min_reward,
        feature_module=cli.feature_module,
    )

    print("\n[ExIt] Готово. Лучший pilot возвращён.")
