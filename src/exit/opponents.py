"""WP1 — Замороженный пул оппонентов.

Единый интерфейс Agent + функция list_opponents(), возвращающая всех доступных
оппонентов для гонки ExIt-цикла.

Пул включает:
  1. random          — случайный агент (baseline нижний порог)
  2. sample_random   — организаторский sample random (другая дека)
  3. sample_lucario  — организаторский sample Mega Lucario ex эвристика
  4. lucario_heur    — reference heuristic v002 (Mega Lucario ex, μ≈800)
  5. crustle_il      — лучший Crustle IL агент v009a (Dwebble/Crustle, μ≈946)
  6. iono_heur       — Iono Electric Swarm эвристика v012 (hand-written, ~0.70 vs Crustle)

КРИТИЧНЫЕ ТЕХНИЧЕСКИЕ ЗАМЕЧАНИЯ:
  - Движок (libcg.so) — Linux x86_64 only. Этот модуль безопасно импортируется
    на macOS: никаких top-level импортов из cg/.
  - Все policy-функции используют ЛЕНИВУЮ загрузку (lazy loading): модуль агента
    загружается только при ПЕРВОМ вызове policy(), не при создании Agent.
  - Благодаря lazy loading list_opponents() работает на macOS (проверяет только
    файловую систему), а партии гоняются только на Linux в Docker.
  - main_py_path: str | None — путь к main.py для pickle-safe передачи в
    multiprocessing-worker'ы (WP2). None = агент не имеет отдельного main.py
    (например, inline random).
  - policy: Callable[[dict], list[int]] — точный контракт агентской функции.
    Первый вызов (obs_dict["select"] is None): возвращает list[int] из 60 card IDs.
    Остальные: возвращает list[int] — индексы опций в [minCount, maxCount], без дублей.
  - deck: list[int] — 60 card IDs (может содержать дубли — это норма для TCG).
  - Оба поля обязательны; policy НИКОГДА не должна выбрасывать исключение на Kaggle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

# Корень репо — вычисляем статически, движок не нужен
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Agent:
    """Описание одного агента-оппонента.

    Attributes:
        name:         Уникальный идентификатор (используется в логах, winrate-таблице).
        deck:         60 card IDs (список может содержать повторы — норма для TCG).
        policy:       Функция agent(obs_dict: dict) -> list[int] контракт агента.
                      Первый вызов (obs_dict["select"] is None) → 60 card IDs.
                      Последующие → список индексов опций, нет дублей, длина в [min, max].
        mu:           Примерный рейтинг (μ) на Kaggle для ориентации (0 = неизвестно).
        notes:        Свободный комментарий.
        main_py_path: Абсолютный путь к main.py агента для pickle-safe передачи в
                      multiprocessing-worker'ы (WP2). None = нет отдельного main.py
                      (inline агент). Worker сам загружает модуль по этому пути,
                      передача callable через pickle небезопасна с ctypes/замыканиями.
    """
    name: str
    deck: list[int]
    policy: Callable[[dict], list[int]]
    mu: float = 0.0
    notes: str = ""
    main_py_path: str | None = None


# ---------------------------------------------------------------------------
# Вспомогательные функции загрузки
# ---------------------------------------------------------------------------

def _load_deck_csv(deck_csv_path: str) -> list[int]:
    """Читает deck.csv (одна строка = один card ID). Работает на macOS."""
    ids: list[int] = []
    with open(deck_csv_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(int(line))
    if len(ids) != 60:
        raise ValueError(f"Deck must be 60 cards, got {len(ids)}: {deck_csv_path}")
    return ids


def _load_agent_module(main_py_path: str, module_name: str):
    """Загружает агента из main.py через importlib.

    Временно переключает CWD в директорию main.py, чтобы модульный код,
    читающий 'deck.csv' из CWD (паттерн sample агентов), находил правильный
    файл. CWD восстанавливается после загрузки.

    Устанавливает mod.__file__ явно (Kaggle-safe: Kaggle грузит через exec()
    без __file__, наш importlib-путь создаёт __file__ корректно).

    Returns:
        module: импортированный модуль (module.agent — callable).

    Raises:
        FileNotFoundError: если main_py_path не существует.
        RuntimeError: если движок недоступен (Linux only); поднимается на macOS.
    """
    import importlib.util
    import sys

    if not os.path.exists(main_py_path):
        raise FileNotFoundError(f"Agent not found: {main_py_path}")

    agent_dir = os.path.dirname(os.path.abspath(main_py_path))

    # Добавляем директорию агента в sys.path (для `from cg.api import ...`)
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

    original_cwd = os.getcwd()
    try:
        # CWD-трюк: sample агенты читают deck.csv из текущей директории
        os.chdir(agent_dir)
        spec = importlib.util.spec_from_file_location(module_name, main_py_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_py_path  # Kaggle-safe
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)

    if not hasattr(mod, "agent"):
        raise AttributeError(f"No 'agent' function in {main_py_path}")
    return mod


def _make_lazy_policy(main_py_path: str, module_name: str) -> Callable[[dict], list[int]]:
    """Создаёт policy-функцию с ленивой загрузкой модуля агента.

    Модуль загружается только при ПЕРВОМ вызове policy(). Это позволяет:
    - Безопасно создавать Agent на macOS (без libcg.so)
    - Корректно запускать партии на Linux в Docker (libcg.so доступна)

    Closure хранит модуль в списке (mutable container) для обхода ограничений
    nonlocal в Python 3.10.
    """
    _mod_ref: list = [None]  # [0] = загруженный модуль или None

    def _lazy_policy(obs_dict: dict) -> list[int]:
        if _mod_ref[0] is None:
            _mod_ref[0] = _load_agent_module(main_py_path, module_name)
        return _mod_ref[0].agent(obs_dict)

    return _lazy_policy


# ---------------------------------------------------------------------------
# Inline random policy — не требует cg, работает на macOS
# ---------------------------------------------------------------------------

def _make_inline_random_policy(deck: list[int]) -> Callable[[dict], list[int]]:
    """Создаёт простую случайную policy без зависимости от cg."""
    def _random_policy(obs_dict: dict) -> list[int]:
        import random as _rng
        sel = obs_dict.get("select") if obs_dict else None
        if sel is None:
            return list(deck)
        opts = sel.get("option") or []
        n = len(opts)
        if n == 0:
            return []
        min_c = int(sel.get("minCount") or 1)
        max_c = int(sel.get("maxCount") or 1)
        k = max(1, min_c)
        k = min(k, max_c, n)
        return _rng.sample(range(n), k)
    return _random_policy


# ---------------------------------------------------------------------------
# Фабричные функции для каждого оппонента
# (вызываются только при list_opponents() — полностью ленивые)
# ---------------------------------------------------------------------------

def _make_random_agent() -> Agent:
    """Random агент — базовая линия (μ≈483). Inline policy, нет cg на macOS."""
    deck_path = os.path.join(_REPO, "submissions", "v001_random", "deck.csv")
    if not os.path.exists(deck_path):
        # Fallback на sample_submission deck
        deck_path = os.path.join(_REPO, "data", "sample_submission", "deck.csv")
    deck = _load_deck_csv(deck_path)

    main_py = os.path.join(_REPO, "submissions", "v001_random", "main.py")
    return Agent(
        name="random",
        deck=deck,
        policy=_make_inline_random_policy(deck),
        mu=483.0,
        notes="v001: random policy (baseline, inline Python — нет зависимости от cg)",
        main_py_path=main_py if os.path.exists(main_py) else None,
    )


def _make_sample_random() -> Agent:
    """Организаторский sample random агент (из data/sample_submission).

    Использует другую деку (Abomasnow-like), та же random policy.
    """
    submission_dir = os.path.join(_REPO, "data", "sample_submission")
    deck_csv = os.path.join(submission_dir, "deck.csv")
    main_py = os.path.join(submission_dir, "main.py")

    if not os.path.exists(deck_csv):
        raise FileNotFoundError(f"sample_submission not found: {deck_csv}")

    deck = _load_deck_csv(deck_csv)
    return Agent(
        name="sample_random",
        deck=deck,
        policy=_make_inline_random_policy(deck),
        mu=600.0,
        notes="sample_submission: организаторский random агент (другая дека)",
        main_py_path=main_py if os.path.exists(main_py) else None,
    )


def _make_sample_lucario() -> Agent:
    """Организаторский sample Mega Lucario ex эвристика (из data/sample_agents).

    main.py загружается лениво: на macOS создаётся Agent без загрузки cg.
    Партии — только на Linux в Docker.
    """
    agent_dir = os.path.join(_REPO, "data", "sample_agents", "mega_lucario_ex")
    main_py = os.path.join(agent_dir, "main.py")
    deck_csv = os.path.join(agent_dir, "deck.csv")

    if not os.path.exists(main_py):
        raise FileNotFoundError(f"sample_lucario not found: {main_py}")

    deck = _load_deck_csv(deck_csv)
    return Agent(
        name="sample_lucario",
        deck=deck,
        policy=_make_lazy_policy(main_py, "_sample_lucario_main"),
        mu=700.0,
        notes="sample Mega Lucario ex: организаторский промежуточный эвристик",
        main_py_path=main_py,
    )


def _make_lucario_heur() -> Agent:
    """Lucario heuristic (v002, μ≈800). Загружается лениво."""
    submission_dir = os.path.join(_REPO, "submissions", "v002_lucario")
    main_py = os.path.join(submission_dir, "main.py")
    deck_csv = os.path.join(submission_dir, "deck.csv")

    if not os.path.exists(main_py):
        raise FileNotFoundError(f"v002_lucario not found at {submission_dir}")

    deck = _load_deck_csv(deck_csv)
    return Agent(
        name="lucario_heur",
        deck=deck,
        policy=_make_lazy_policy(main_py, "_v002_lucario_main"),
        mu=800.0,
        notes="v002: Mega Lucario ex + reference heuristic pilot",
        main_py_path=main_py,
    )


def _make_crustle_il() -> Agent:
    """Crustle IL агент v009a (лучший Crustle, μ≈946). Загружается лениво."""
    submission_dir = os.path.join(_REPO, "submissions", "v009a_crustle")
    main_py = os.path.join(submission_dir, "main.py")
    deck_csv = os.path.join(submission_dir, "deck.csv")

    if not os.path.exists(main_py):
        raise FileNotFoundError(f"v009a_crustle not found at {submission_dir}")

    deck = _load_deck_csv(deck_csv)
    return Agent(
        name="crustle_il",
        deck=deck,
        policy=_make_lazy_policy(main_py, "_v009a_crustle_main"),
        mu=946.0,
        notes="v009a: Dwebble/Crustle deck + LightGBM IL ranker v3",
        main_py_path=main_py,
    )


def _make_iono_heur() -> Agent:
    """Iono Electric Swarm heuristic (v012, ~0.70 vs Crustle). Загружается лениво.

    Пробует v012_ref (src/agents/iono_heuristic.py обёрнутый), затем v012_iono_heur.
    """
    # Предпочитаем v012_ref (более свежий)
    for version in ("v012_ref", "v012_iono_heur", "v013_iono_heur"):
        submission_dir = os.path.join(_REPO, "submissions", version)
        main_py = os.path.join(submission_dir, "main.py")
        deck_csv = os.path.join(submission_dir, "deck.csv")
        if os.path.exists(main_py) and os.path.exists(deck_csv):
            deck = _load_deck_csv(deck_csv)
            return Agent(
                name="iono_heur",
                deck=deck,
                policy=_make_lazy_policy(main_py, "_iono_heur_main"),
                mu=0.0,  # Рейтинг пока неизвестен (PENDING)
                notes=f"{version}: Iono Electric Swarm + hand-written heuristic; counters ~85% Crustle field",
                main_py_path=main_py,
            )

    raise FileNotFoundError("Iono heuristic submission not found: tried v012_ref, v012_iono_heur, v013_iono_heur")


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------

# Реестр оппонентов: имя → фабричная функция
# Порядок отражает примерную силу (от слабого к сильному для информации)
_OPPONENT_REGISTRY: dict[str, Callable[[], Agent]] = {
    "random":          _make_random_agent,
    "sample_random":   _make_sample_random,
    "sample_lucario":  _make_sample_lucario,
    "lucario_heur":    _make_lucario_heur,
    "crustle_il":      _make_crustle_il,
    "iono_heur":       _make_iono_heur,
}


def list_opponents(
    names: list[str] | None = None,
    skip_errors: bool = False,
) -> list[Agent]:
    """Возвращает пул замороженных оппонентов для ExIt-цикла.

    Все policy-функции используют ленивую загрузку: вызов этой функции
    безопасен на macOS (libcg.so не нужна). Загрузка модулей происходит
    только при первом вызове policy(obs_dict) — то есть в Docker на Linux.

    Args:
        names:       Подмножество имён из реестра. None → все доступные.
        skip_errors: Если True, оппоненты которых не удалось загрузить
                     (FileNotFoundError, RuntimeError) пропускаются с
                     предупреждением вместо исключения. По умолчанию False.

    Returns:
        Список Agent, отсортированный по μ (от слабого к сильному).
        Агенты с μ=0 (неизвестен) идут в конец.

    Пример:
        >>> from src.exit.opponents import list_opponents
        >>> pool = list_opponents()
        >>> for a in pool:
        ...     print(a.name, a.mu, a.main_py_path)

        >>> # Быстрый smoke-тест (macOS-safe: policy НЕ вызывается):
        >>> names = [a.name for a in list_opponents()]
        >>> assert len(names) >= 5
    """
    if names is None:
        names = list(_OPPONENT_REGISTRY.keys())

    result: list[Agent] = []
    for name in names:
        if name not in _OPPONENT_REGISTRY:
            raise KeyError(f"Unknown opponent: {name!r}. Available: {list(_OPPONENT_REGISTRY)}")
        factory = _OPPONENT_REGISTRY[name]
        try:
            result.append(factory())
        except (FileNotFoundError, RuntimeError, OSError, AttributeError) as e:
            if skip_errors:
                import warnings
                warnings.warn(f"Skipping opponent {name!r}: {e}", stacklevel=2)
            else:
                raise

    # Сортируем по μ (0.0 = неизвестно → в конец)
    result.sort(key=lambda a: (a.mu == 0.0, a.mu))
    return result
