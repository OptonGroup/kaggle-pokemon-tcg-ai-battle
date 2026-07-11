"""src/exit — Expert Iteration (ExIt) loop против замороженного поля оппонентов.

Пакет реализует "AlphaZero в миниатюре" для Pokemon TCG с неполной информацией:
вместо self-play — итерационное улучшение против ФИКСИРОВАННОГО пула оппонентов.

Компоненты:
  opponents.py  — WP1: замороженный пул оппонентов, единый интерфейс Agent
  harness.py    — WP2: массовые прогоны агент × пул через multiprocessing
  pimc_agent.py — WP3: PIMC-эксперт (детерминизация + rollout движка)
  pilot.py      — WP4: BC-prior / policy-prior (обучение + инференс)
  exit_loop.py  — WP5: цикл expert-iteration (pilot+PIMC → фильтрация → обучение → повтор)

Северная звезда: ВИНРЕЙТ ПРОТИВ ФИКСИРОВАННОГО ПОЛЯ, не против себя.

ВАЖНО: движок (libcg.so) — Linux x86_64 only. Все импорты движка — ЛЕНИВЫЕ (внутри
функций), чтобы модули грузились и на macOS без ошибок import-time.
"""

from .opponents import Agent, list_opponents
from .harness import run_games, GameResult

__all__ = [
    "Agent",
    "list_opponents",
    "run_games",
    "GameResult",
]
