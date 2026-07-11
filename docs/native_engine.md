# Нативный движок (macOS arm64) — libcg.dylib

**Дата настройки:** 2026-07-03. Организатор 2026-07-01 выложил нативные бинарники
(macOS `libcg.dylib`, Linux ARM64 `libcg-arm64.so`) и полный исходник движка
(`ptcg_engine`). Self-play больше НЕ требует Docker-эмуляции — ускорение ~1.9–2.5×
на шаг плюс полноценный multiprocessing на всех ядрах Mac.

---

## Что скачано и где лежит

Скачано через `.venv/bin/kaggle competitions download -c pokemon-tcg-ai-battle -f <name>`
(Data-таб обновлён организатором 2026-07-01 15:01 UTC):

| Что | Куда | Содержимое |
|---|---|---|
| `sample_submission/sample_submission/*` (11 файлов) | `data/engine/sample_submission_cg/` | обновлённая обёртка `cg/` + **4 бинарника**: `cg.dll` (Win x64), `libcg.dylib` (**macOS arm64**), `libcg-arm64.so` (Linux aarch64), `libcg.so` (Linux x86_64, **ПЕРЕСОБРАН** — см. warning ниже) + `main.py`/`deck.csv` сэмпл-агента |
| `ptcg_engine/ptcgProgram 22/*` (45 файлов) | `data/engine/ptcg_src/` | полный C++20 исходник движка (header-only, entry points в `Export.cpp`), VS2022 `game.sln`, LICENSE, README |

`data/` в .gitignore для крупных файлов — ок.

**SHA256 бинарников (2026-07-03):**

```
77bb978a8129b094452679e0daf0da69593afda7331685f4642c0d4a94d39d82  src/cg_native/libcg.dylib      (новый, macOS arm64)
030b4728ce9fb9e90b75830b7cf7236f71859732a05ec4a377078eee0421bbe5  src/cg_native/libcg-arm64.so   (новый, Linux aarch64)
ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c  src/cg_native/libcg.so         (НОВЫЙ Linux x86_64 — пересобран!)
75d7d619b56e5ad4c5caeadc698e61faecc650678dceab52ad687f08d5676beb  src/cg/libcg.so                (СТАРЫЙ Linux x86_64, во всех submissions/*/cg/)
```

## ⚠️ WARNING: linux-движок пересобран — сабмиты с поиском должны нести НОВЫЙ libcg.so

`libcg.so` (Linux x86_64) в новом бандле организатора имеет **другой хеш**, чем тот, что
лежит в `src/cg/` и во всех существующих `submissions/*/cg/` — движок пересобрали
(вероятно, вместе с фиксами правил/карт). Платформа Kaggle играет матчи СВОЕЙ версией
движка; `cg/` внутри сабмита используется только НАШИМ агентом как внутренний симулятор
(`search_begin`/`search_step`). Расхождение версий = поиск моделирует не те правила, что
судья.

- **Все БУДУЩИЕ search-based сабмиты (PIMC/MCTS/exit) обязаны бандлить НОВЫЙ
  `libcg.so`** из `data/engine/sample_submission_cg/libcg.so` (= копия в
  `src/cg_native/libcg.so`), а не копировать старый `src/cg/`.
- Новая обёртка (`sim.py` с автодетектом платформы) — drop-in: `api.py`/`game.py`/
  `utils.py`/`__init__.py` **байт-в-байт идентичны** старым, изменился только `sim.py`.
  Для сабмита достаточно `libcg.so` (Kaggle = linux x86_64); остальные бинарники можно
  не класть, чтобы не раздувать tar (все 4 = ~5.4 MB).
- Существующие `submissions/*/cg/` **не трогаем** (правило проекта: engine dirs read-only;
  чемпион должен оставаться пересабмитимым как есть). Нынешний чемпион — эвристика без
  поиска, для него расхождение не критично.
- `src/cg/` тоже не трогаем — это рабочая копия для Docker-гейта; миграцию гейта на новый
  движок делать отдельным осознанным шагом (пере-прогнать гейт после смены).

## Как подключено (loading path)

**`src/cg_native/`** — НОВЫЙ пакет, копия обновлённой обёртки организатора + все 4
бинарника. Существующие `src/cg/` и `submissions/*/cg/` не изменялись.

Ключевое отличие новой обёртки — `sim.py` теперь сам выбирает бинарник по платформе:

```python
os_name = platform.system()
if os_name == 'Windows':      lib_path = .../cg.dll
elif os_name == "Darwin":     lib_path = .../libcg.dylib      # ← наш случай (macOS arm64)
elif platform.machine() in ('arm64','aarch64'): lib_path = .../libcg-arm64.so
else:                          lib_path = .../libcg.so
```

Импорт в коде (вместо `from cg.game import ...`):

```python
import sys; sys.path.insert(0, "src")
from cg_native.game import battle_start, battle_select, battle_finish
from cg_native.api import to_observation_class, search_begin, search_step, search_end
```

API полностью тот же (обёртки идентичны старым, кроме автодетекта в sim.py).

## Как гонять нативный self-play

```bash
# smoke + полный бенчмарк (60 партий random-vs-random, 50 search-rollout, детерминизм)
.venv/bin/python scripts/sim_native.py            # [n_games] [n_rollouts] опционально

# проверка thread-safety (запускать ТОЛЬКО отдельным процессом)
.venv/bin/python scripts/thread_check_native.py   # exit 0 = потоки выжили
```

Для параллельного self-play — **только multiprocessing** (spawn), по процессу на ядро;
воркеры должны импортировать `cg_native` каждый у себя (см. тест ниже).

## Бенчмарк: native (arm64 dylib) vs Docker-эмуляция (x86_64 qemu)

Методология идентична `scripts/bench_engine.py` (те же шаги, random-vs-random,
deck = `submissions/v001_random/deck.csv`). MacBook Air, Apple Silicon (10 ядер: 4P+6E),
Python 3.11.15. Замер 2026-07-03 (`scripts/sim_native.py 60 50`).

| Метрика (1 процесс) | Docker-эмуляция | **Native dylib** | Ускорение |
|---|---|---|---|
| `battle_select`, шагов/сек | 3 728 | **9 320** | **×2.5** |
| Партий random-vs-random, шт/сек | ~81 (производное: 3728 / ~46 шагов) | **202** (12 147/мин) | **×2.5** |
| `search_step`, шагов/сек | 8 235 | **15 847** | **×1.9** |
| Полных rollout-до-терминала, шт/сек | ~200 | **285** (fresh battle) / **334** (PIMC: 100 rollout с одного корня) | **×1.4–1.7** |
| Время одного rollout, мс | ~5 | **3.5** | ×1.4 |
| **Multiprocessing, 4 процесса** | n/a (эмуляция душила) | **514 партий/сек (30 862/мин)** | ×2.5 к своему 1-proc |

Примечания:
- Шагов/партию: медиана 41, среднее 46 (Docker-замер видел медиану 34 — RNG движка не
  сидируется, каждая партия разной длины; порядок тот же).
- ~0.107 мс/ход native включает Python-оверхед (`to_observation_class` + json); чистый
  C-шаг ещё дешевле. Прежние «0.089 мс/ход» из sim-harness — другая метрика (время агента).
- Rollout ускорился меньше, чем сырой `search_step` — узкое место смещается в Python
  (json→dataclass на каждом шаге), а не в C-движок.
- Бюджет поиска: при медиане ~41 решения/партию и банке 600 с → ~14.6 с/ход ≈
  **~4 200 rollout/ход** в 1 процесс (в Docker было ~2 900 по той же формуле).

## Threading: ВСЁ ЕЩЁ НЕБЕЗОПАСНО → только multiprocessing

3 запуска `scripts/thread_check_native.py` (2 потока, каждый: своя партия + search-rollout,
изолированный subprocess):

- SIGSEGV **не** воспроизвёлся (на linux x86_64 либе падал), НО
- **каждый раз порча данных**: `JSONDecodeError` (гонка на общем буфере сериализации),
  `ValueError: Released item` (один поток снёс search-state другого — `agent_ptr` и
  `Battle.battle_ptr` глобальные в обёртке by design).

**Вердикт: НЕ thread-safe. Параллелизм — только multiprocessing** (проверено: 4 spawn-процесса
× 50 партий — все ок, 514 партий/сек суммарно).

## Детерминизм

Нет и не будет через публичный API: движок сидирует `std::mt19937` из `std::random_device`
(`Api.h`), а тасовка колоды (`CardMove.h:263`) использует `std::random_device` напрямую.
Экспортируемый `BattleStart(int* cards)` параметра seed не имеет. Python `random.seed`
влияет только на выбор действий агентом. Для A/B-сравнений — большие выборки, как раньше.

## Лицензия исходника движка (ВАЖНО)

`data/engine/ptcg_src/` — под `LicenseRef-PTCG-ABC-Competition-Use-Only`:

- использовать **только** для сборки/тестирования своих сабмитов этого соревнования,
  пока оно идёт; **не** распространять/публиковать; **удалить копии** после окончания;
- всё производное = "Pokémon Elements" (принадлежит Pokémon/Nintendo/Creatures/GAME FREAK);
- юридически главенствуют Competition Rules на Kaggle; **эксплуатация багов движка
  запрещена** правилами соревнования — исходник только для локальной верификации
  поведения (какие правила/эффекты как реализованы), не для поиска эксплойтов.

Из любопытного в исходнике (без rabbit holes): C++20 header-only, сборка — только VS2022
`game.sln` (Makefile/CMake нет; при большом желании `clang++ -shared -std=c++20 Export.cpp`
должен собрать dylib, но это не нужно — бинарники дал организатор); пул search-состояний
фиксирован — `std::array<State, 128>` на agent_ptr (`Search.h`), поэтому `search_release`/
`search_end` обязательны; в `GameConfig` существует внутренний `seed`/`deviceRand`, но
наружу через C-экспорт не выведен.

## Что ПО-ПРЕЖНЕМУ требует Docker (linux x86_64)

Runtime сабмита на Kaggle = **linux x86_64**. Нативный dylib — только для скорости
локальной разработки. Обязательно в Docker (x86_64):

1. **Финальная pre-submit валидация** каждого кандидата: тот самый `libcg.so`, что
   поедет в tar (для новых search-агентов — НОВЫЙ `libcg.so`, см. warning), проверка
   `exec(main.py)` без `__file__` (Kaggle-специфика), гейт vs чемпион.
2. Любые замеры time-bank'а «как на Kaggle» (нативные мс/ход НЕ переносимы на облако 1:1).
3. Периодическая перекалибровка Docker-цифр после смены движка.

Быстрые локальные итерации (self-play, PIMC-тюнинг, IL-генерация данных, гейты-черновики) —
теперь нативно. Финальный гейт перед сабмитом — Docker, как раньше.
