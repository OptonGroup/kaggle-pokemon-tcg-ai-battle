# ExIt Plan — Expert Iteration против замороженного поля

> Единый план для WP1–WP5. Скелет живёт в `src/exit/` (аддитивно, ничего существующего
> не ломает). Северная звезда каждой итерации = **ВИНРЕЙТ ПРОТИВ ФИКСИРОВАННОГО ПОЛЯ**,
> не против себя. Это НЕ голый self-play, а expert-iteration против замороженного пула.

Дата рекона: 2026-07-01. Автор скелета: WP0 субагент.

---

## (а) Reuse-map — что уже есть и где переиспользовать

| Артефакт | Путь | Что делает | Готовность | Переиспользуется в |
|---|---|---|---|---|
| **Обёртки движка** | `src/cg/{game,api,sim,utils}.py` | `battle_start/select/finish`, `search_begin/step/end/release`, `to_observation_class`, `all_card_data`, `all_attack`; dataclasses Observation/Option/State/CardData | ✅ готово (Linux) | WP2, WP3 (базис) |
| **PIMC guided** | `src/pilot/pimc_guided.py` | PIMC с direct-call `_decide(obs)` из v012 эвристики как rollout policy; решает только 1й шаг MAIN | ✅ рабочий (лучший вариант) | **WP3 ядро** |
| PIMC v1 / v2 / hybrid | `src/pilot/{pimc,pimc_v2,pimc_hybrid}.py` | Предыдущие итерации PIMC (random rollout / guided / hybrid) | ✅ рабочие | WP3 (референс/фолбэк) |
| **Iono эвристика** | `src/agents/iono_heuristic.py` | Hand-written pilot Iono Electric Swarm (счёт по опциям, never-throw); ~0.70 vs Crustle | ✅ готово | WP1 (оппонент), WP3 (rollout policy) |
| **v014 сабмит** | `submissions/v014_pimc_pilot/main.py` | Self-contained PIMC v2 (smart candidates + heuristic rollout, mirror opp) | ✅ упакован | WP3 (образец упаковки) |
| **v012 сабмит** | `submissions/v012_ref/main.py` + `v012_iono_heur/` | Iono эвристика как сабмит (rollout policy для pimc_guided) | ✅ упакован | WP1 оппонент, WP3 rollout |
| **Фичи Iono** | `src/il/features_iono.py` | 66-мерные per-option фичи из raw obs_dict (без libcg) | ✅ готово | **WP4 feature_fn** |
| Фичи Crustle/v011 | `src/il/features_v011.py`, `features.py` (49d) | per-option фичи для Crustle/общие | ✅ готово | WP4 (др. архетипы) |
| **IL dataset builder** | `src/il/dataset.py` | (obs,action) → X/y/groups/weights, skill-weight, сплит по episode | ✅ готово | **WP4 train** |
| **IL train** | `src/il/train.py` | LightGBM lambdarank обучение ranker | ✅ готово | **WP4 train** |
| Iono IL пайплайн | `scripts/train_iono_il.py` | Фильтр Iono-слотов из реплеев → dataset → train → `models/il_iono_a.txt` | ✅ готово | WP4 (init pilot) |
| v011 IL пайплайн | `scripts/train_v011_il.py` | То же для Crustle/Typhlosion | ✅ готово | WP4 (др. архетип) |
| **Парсер реплеев** | `src/replays/parse.py` | `parse_replay()` → (obs,action,ctx,won,player); stdlib, работает на macOS | ✅ готово | WP4 (seed-данные) |
| **Турнир-харнесс** | `src/sim_harness.py` | `tournament()` + Wilson CI + move-time; `run_game_timed` | ✅ готово (1 процесс) | **WP2 образец game-loop** |
| Простой sim | `src/simulate.py` | `run_game`, `run_tournament` (базовый цикл) | ✅ готово | WP2 (базис) |
| **Бенч движка** | `scripts/bench_engine.py` | Замер скорости: партий/сек, search_step/сек, threading (SIGSEGV), детерминизм | ✅ готово | WP2/WP3 (профиль бюджета) |
| Валидатор PIMC | `scripts/validate_pimc.py` | Прогон PIMC vs v012/v009a + замер rollout-скорости | ✅ готово | WP2/WP3 (референс прогона) |
| **Gate runner** | `src/eval/gate_runner.py` + `scripts/gate.sh` | Candidate-vs-champion через importlib (CWD-трюк для deck.csv) | ✅ готово | WP5 (гейт) |
| Crustle recon | `src/eval/crustle_list_recon.py` | Реконструкция лучшего Crustle-листа из реплеев | ✅ готово | WP3 (`CRUSTLE_DECK_APPROX`) |
| **Docker** | `docker/Dockerfile` + `scripts/sim.sh` | linux/amd64 py3.11 + numpy/pandas/lightgbm, PYTHONPATH=src | ✅ готово | всё (run-story) |
| Sample агент | `data/sample_submission/{main.py,deck.csv,cg/}` | Организаторский random-агент + движок (.so + .dll) | ✅ есть | WP1 (референс контракта) |

**Ключевой вывод рекона:** ~80% строительных блоков ExIt-цикла УЖЕ написаны и рабочие.
`src/exit/` — это тонкий **оркестрирующий слой** поверх них, а не переписывание с нуля.

### Что НЕ дублировать
- PIMC — брать `src/pilot/pimc_guided.py`, НЕ писать новый (WP3 = параметризованный враппер).
- Фичи — брать `src/il/features_iono.py` (66d, из raw dict), НЕ изобретать.
- Обучение — брать `src/il/dataset.py` + `train.py`, НЕ писать свой LightGBM-пайплайн.
- Game-loop — брать логику из `src/sim_harness.py:run_game_timed`, обернуть в multiprocessing.

---

## (б) Тех-задания WP1–WP5

### WP1 — Пул оппонентов (`src/exit/opponents.py`)
**Статус скелета:** ✅ создан (Agent dataclass, `_load_agent_module`, реестр, `list_opponents`).
**Вход/выход:**
- `list_opponents(names=None, skip_errors=False) -> list[Agent]`.
- `Agent = {name: str, deck: list[int], policy: Callable[[dict], list[int]], mu: float, notes: str}`.
**Дела реализатора:**
1. `_make_lucario_heur` / `_make_crustle_il` / `_make_iono_heur` уже загружают через importlib —
   ПРОВЕРИТЬ в Docker что `mod.agent` вызывается без ошибок (нужен libcg).
2. `_make_random_agent` — заменить inline-fallback на реальный `agents.random_agent.agent`.
   ⚠️ NB: `agents.random_agent.agent` возвращает `[]` при deck-select (caller подаёт деку в
   `battle_start`), а наш harness сам подаёт деки — согласовать контракт (см. WP2).
3. Добавить sample Dragapult / Abomasnow, когда сохранены в `data/sample_agents/`.
4. Каждому оппоненту нужен **`main_py_path`** — путь к его main.py для передачи в worker
   (WP2 передаёт пути, не объекты). Добавить это поле в `Agent` (сейчас его нет — ДЫРА).
**Критерий приёмки:** `list_opponents()` в Docker возвращает ≥4 Agent-а, каждый отдаёт
валидную 60-карточную деку на первом вызове и не падает на 5 тестовых obs.

### WP2 — Харнесс массовых прогонов (`src/exit/harness.py`)
**Статус скелета:** ✅ создан (dataclasses Trajectory/GameResult/HarnessResult, `_worker_fn`,
`run_games`, `_wilson_ci`, `_aggregate_results` — тела помечены NotImplementedError + закомментированный референс).
**Вход/выход:**
- `run_games(candidate: Agent, opponents: list[Agent], n, workers, seed, collect_trajectories) -> HarnessResult`.
- `HarnessResult = {winrates: dict[str,float], game_results, trajectories, games_per_sec, wilson_lo/hi, total_games/wins}`.
**Дела реализатора:**
1. Реализовать `_worker_fn` (game-loop уже расписан в комментарии; взять из `sim_harness.run_game_timed`).
   Чередование первый/второй игрок обязательно (несмещённость).
2. Реализовать `run_games` через `concurrent.futures.ProcessPoolExecutor` (НЕ threading!).
3. Реализовать `_aggregate_results` (winrate per-opp + aggregate + Wilson).
4. Передавать агентов в worker КАК ПУТИ к main.py (pickle-safe), НЕ как замыкания/ctypes.
   → нужен `Agent.main_py_path` (см. WP1). Для in-memory PIMC/Pilot — сначала
   `pilot.export_to_submission()` во временный main.py.
5. Timeout по `MAX_GAME_STEPS` → ничья (защита от зависания).
**Критерий приёмки:** в Docker `run_games(random, [random], n=20, workers=2)` → winrate≈0.5,
0 исключений, games_per_sec>0, `len(trajectories)>0`. Multiprocessing без SIGSEGV.

### WP3 — PIMC-эксперт (`src/exit/pimc_agent.py`)
**Статус скелета:** ✅ создан (`PIMCAgent` класс с lazy-init над `pilot.pimc_guided`,
`detect_opponent_deck`, `make_pimc_agent` профили).
**Вход/выход:**
- `PIMCAgent(budget, rollout_depth, n_worlds_min, max_candidates, prior_agent, deck, opponent_model)`.
- `.agent(obs_dict) -> list[int]` (never-throw, совместим с Agent.policy).
**Дела реализатора:**
1. `_lazy_init` уже пробрасывает бюджет в `pimc_guided` — ПРОВЕРИТЬ в Docker что работает.
2. Реализовать подключение `prior_agent` как rollout policy (сейчас `pass` — ДЫРА; это точка
   интеграции WP3↔WP4: PIMC rollout = обученный pilot вместо эвристики).
3. Реализовать `detect_opponent_deck` (сейчас всегда fallback) — детект Crustle/Iono/Lucario
   по revealed картам; заполнить `CRUSTLE_DECK_APPROX` из `crustle_list_recon.py`.
4. Опционально: UCB-выбор кандидатов вместо round-robin; ранний выход при доминировании.
**Критерий приёмки:** в Docker `PIMCAgent(budget=1.0).agent(obs_dict)` возвращает валидное
действие; прогон vs iono_heur через WP2 не падает; `.stats` показывает rollouts>0.

### WP4 — BC-prior / pilot (`src/exit/pilot.py`)
**Статус скелета:** ✅ создан (`Pilot` класс: train/score/agent/save/load; `export_to_submission`).
**Вход/выход:**
- `Pilot(feature_fn, model_type, deck)`.
- `.train(trajectories, min_reward=1.0, skill_weight, num_trees, val_frac) -> dict метрик`.
- `.score(obs_dict) -> list[float]`; `.agent(obs_dict) -> list[int]` (never-throw).
- `export_to_submission(pilot, deck, out_dir, feature_module) -> str`.
**Дела реализатора:**
1. `train`: фильтр `final_reward >= min_reward` (**quality>volume** — только победы!),
   переиспользовать `src/il/dataset.py` (X/y/groups) + `src/il/train.py` (LightGBM).
2. `score`/`agent`: `feature_fn(obs_dict)` (=`features_iono.extract_features`) → `model.predict`.
3. `save`/`load`: LightGBM `.txt`. Ленивый import lightgbm/numpy.
4. `export_to_submission`: образец — `submissions/v009a_crustle/main.py` (**__file__ guard**
   обязателен — Kaggle грузит через exec() БЕЗ `__file__`, см. MEMORY).
**Критерий приёмки:** offline top-1 на held-out выше option-0 baseline; упакованный сабмит
проходит exec()-без-`__file__` тест; live winrate через WP2 измерим.

### WP5 — Драйвер ExIt (`src/exit/exit_loop.py`)
**Статус скелета:** ✅ создан (`passes_gate`, `run_iteration`, `run_exit_loop`, `_log_iteration`,
`IterationResult`).
**Вход/выход:**
- `run_iteration(pilot, pool, iteration, games_per_opp, workers, pimc_budget, min_reward, out_dir) -> (Pilot, IterationResult)`.
- `run_exit_loop(init_pilot, pool, n_iters, games_per_opp, workers, pimc_budget, log_path) -> Pilot`.
**Дела реализатора:**
1. `run_iteration`: собрать `PIMCAgent(prior_agent=pilot.agent)` (WP3) → `run_games` vs ЗАМОРОЖЕННЫЙ
   pool (WP2) → фильтр выигранных траекторий → `pilot.train` (WP4) → переоценка → гейт.
2. Anti-regression: не промоутить если winrate упал (урок v008).
3. `run_exit_loop`: цикл + best-tracking + лог в `results/exit_log.csv` + plateau early-stop.
**Критерий приёмки:** ≥1 итерация в Docker поднимает aggregate winrate против фикс. поля и
логирует `IterationResult`; гейт (55%/300/Wilson>50%) применяется корректно.

---

## (в) Docker / run-story (Linux-движок)

Движок `libcg.so` — **Linux x86_64 only**, на macOS ARM НЕ грузится. Всё, что реально гоняет
партии, — в Docker linux/amd64 или Kaggle-ноутбуке.

**Готово:**
- `docker/Dockerfile`: `FROM --platform=linux/amd64 python:3.11-slim`, ставит
  numpy/pandas/lightgbm, `COPY . /workspace`, `ENV PYTHONPATH=/workspace/src`.
- `scripts/sim.sh`, `scripts/gate.sh`, `scripts/probe.sh` — обёртки запуска в Docker.

**Как прогнать (типовой рецепт для WP2/WP3):**
```bash
# сборка образа (один раз)
docker build --platform linux/amd64 -t pokemon-tcg-sim:latest -f docker/Dockerfile .

# прогон произвольного скрипта внутри движка
docker run --rm --platform linux/amd64 \
  -v "$(pwd)":/workspace -e PYTHONPATH=/workspace/src \
  pokemon-tcg-sim:latest python /workspace/scripts/validate_pimc.py --games 20
```

**Замеры движка (из `scripts/bench_engine.py` / PLAN.md Q-SEARCH):**
- ~52 rollout/сек (ARM-эмуляция; native на Kaggle быстрее), ~88 шагов/rollout.
- **threading → SIGSEGV; только multiprocessing.**
- Бюджет: 600с/партию, ~120 решений → ~5с/решение; при budget=3с ≈ 80–150 rollouts/решение.

**Чего не хватает (ДЫРА):**
- В `Dockerfile` НЕТ `torch` (только numpy/pandas/lightgbm). Для WP4-LightGBM хватает; для
  MLP-pilot на torch — добавить `torch --index-url .../cpu` в образ.
- Нет отдельного скрипта запуска WP2-харнеса — реализатор WP2 добавит `scripts/run_field.sh`
  (по образцу `sim.sh`).
- Docker сам не запускался в этом реконе (macOS); опираемся на существующие result-файлы и
  прошлые прогоны из PLAN.md Status Log.

---

## (г) Границы параллельной волны

**Безопасно параллелить (разные файлы, нет конфликтов):**
- **WP1** (`opponents.py`) — самостоятельный; зависит только от существующих сабмитов.
- **WP3** (`pimc_agent.py`) — самостоятельный; обёртка над готовым `src/pilot/pimc_guided.py`.
- **WP4** (`pilot.py`) — самостоятельный; обёртка над готовыми `src/il/*`.

Эти три WP трогают РАЗНЫЕ файлы в `src/exit/` и разные существующие модули → **волна WP1‖WP3‖WP4
безопасна**. Общий контракт между ними — только dataclass `Agent` (уже зафиксирован в opponents.py).

**Конфликтуют / нужен порядок:**
- **WP2** (`harness.py`) зависит от `Agent.main_py_path` — поле надо добавить в `opponents.py`
  (WP1). Поэтому WP2 стартует после того как WP1 добавит `main_py_path` (мелкая правка) ИЛИ
  WP1+WP2 делает один исполнитель. WP2 — самый тяжёлый (game-loop + multiprocessing).
- **WP5** (`exit_loop.py`) — интегратор: зависит от ВСЕХ (WP1–WP4). Делается ПОСЛЕДНИМ, когда
  остальные проходят свои критерии приёмки.

**Рекомендованный порядок:** WP1 (быстро, +`main_py_path`) → волна {WP2, WP3, WP4} параллельно
→ WP5. Либо: WP1‖WP3‖WP4 сразу (первая параллельная волна), WP2 следом, WP5 в конце.

---

## (д) Риски / открытые вопросы

1. **`Agent.main_py_path` отсутствует** — WP2 передаёт агентов в worker-процессы как пути к
   файлам (pickle-safe), а текущий `Agent` хранит только `policy` (callable, не pickle-able
   с ctypes/замыканиями). Нужно добавить `main_py_path: str = ""` в dataclass. Помечено в WP1/WP2.
2. **In-memory агенты (PIMC/Pilot) для WP2** — их нельзя просто передать в worker. Нужен
   `export_to_submission` (WP4), чтобы материализовать во временный main.py, ИЛИ worker создаёт
   агента по конфигу внутри процесса. Развилка — решить в WP2.
3. **Модель оппонента = mirror** во всех PIMC (предполагаем Iono-дек оппонента). Против Crustle
   (85% поля) это неточно → `detect_opponent_deck` (WP3) важен для качества rollout-ов.
4. **quality>volume** (урок v008): в WP4 обязателен фильтр `final_reward>0`. Соблазн взять все
   траектории приведёт к регрессии.
5. **Throughput на ARM-эмуляции** медленный (~52 rollout/сек). Большие прогоны (300×N партий,
   PIMC budget≥3с) — только на Kaggle notebook (9ч сессии), Docker — для smoke/корректности.
6. **Детерминизм**: coin-флипы движка НЕ сидируются нашим seed (PLAN.md P2-3) → шум ±5% на
   300 партиях. Гейт-порог (Wilson_lo>50%) это учитывает.
7. **torch не в Docker-образе** — если WP4 выберет MLP на torch, обновить Dockerfile.
8. **Не запускать `kaggle submit`** без явного одобрения человека (правило проекта).
9. **Kaggle exec без `__file__`** — любой упакованный сабмит (WP4 export) обязан иметь
   `__file__` guard, иначе self-validation падает NameError (история v006).
