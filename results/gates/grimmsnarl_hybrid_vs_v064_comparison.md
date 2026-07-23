# v064 vs ext_prvsiyan_grimmsnarl_hybrid — сравнительный гейт (2026-07-23)

Разведочный прогон ("ради прикола"): наш v064_starmie_scaling против внешнего публичного агента prvsiyan/ptcg-rmy-grimmsnarl-replay-hybrid-v1 (attention-BC на 125 публичных реплеях Rmy Grimmsnarl/Froslass) по одному и тому же CORE_PAIRS пулу оппонентов, N=1000 на пару, cg_native движок.

Оба кандидата используют **одну и ту же деку Grimmsnarl** (naunju-стиль), поэтому сравнение по общим оппонентам — это прямое сравнение силы пилота, а не архетипа.

## Прямой head-to-head (v064 vs ext_prvsiyan_grimmsnarl_hybrid)

| Направление | WR | Wilson95 CI | N |
|---|---|---|---|
| v064 (candidate) vs hybrid, seed=v064-run | 0.398 | [0.368, 0.429] | n=1000 |
| hybrid (candidate) vs v064, seed=hybrid-run | 0.615 | [0.584, 0.645] | n=1000 |

Обе строки измеряют один и тот же матчап с независимыми сидами/раскладкой первого хода (seat-balanced внутри каждого прогона). Согласованность между строками — sanity-check методологии.

## Сравнительная таблица по общим оппонентам пула

| Оппонент | v064 WR [CI] | hybrid WR [CI] | Кто сильнее |
|---|---|---|---|
| v017_alakazam_heur | 0.965 [0.952, 0.975] | 0.982 [0.972, 0.989] | ≈ ничья |
| v016_alakazam | 0.977 [0.966, 0.985] | 0.988 [0.979, 0.993] | ≈ ничья |
| v009a_crustle | 0.219 [0.194, 0.246] | 0.440 [0.409, 0.470] | hybrid |
| v044_archaludon_abomasnow | 0.310 [0.282, 0.339] | 0.434 [0.404, 0.465] | hybrid |
| v049_bench_sticky_guard | 0.341 [0.312, 0.371] | 0.454 [0.423, 0.485] | hybrid |
| ext_prvsiyan_alak_v8 | 0.385 [0.355, 0.416] | 0.554 [0.523, 0.585] | hybrid |
| ext_prvsiyan_alak_v9 | 0.350 [0.321, 0.380] | 0.527 [0.496, 0.558] | hybrid |
| ext_maulikgajera_lucario | 0.227 [0.202, 0.254] | 0.374 [0.345, 0.404] | hybrid |
| ext_nursrijan_lucario | 0.244 [0.219, 0.272] | 0.403 [0.373, 0.433] | hybrid |

*(v064 WR — доля побед v064 против общего оппонента; hybrid WR — доля побед hybrid против того же общего оппонента; оба прогона N=1000, независимые. "Кто сильнее" — эвристическое сравнение WR против общего оппонента, не прямой h2h — см. раздел выше для прямого h2h.)*

## Диагностика 0/4 (первый смок-тест)

**Реальная причина (подтверждена через `faulthandler`, не гипотеза): конфликт OpenMP-рантаймов.**
Кандидат тянет `torch`, а `v017_alakazam_heur/main.py` на импорте инициализирует `lightgbm`
Booster — оба бандлят свой `libomp`, и при загрузке обоих в одном процессе второй
падает нативным abort'ом:

```
OMP: Error #179: Function pthread_mutex_init failed: System error #22: Invalid argument
Fatal Python error: Aborted
  File ".../lightgbm/basic.py", line 3698 in __init__
  File ".../v017_alakazam_heur/main.py", line 159 in <module>
  ...
```

Это настоящий процесс-левел abort (worker-процесс ProcessPoolExecutor убивается ОС),
`fut.result()` в гейте кидает `concurrent.futures.process.BrokenProcessPool` — что и
классифицируется гейтом как `worker_crash`/`crashes` в каждой из 4 игр смок-теста.
(Предыдущая версия этого файла содержала альтернативное объяснение — "гонка по
времени записи model.pt" — это НЕ подтвердилось: `fut.result()` в этом случае кидал
бы обычный `FileNotFoundError`, а не `BrokenProcessPool`; исправлено после
верификации через `faulthandler`.)

**Фикс**: `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` в env перед прогоном
(стандартный macOS-фикс двойной инициализации libomp) — после этого 0 крашей на
всех парах в обоих полных N=1000 прогонах.

Примечание: `scripts/diag_grimmsnarl_hybrid_trace.py` (ручная одно-процессная
трассировка модельного/fallback-пути) при повторном запуске без env-фикса зависает
на той же паре torch+lightgbm, не завершившись — числа "8/8 побед, 527/527 через
модельный путь", ранее заявленные в этом файле, не были воспроизведены и удалены
как неподтверждённые. Верификация того, что модель реально работает (а не тихо
падает в fallback), сделана иначе — через полный N=1000 прогон
(`ext_prvsiyan_grimmsnarl_hybrid_vs_pool.json`): 0 crashes/exceptions/illegal на
всех 12 парах, mirror-санити 48.8% (не константа/не вырожденный фоллбэк).
