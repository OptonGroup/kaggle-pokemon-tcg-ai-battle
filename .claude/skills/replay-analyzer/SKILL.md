---
name: replay-analyzer
description: Массовый форензик live-игр с Kaggle — скачать реплеи+логи сабмита, превратить в читаемые дайджесты и прогнать по одному sonnet-субагенту на каждую игру, получив список причин побед/поражений с доказательствами. Использовать, когда юзер просит разобрать реплеи/эпизоды сабмита ("почему проиграл", "проанализируй игры vNNN", "форензик реплеев").
---

# Replay Analyzer — массовый форензик live-игр

Разбирает КАЖДУЮ игру сабмита отдельным субагентом (реплей + decision-trace нашего агента),
сводит причины побед/поражений в приоритизированный список с доказательствами. Результат —
готовое ТЗ для следующей версии агента.

Аргументы вызова: submission ref(ы) (напр. `54887093 54887100`) ИЛИ путь к уже скачанной
директории (`out/v062_full_download`). Если не переданы — спросить юзера или взять tracked-2
из results/submissions.csv.

## Шаг 1 — Скачать (пропустить, если директория уже есть)

```bash
.venv/bin/python scripts/download_episodes.py out/<name>_download A=<ref1> B=<ref2>
```

- Использует kagglesdk напрямую (CLI не берёт KGAT-токен), 429-бэкофф встроен, ≤2 потока.
- Выход: `replays/{ep}.json`, `agent_logs_raw/{ep}-agent-{idx}.json`, `index.csv`
  (колонки: episode_id, copy, our_index, our_reward, opp_team_name, opp_submission_id, ...).

## Шаг 2 — Извлечь decision-trace и сгенерировать дайджесты

```bash
python3 scripts/extract_agent_logs.py out/<name>_download    # -> agent_logs_extracted/*.txt
python3 scripts/replay_digest.py --batch out/<name>_download # -> replay_digest/*.txt
```

- ПРОВЕРИТЬ, что extracted-файлы непустые (`stat -f%z`); пустые = логи не нашего агента или
  сабмит без [dec]-логирования.
- Дайджест — читаемая хронология: шапка (архетипы дек, деклист соперника), события с именами
  карт ([sNNN] = шаг), снапшоты борда на каждый ход (prz_left = сколько призов ОСТАЛОСЬ
  добрать стороне; меньше = ближе к победе), RESULT с причиной. Медиана ~12КБ.
- Если дека сабмита не jiatu.l/Grimmsnarl — обновить ARCHETYPE_RULES в scripts/replay_digest.py
  при появлении новых архетипов.

## Шаг 3 — Workflow: по одному субагенту на игру

Собрать список эпизодов из index.csv в JSON (`ep, copy, idx=our_index, res=win|loss,
opp=opp_team_name`) и запустить Workflow-скрипт: фан-аут `parallel` по всем эпизодам,
**model='sonnet', effort='medium'** (правило проекта: субагенты только sonnet), каждый агент
получает schema-форсированный структурированный вывод.

ВАЖНО: в скрипте workflow парсить args устойчиво: `const EPISODES = typeof args === 'string'
? JSON.parse(args) : args` (args может прийти строкой).

Промпт агента (шаблон — подставить пути, факты эпизода и ОПИСАНИЕ НАШЕЙ ДЕКИ из
DESIGN.md текущего сабмита):

- Файлы: дайджест `replay_digest/{ep}.txt`, трейс `agent_logs_extracted/{ep}-agent-{idx}.txt`,
  сырой реплей `replays/{ep}.json` (только для углубления; steps[i][idx].observation).
- Как читать трейс: `[dec]` = выбранная опция + score + 2-й кандидат; `[alt]` = ВСЕ опции
  решения со score; `[dbg] guard=...` = сработал жёсткий guard в обход скоринга;
  `[galt]` = опции guard-решения.
- Для ПРОИГРЫША проверить гипотезы (подтвердить/отвергнуть с доказательствами):
  защитный ретрит не сделан / скамейка опустела / пропущен летал / стена матчапа.
- Для ПОБЕДЫ определить механизм (наш guard-летал? объём Adrena? соперник слил?).
- СКЕПСИС: "ошибка" только если в [alt] этого решения реально была лучшая опция;
  dead draw / отсутствие выбора = НЕ misplay.
- Схема ответа: episode_id, result, opp_archetype, game_summary (RU), primary_cause (RU),
  cause_tags[], misplays[{turn, step, what, better, evidence, severity}], good_plays[],
  final_prizes, confidence.
- Словарь тегов — поражения: L_no_defensive_retreat, L_bench_depletion, L_missed_lethal,
  L_wrong_attack_choice, L_energy_misallocation, L_wrong_promotion, L_slow_setup, L_dead_draw,
  L_matchup_wall, L_opp_hand_ohko, L_gust_snipe, L_deck_out, L_prize_race_fair,
  L_resource_waste, L_ability_denial, L_status_lock, L_step_limit; победы: W_lethal_found,
  W_adrena_volume, W_munkidori_engine, W_ability_punish, W_opp_misplay, W_opp_dead_draw,
  W_prize_race, W_gust_value, W_opp_slow_setup; свободные с префиксом `other:`.

Референс готового скрипта: `docs/gold/replay_forensic_workflow.js` (рабочая версия с прогона
v062, 2026-07-22) — скопировать и поменять BASE-путь и факты деки.

## Шаг 4 — Синтез

1. Сохранить сырые результаты workflow в `out/<name>_download/forensic_games.json`.
2. Агрегировать: частоты тегов по win/loss, разрез по архетипам соперника (matchup-таблица
   W-L + доминирующие причины), список critical-misplays с эпизодами.
3. Кросс-валидация: у КАЖДОЙ систематической причины (тег ≥3 игр) перечитать 1-2 названных
   эпизода лично (дайджест+трейс) — субагенты ошибаются; не строить ТЗ на одном агенте.
4. Написать `out/<name>_download/ANALYSIS_<version>.md`: сводка WR, matchup-таблица,
   ранжированный список причин поражений (с числом игр и доказательствами), причины побед,
   приоритизированные правки для следующей версии (что чинит сколько игр).

## Шаг 5 — Отчёт юзеру

Кратко: WR и разбивка, топ-3 причины поражений с числами, что предлагается править,
ссылка на ANALYSIS-файл. НЕ сабмитить ничего на Kaggle без явного "ок".

## Грабли

- `agent_logs_raw/*.json`: шаг = СПИСОК словарей (по агенту), не словарь — extract_agent_logs.py
  это учитывает; самописные фильтры по `item.get("stderr")` дадут пустые файлы.
- Один эпизод ~3.3МБ реплея — субагентам давать дайджест, а не сырой JSON.
- Повторные скачивания идемпотентны (existing files skipped) — безопасно дозапускать.
- prz_left в дайджесте = призы, которые стороне ОСТАЛОСЬ добрать (не "потеряла").
- Сabt-движок может дописывать поля/enum'ы — при странных type-кодах свериться с
  src/cg/api.py (LogType/AreaType/SelectContext).
