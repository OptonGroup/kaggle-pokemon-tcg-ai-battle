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
  добрать стороне; меньше = ближе к победе), блок `== ENDGAME ... ==` + строка `**** FINAL ****`.
  Медиана ~12КБ.
- **ПРОВЕРИТЬ целостность концовки** (иначе весь форензик слепой к развязке): `obs.logs`
  обновляется движком только когда агента ПРОМПТЯТ, поэтому игры, кончающиеся В ХОД СОПЕРНИКА
  (обычно >половины), не доходят до нашего наблюдения — replay_digest.py восстанавливает их из
  наблюдений оппонента (блок ENDGAME). Быстрый чек: `grep -L 'FINAL:' replay_digest/*.txt`
  (должно быть пусто) и `grep -c ENDGAME replay_digest/*.txt` (у ~50-65% игр = 1). Если у части
  игр обрыв на нашем последнем ходу без ENDGAME/FINAL — дайджест-генератор регрессировал,
  чинить ДО workflow (иначе субагенты не увидят, кто добрал последние призы).
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
- Схема ответа: episode_id, result (enum win|loss|**draw**), opp_archetype, game_summary (RU),
  primary_cause (RU), cause_tags[], misplays[{turn, step, what, better, evidence, severity}],
  good_plays[], final_prizes, confidence.
- **fix_observations[]** (опц., но добавлять при итеративном форензике): проверка СВОИХ
  прошлых правок. В промпт вписать нумерованный список новых механизмов текущей версии (из её
  DESIGN.md), попросить агента отметить с цитатой-доказательством, какие СРАБОТАЛИ, какие
  ДОЛЖНЫ были и НЕ сработали (это находка!), какие дали побочку. Так один прогон и разбирает
  поражения, и валидирует, что фиксы прошлой версии реально живут live.
- Словарь тегов — поражения: L_no_defensive_retreat, L_bench_depletion, L_missed_lethal,
  L_wrong_attack_choice, L_energy_misallocation, L_wrong_promotion, L_slow_setup, L_dead_draw,
  L_matchup_wall, L_opp_hand_ohko, L_gust_snipe, L_deck_out, L_prize_race_fair,
  L_resource_waste, L_ability_denial, L_status_lock, L_step_limit; победы: W_lethal_found,
  W_adrena_volume, W_munkidori_engine, W_ability_punish, W_opp_misplay, W_opp_dead_draw,
  W_prize_race, W_gust_value, W_opp_slow_setup; свободные с префиксом `other:`.

Референс готового скрипта: `docs/gold/replay_forensic_workflow.js` (рабочая версия с прогона
v062, 2026-07-22) — скопировать и поменять BASE-путь и факты деки.

## Шаг 4 — Синтез

1. **Собрать результаты из журнала workflow** (НЕ из notification — она усечена). Полные
   вердикты лежат в `<transcriptDir>/journal.jsonl`, по строке `{"type":"result",...}` на
   агента, где поле **`result` — Python-repr СТРОКА** (не JSON! `json.loads` не берёт — нужен
   `ast.literal_eval`). Готовый сборщик:
   ```python
   import json, ast, glob
   J = glob.glob('<...>/subagents/workflows/wf_*/journal.jsonl')[-1]  # путь из вывода Workflow
   games = []
   for line in open(J):
       rec = json.loads(line)
       if rec.get('type') != 'result': continue
       v = rec['result']
       v = v if isinstance(v, dict) else ast.literal_eval(v)   # repr-строка -> dict
       if isinstance(v, dict) and 'episode_id' in v: games.append(v)
   byep = {g['episode_id']: g for g in games}   # дедуп по эпизоду
   json.dump({'games': list(byep.values())}, open('out/<name>_download/forensic_games.json','w'),
             ensure_ascii=False, indent=1)
   ```
   **Session-limit fallback:** сверить число собранных с числом эпизодов; упавшие агенты
   (в failures notification или отсутствующие ep) — либо `Workflow({scriptPath, resumeFromRunId})`
   (кэш вернёт готовые, доиграет упавшие), либо разобрать 2-5 недостающих вручную (дайджест+трейс)
   и дописать в `games` тем же форматом с полем `analyst:"main-loop"`.
2. Агрегировать: частоты тегов по win/loss; **matchup-таблица по TRUE-deck** (архетип по скану
   деки оппонента из реплея, НЕ по opp_team_name): `arch = re.search(r'OPP archetype: (.+)$',
   open(f'replay_digest/{ep}.txt').read(400), re.M).group(1).strip()` → W-L + доминирующие
   причины на архетип; список
   critical-misplays с эпизодами. При наличии предыдущего форензика — дельта WR по архетипам.
3. Кросс-валидация: у КАЖДОЙ систематической причины (тег ≥3 игр) перечитать 1-2 названных
   эпизода лично (дайджест+трейс) — субагенты ошибаются; не строить ТЗ на одном агенте.
4. Написать `out/<name>_download/ANALYSIS_<version>.md`: сводка WR, matchup-таблица (+дельта к
   прошлой версии), ранжированный список причин поражений (с числом игр и доказательствами),
   причины побед, **статус своих прошлых правок** (из fix_observations), приоритизированные
   правки для следующей версии (что чинит сколько игр).

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
- **journal.jsonl: `result` — repr-строка, не JSON** — парсить `ast.literal_eval`, не `json.loads`
  (см. Шаг 4.1). `value`-поля там нет (частая ошибка).
- **Дайджест-эндшпиль:** `obs.logs` — залежавшееся окно на INACTIVE-шагах; наивный проход по
  всем шагам дублирует события и теряет ход, в котором игра кончилась (обычно ход соперника).
  replay_digest.py уже это чинит (fresh-окна + ENDGAME-хвост из наблюдений оппонента) — при
  правке генератора не сломать; проверять `grep -L FINAL: replay_digest/*.txt` пустой.
- Один прогон ≈ N sonnet-агентов; на 90+ играх часть может упасть по session-лимиту — это НЕ
  провал, добрать через resumeFromRunId или вручную (Шаг 4.1).
- **Имена архетипов в дайджесте могут содержать пробелы** — метки идут из общей таблицы
  `scripts/classify_opponent_archetype.py`, а незнакомая дека получает кластерное имя вида
  `auto:Mega Starmie ex + Cinderace`. Поэтому regex Шага 2 — `(.+)$` с `re.M`, а НЕ `(\S+)`:
  старый вариант резал такое имя по первому пробелу и склеивал разные деки в один бакет.
- **Метка ≠ дека.** Одна метка может покрывать несколько разных списков (исторический пример —
  `crustle`: настоящая стена Dwebble/Crustle и Great Tusk library-out). Если matchup-таблица
  показывает странный WR у метки, прогнать
  `classify_opponent_archetype.py <dir> --audit` и посмотреть разбивку по кластерам, прежде
  чем строить ТЗ на «мы проигрываем архетипу X».
