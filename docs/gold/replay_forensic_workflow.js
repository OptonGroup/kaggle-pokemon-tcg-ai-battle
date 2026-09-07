export const meta = {
  name: 'v063-replay-forensic-92',
  description: 'Разбор всех 92 live-игр v063_forensic86: по одному агенту на эпизод (реплей-дайджест + decision-trace), затем агрегация причин',
  phases: [
    { title: 'Analyze', detail: 'один sonnet-агент на каждый эпизод', model: 'sonnet' },
  ],
}

const BASE = '/Users/max/kaggle-pokemon-tcg-ai-battle/out/v063_full_download'

const GAME_SCHEMA = {
  type: 'object',
  required: ['episode_id', 'result', 'opp_archetype', 'game_summary', 'primary_cause', 'cause_tags', 'fix_observations', 'confidence'],
  properties: {
    episode_id: { type: 'string' },
    result: { type: 'string', enum: ['win', 'loss', 'draw'] },
    opp_archetype: { type: 'string', description: 'архетип деки соперника из шапки дайджеста (Alakazam/Crustle/Starmie/Lucario/...)' },
    game_summary: { type: 'string', description: '2-4 предложения по-русски: как шла партия, где переломный момент' },
    primary_cause: { type: 'string', description: 'одно предложение по-русски: ГЛАВНАЯ причина исхода' },
    cause_tags: { type: 'array', items: { type: 'string' }, description: 'теги из словаря (можно несколько), нестандартное с префиксом other:' },
    misplays: {
      type: 'array',
      items: {
        type: 'object',
        required: ['what', 'severity'],
        properties: {
          turn: { type: 'integer' },
          step: { type: 'string', description: 'номер шага sNNN из дайджеста или stepNNN из трейса' },
          what: { type: 'string', description: 'что сделал агент не так (по-русски)' },
          better: { type: 'string', description: 'какая опция была лучше — ДОЛЖНА существовать в [alt]/[galt] списке этого решения' },
          evidence: { type: 'string', description: 'цитата [dec]/[alt] строки и/или шаги дайджеста' },
          severity: { type: 'string', enum: ['critical', 'moderate', 'minor'] },
        },
      },
    },
    good_plays: { type: 'array', items: { type: 'string' }, description: 'что агент сделал правильно и это повлияло на исход' },
    fix_observations: { type: 'array', items: { type: 'string' }, description: 'наблюдения по НОВЫМ механизмам v063 (см. список в промпте): что сработало / не сработало / дало побочный эффект, с цитатой-доказательством. Пустой массив, если ни один не проявился' },
    final_prizes: { type: 'string', description: 'счёт по призам в конце, формат "нам осталось X / оппоненту Y"' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

function resWord(res) {
  return res === 'win' ? 'ВЫИГРАЛ' : res === 'draw' ? 'сыграл ВНИЧЬЮ' : 'ПРОИГРАЛ'
}

function mkPrompt(e) {
  return `Ты — форензик-аналитик партий Pokemon TCG (Kaggle cabt engine). Разбери ОДНУ партию и верни структурированный вердикт: почему наш агент ${resWord(e.res)}.

## Файлы (читай оба полностью)
1. Читаемая хронология партии (дайджест реплея): ${BASE}/replay_digest/${e.ep}.txt
2. Decision-trace нашего агента (stderr): ${BASE}/agent_logs_extracted/${e.ep}-agent-${e.idx}.txt
3. (опционально, только если нужно копнуть глубже) сырой реплей: ${BASE}/replays/${e.ep}.json — JSON, steps[i][${e.idx}].observation = наше наблюдение (current=состояние борда, logs=события с прошлого шага).

Факты: episode_id=${e.ep}, наш player index=${e.idx}, исход для нас = ${e.res.toUpperCase()}, соперник "${e.opp}", копия сабмита ${e.copy}.

## Наша дека (jiatu.l Grimmsnarl toolbox)
3x Marnie's Grimmsnarl ex (главный атакер: абилка Punk Up — ускорение тёмной энергии из руки; Shadow Bullet 180 + 30 снайп по скамейке; отдаёт 2 приза при KO), линия 4x Marnie's Impidimp / 3x Morgrem (+3 Rare Candy), 4x Munkidori (Adrena-Brain: переносит 30 дамага с себя на любого покемона соперника, если на нём тёмная энергия), 2x Snorunt/2x Froslass (Freezing Shroud: дамаг за абилки соперника), 10x Dark Energy, саппорты: 4 Lillie's Determination, 4 TR Petrel, 2 Boss's Orders, 1 Dawn, 1 Xerosic, 1 Unfair Stamp; 4 Poffin, 4 Poke Pad, 3 Night Stretcher, 4 Spikemuth Gym, 1 Pokegear.

## Как читать decision-trace
- "stepNNN [dec] tT m=МАТЧАП ctxC nN -> #i 'ОПИСАНИЕ' s=SCORE | 2nd #j '...' pick=[...]" — выбранная опция, её score и второй кандидат. t=номер хода, n=число опций.
- "[alt]" — ПОЛНЫЙ список опций этого решения с их score (что агент МОГ выбрать).
- "[dbg] guard=ИМЯ pick=[...]" — сработал жёсткий guard (обходит скоринг): ko_lethal_gamewinning (наш летал), adrena_trigger/adrena_seq (снайп Adrena-Brain), munkidori_energy, energy_attach, promotion (выбор кого ставить в актив), loop_breaker (ротация при зацикливании).
- "[galt]" — опции в момент guard-решения (t=тип опции, не score).

## Пилот v063 — НОВЫЕ механизмы (проверяй по факту, НЕ принимай на веру, что они работают)
Это версия v062 + 8 правок по форензику. В отличие от v062, здесь ДОЛЖНЫ работать:
1. promotion-guard видит bench-слот 0 (в v062 был баг — здоровый Grimmsnarl ex в слоте 0 игнорировался);
2. приоритет промоушена учитывает заряженность (заряженная копия > пустой при том же тире);
3. поиск (Spikemuth/Dawn/Night Stretcher/Poke Pad) НЕ тащит мёртвые копии Grimmsnarl ex без эволюционной базы; при пустой скамейке тащит любого Basic;
4. emergency-bench: при bench==0 бонус +4000 на выставление базовых;
5. защитный своп на матчпойнте: если KO нашего актива отдаёт сопернику последние призы и есть замена — retreat скорится ~7000 (в v062 retreat всегда ~-100);
6. unstick: retreat +6000 для застрявших Munkidori/Snorunt/Froslass в активе; крепление энергии на застрявший 0-энергетический актив 5000;
7. loop_breaker: 3-й идентичный select подряд → ротация выбора ([dbg] guard=loop_breaker);
8. deck-out guard: при deckCount≤3 штраф −3000 на декопоисковые итемы.
В поле fix_observations перечисли, какие из этих механизмов в ЭТОЙ партии сработали (с цитатой), какие ДОЛЖНЫ были сработать, но не сработали (это критично — новая находка!), и любые их побочные эффекты (напр., защитный своп сработал не вовремя и слил темп).

## Что сделать
1. Прочитай дайджест: восстанови ход партии — кто ставил темп, размены призами, переломный момент. В дайджесте: "prz_left=N" = сколько призов ОСТАЛОСЬ добрать стороне (меньше = ближе к победе), [sNNN] = номер шага, KO/discard = нокаут, "took a PRIZE" = добрал приз. RESULT в конце с причиной.
2. Сопоставь с decision-trace: в ключевые моменты — что агент видел, какие опции были ([alt]), что выбрал и почему ([dec]/[dbg]).
3. Для ПРОИГРЫША (и ничьей) обязательно проверь гипотезы (каждую подтверди или отвергни доказательствами):
   a) no_defensive_retreat: наш повреждённый атакер (особенно Grimmsnarl ex, отдаёт 2 приза) остался в active в зоне OHKO соперника, при этом в [alt] был retreat/switch, а исход партии решили эти 2 приза (в v063 тут должен был сработать защитный своп — если не сработал, это находка);
   b) bench_depletion: скамейка опустела/осталась 1 фигура → проигрыш через "no Pokemon in Active" или через снайп Boss's Orders;
   c) missed_lethal: у нас существовал выигрывающий ход (летал/добор призов), но агент выбрал другое — ищи в [alt] отвергнутую опцию;
   d) matchup_wall: дека соперника структурно бьёт нашу (танки 330+hp, Rock Inn-блок Crustle, OHKO-скейлинг Alakazam Powerful Hand, heal-tank Starmie) — тогда misplays могут быть вторичны. Для Crustle и Starmie разбор ОСОБО тщательный: это две главные дыры v063, различай "структурная стена" vs "наши ошибки".
4. Для ПОБЕДЫ определи, ЧТО именно выиграло (наши guard-леталы? Adrena-объём по скамейке? соперник слил сам?).
5. БУДЬ СКЕПТИКОМ: "ошибка" = в [alt] этого решения РЕАЛЬНО была лучшая опция. Если руки/приход карт не оставляли выбора — это НЕ misplay, это dead draw / matchup. Не подгоняй под известные гипотезы.

## Словарь тегов
Поражение: L_no_defensive_retreat, L_bench_depletion, L_missed_lethal, L_wrong_attack_choice, L_energy_misallocation, L_wrong_promotion, L_slow_setup, L_dead_draw, L_matchup_wall, L_opp_hand_ohko, L_gust_snipe, L_deck_out, L_prize_race_fair, L_resource_waste, L_ability_denial, L_status_lock, L_step_limit.
Победа: W_lethal_found, W_adrena_volume, W_munkidori_engine, W_ability_punish, W_opp_misplay, W_opp_dead_draw, W_prize_race, W_gust_value, W_opp_slow_setup.
Можно добавлять other:<свободный_тег>. Текстовые поля — ПО-РУССКИ, цитаты логов оставляй как есть.

Верни структурированный результат по схеме. episode_id="${e.ep}", result="${e.res}".`
}

const EPISODES = typeof args === 'string' ? JSON.parse(args) : args

phase('Analyze')
log(`Разбор ${EPISODES.length} эпизодов v063 (по одному агенту на игру)`)

const results = await parallel(EPISODES.map(e => () =>
  agent(mkPrompt(e), {
    label: `${e.res === 'win' ? 'W' : e.res === 'draw' ? 'D' : 'L'}:${e.ep}:${e.opp || 'noname'}`.slice(0, 60),
    phase: 'Analyze',
    schema: GAME_SCHEMA,
    model: 'sonnet',
    effort: 'medium',
  })
))

const ok = results.filter(Boolean)
const failed = EPISODES.filter((e, i) => !results[i]).map(e => e.ep)
if (failed.length) log(`НЕ разобраны (агент упал/пропущен): ${failed.join(', ')}`)

const tagCounts = { win: {}, loss: {}, draw: {} }
for (const g of ok) {
  for (const t of g.cause_tags || []) {
    tagCounts[g.result][t] = (tagCounts[g.result][t] || 0) + 1
  }
}
const missedSeverity = { critical: 0, moderate: 0, minor: 0 }
for (const g of ok) for (const m of g.misplays || []) missedSeverity[m.severity] = (missedSeverity[m.severity] || 0) + 1

log(`Готово: ${ok.length}/${EPISODES.length} игр разобрано; misplays: ${JSON.stringify(missedSeverity)}`)
return { games: ok, failed, tag_counts: tagCounts, misplay_severity: missedSeverity }