"""WP3 — PIMC-эксперт (Perfect Information Monte Carlo).

Детерминизирует скрытую информацию (рука/дека оппонента) → запускает rollout
через движок → выбирает действие с максимальной ожидаемой победой.

PIMC = Perfect Information Monte Carlo:
  1. Сэмплируем K "миров" (детерминизаций скрытого состояния оппонента).
  2. В каждом мире вызываем search_begin + search_step для оценки кандидатов.
  3. Rollout до конца (или до MAX_ROLLOUT_DEPTH) эвристической политикой.
  4. Выбираем кандидата с лучшим средним результатом по всем мирам.

НА ЧЁМ УЖЕ ПОСТРОЕНО (из src/pilot/):
  - pimc.py       — v1: случайный rollout, smart_candidates по типу действия
  - pimc_v2.py    — v2: guided rollout через v012 эвристику
  - pimc_hybrid.py — hybrid: PIMC только для MAIN ATTACK decisions, эвристика для остальных
  - pimc_guided.py — guided: PIMC с direct call _decide(obs) из v012 (лучший на сегодня)

PIMCAgent в этом файле — ПАРАМЕТРИЗОВАННЫЙ враппер над src/pilot/pimc_guided.py
(или другой выбранной реализацией) с настраиваемым бюджетом.

КЛЮЧЕВЫЕ ТЕХНИЧЕСКИЕ ФАКТЫ:
  - search_begin/search_step — НЕ MCTS, это симулятор детерминизированного мира.
  - threading → SIGSEGV. Только multiprocessing.
  - Один rollout ≈ 88 шагов ≈ 19.3 ms (ARM эмуляция). Native faster.
  - Бюджет: 600s / ~120 решений/партию ≈ 5s/решение. Но реально gate показывает
    ~8ms/ход на v003 (почти без поиска). С budget=3s можно ≈80-150 rollouts/решение.
  - Модель оппонента: opponent_model="mirror"|"crustle"|"auto"|"meta" — меняет prior
    для детерминизации. "auto"/"meta" (S2, 2026-07-05): накапливают revealed card ID
    оппонента за партию, лочатся на архетип при однозначной сигнатуре, а до этого
    сэмплируют дек оппонента ПО ОТДЕЛЬНОМУ МИРУ детерминизации из META_PRIORS
    (реальные доли топ-100 меты на 2026-07-05: Alakazam 33%, Archaludon 22%,
    Cubchoo 7%, Grimmsnarl 7%, Crustle 5%, остальное generic).
  - S3 bank-manager (2026-07-05): `budget=` в конструкторе — это ПОТОЛОК/дефолт для
    _lazy_init (нужен для смоков с фикс. бюджетом), но КАЖДОЕ реальное решение
    получает свой АДАПТИВНЫЙ budget_this из PIMCAgent._compute_budget() —
    (BANK_TOTAL_SEC*BANK_SAFETY - self._bank_spent) / est_remaining_decisions,
    зажатый в [MIN_DECISION_BUDGET, MAX_DECISION_BUDGET]. Плюс decision-gating
    (PIMCAgent._is_search_worthy): поиск запускается ТОЛЬКО на MAIN-развилках
    с >=2 нетривиальными опциями — всё остальное (форсировано/YES_NO/COUNT/
    setup/guard) уходит дешёвым путём (_cheap_decision: prior_agent, иначе
    локальный тривиальный выбор) БЕЗ _lazy_init/поиска вообще. См. PLAN.md §S3.

TODO для WP3+ (следующие итерации):
  [ ] UCB-выбор следующего кандидата вместо round-robin (EXP3/UCB1 по типу).
  [ ] Ранний выход если один кандидат доминирует (разрыв > порога, все N_WORLDS_MIN набраны).
  [ ] Обновлять reveal-базу нарастающим итогом (накапливать revealed IDs, не только текущий obs).
  [ ] Добавить LUCARIO_DECK_APPROX как только соберём достаточно реплеев.

DOCKER-ГЕЙТ (критерии приёмки, выполняются в Docker linux/amd64):
  Цель 1: PIMCAgent(budget=1.0).agent(obs_dict) возвращает валидное действие за <budget+margin сек.
  Цель 2: Winrate против random_agent ≥ 90% (за 50 партий).
  Цель 3: Winrate против v012_iono_heur (pilot baseline) > 0.5 у PIMCAgent(budget=3.0).
  Цель 4: Winrate против Crustle ≥ 0.70 у PIMCAgent(opponent_model="crustle", budget=3.0).
  Команда: docker run ... python scripts/validate_pimc.py --games 50 --budget 3.0

Пример использования (в Docker):
  from src.exit.pimc_agent import PIMCAgent, make_pimc_agent
  pimc = PIMCAgent(budget=3.0, rollout_depth=150, opponent_model="auto")
  action = pimc.agent(obs_dict)
  print(pimc.stats)   # rollouts_total, decisions_pimc, avg_worlds_per_decision, ...
"""

from __future__ import annotations

import os
import random
import time
from typing import Callable

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# S3 (2026-07-05) — bank-manager: адаптивный бюджет на решение из ОСТАВШЕГОСЯ
# банка партии + decision-gating (поиск только на осмысленных MAIN-развилках).
#
# Модель: у партии есть банк BANK_TOTAL_SEC секунд НА ВСЮ ПАРТИЮ (обе стороны
# независимо). SAFETY-доля банка держится в резерве (никогда не тратится на
# поиск) — таймаут = мгновенный проигрыш, поэтому лучше не досчитать rollout'ы,
# чем рискнуть. budget_this = (BANK_TOTAL*SAFETY - spent) / est_remaining_decisions,
# зажатый в [MIN_DECISION_BUDGET, MAX_DECISION_BUDGET]. spent считается по
# НАСТЕННОМУ времени (time.time() с начала партии) — это самонастраивается и в
# Docker (движок там ~2.5x медленнее native): budget_this там будет таким же,
# просто на него влезет меньше миров, а если резерв начнёт иссякать —
# est_remaining_decisions/spent автоматически ужмут budget_this вниз.
#
# Все параметры переопределяемы через env (для замеров в Docker и для S4-гейта)
# либо явно через конструктор PIMCAgent.
# ---------------------------------------------------------------------------

BANK_TOTAL_SEC_DEFAULT: float = float(os.environ.get("PIMC_BANK_TOTAL_SEC", "600.0"))
BANK_SAFETY_DEFAULT: float = float(os.environ.get("PIMC_BANK_SAFETY", "0.5"))
EST_TOTAL_DECISIONS_DEFAULT: int = int(os.environ.get("PIMC_EST_TOTAL_DECISIONS", "120"))
MIN_DECISION_BUDGET_DEFAULT: float = float(os.environ.get("PIMC_MIN_DECISION_BUDGET", "0.2"))
MAX_DECISION_BUDGET_DEFAULT: float = float(os.environ.get("PIMC_MAX_DECISION_BUDGET", "4.0"))
MIN_REMAINING_DECISIONS_FLOOR: int = int(os.environ.get("PIMC_MIN_REMAINING_DECISIONS", "5"))

# SelectType/SelectContext значения (см. cg_native/api.py) — дублируем как int-
# константы, чтобы не тянуть cg_native (Linux/native-only) в модуль, который
# должен импортироваться и на "холодную" (deck-select, до _lazy_init).
_SELECT_TYPE_COUNT: int = 8    # SelectType.COUNT
_SELECT_TYPE_YES_NO: int = 9   # SelectType.YES_NO
_SELECT_CONTEXT_MAIN: int = 0  # SelectContext.MAIN
_OPTION_TYPE_YES: int = 1      # OptionType.YES

# H3-3 (PLAN.md §H3-3, форум 723591: первый ход +10пп, 55.2/44.8): выбор
# "кто ходит первым" после монетки — отдельный SelectContext, YES_NO-тип.
# Подтверждено эмпирически (2026-07-11, native движок): первый select партии
# при выигрыше монетки = context=41 (IS_FIRST), type=9 (YES_NO),
# option=[{type=1(YES)}, {type=2(NO)}]. Отвечаем YES = "да, хочу первым".
_SELECT_CONTEXT_IS_FIRST: int = 41  # SelectContext.IS_FIRST

# OptionType int-константы (см. cg_native/api.py OptionType) — для generic
# greedy-attack rollout-политики (H3-1, оппонент без скриптованного пилота).
_OPTION_TYPE_PLAY: int = 7
_OPTION_TYPE_ATTACH: int = 8
_OPTION_TYPE_EVOLVE: int = 9
_OPTION_TYPE_ABILITY: int = 10
_OPTION_TYPE_RETREAT: int = 12
_OPTION_TYPE_ATTACK: int = 13
_OPTION_TYPE_END: int = 14
_OPTION_TYPE_CARD: int = 3
# Приоритет типов действия для greedy-fallback, когда нет ATTACK-опции
# (тот же порядок, что в src/pilot/pimc_guided.py._simple_fallback).
_GREEDY_TYPE_PRIORITY: tuple[int, ...] = (
    _OPTION_TYPE_ATTACK, _OPTION_TYPE_ABILITY, _OPTION_TYPE_EVOLVE,
    _OPTION_TYPE_PLAY, _OPTION_TYPE_ATTACH, _OPTION_TYPE_RETREAT,
    _OPTION_TYPE_CARD, _OPTION_TYPE_END,
)

# ---------------------------------------------------------------------------
# Deck константы
# ---------------------------------------------------------------------------

# Iono Electric Swarm — наш текущий основной дек
IONO_DECK: list[int] = [
    265, 265, 265,
    268, 268, 268,
    269, 269, 269,
    270, 270, 270,
    271, 271, 271,
    1233, 1233, 1233, 1233,
    1227, 1227, 1227, 1227,
    1086, 1086, 1086,
    1121, 1121, 1121,
    1254, 1254, 1254,
    1097, 1097,
    1152, 1152,
    1110,
    1118,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4,
]

# Crustle Rock Guard — из v009a (наш замороженный чемпион; поле МЕТА РОТИРОВАЛА
# 2026-07-03/07-05 — Crustle сейчас ~5% топ-100, НЕ 85%. См. META_PRIORS ниже).
# Реконструирован из submissions/v009a_crustle/deck.csv (сверено 2026-07-05, совпадает 1:1).
# При необходимости обновить из src/eval/crustle_list_recon.py (требует реплеев).
CRUSTLE_DECK_APPROX: list[int] = (
    [1] * 19   # Fighting Energy (id=1)
  + [11] * 4   # Arven (id=11)
  + [14] * 4   # (id=14)
  + [18] * 4   # (id=18)
  + [344] * 4  # Dwebble (id=344)
  + [345] * 4  # Crustle (id=345)
  + [1086] * 4 # Rare Candy / consistency trainer (id=1086)
  + [1147] * 4 # (id=1147)
  + [1159] * 1 # (id=1159)
  + [1212] * 4 # (id=1212)
  + [1227] * 4 # Nest Ball (id=1227)
  + [1235] * 4 # (id=1235)
)
assert len(CRUSTLE_DECK_APPROX) == 60, f"CRUSTLE_DECK_APPROX must be 60 cards, got {len(CRUSTLE_DECK_APPROX)}"


# Alakazam (aidy #8, μ1127.4) — PLAN.md §F1-3, canonical decklist reconstructed
# 2026-07-03 (8/8 stable). Топ-1 архетип текущей меты (~33% топ-100).
ALAKAZAM_DECK: list[int] = (
    [741] * 4    # Abra
  + [742] * 4    # Kadabra
  + [743] * 4    # Alakazam
  + [305] * 4    # Dunsparce
  + [66] * 3     # Dudunsparce
  + [1079] * 4   # Rare Candy
  + [1086] * 4   # Buddy-Buddy Poffin
  + [1152] * 4   # Poké Pad
  + [1081] * 4   # Enhanced Hammer
  + [1097] * 3   # Night Stretcher
  + [1129] * 1   # Sacred Ash
  + [1182] * 3   # Boss's Orders
  + [1231] * 4   # Dawn
  + [1225] * 4   # Hilda
  + [1184] * 1   # Lana's Aid
  + [1264] * 1   # Battle Cage
  + [5] * 3      # Basic {P} Energy
  + [19] * 4     # Telepath Psychic Energy
  + [13] * 1     # Enriching Energy
)
assert len(ALAKAZAM_DECK) == 60, f"ALAKAZAM_DECK must be 60 cards, got {len(ALAKAZAM_DECK)}"


# Grimmsnarl ex / Munkidori toolbox (kazuki0123 #3, μ1195, wr0.67) — PLAN.md §F1-5.
# ~7% топ-100 (но топ-3 по рейтингу).
GRIMMSNARL_DECK: list[int] = (
    [648] * 3    # Marnie's Grimmsnarl ex
  + [647] * 3    # Marnie's Morgrem
  + [646] * 3    # Marnie's Impidimp
  + [66] * 2     # Dudunsparce
  + [305] * 3    # Dunsparce
  + [649] * 1    # Marnie's Morpeko
  + [112] * 4    # Munkidori
  + [1159] * 1   # Hero's Cape
  + [1086] * 4   # Buddy-Buddy Poffin
  + [1079] * 3   # Rare Candy
  + [1152] * 4   # Poké Pad
  + [1231] * 4   # Dawn
  + [1227] * 4   # Lillie's Determination
  + [1197] * 2   # Xerosic's Machinations
  + [1097] * 2   # Night Stretcher
  + [1119] * 1   # Energy Search
  + [1139] * 1   # Energy Recycler
  + [1259] * 3   # Spikemuth Gym
  + [7] * 12     # Basic {D} Energy
)
assert len(GRIMMSNARL_DECK) == 60, f"GRIMMSNARL_DECK must be 60 cards, got {len(GRIMMSNARL_DECK)}"


# Cubchoo retreat-lock/mill (MtN #14, μ1094.4) — PLAN.md §F1-4. Комбо/лок-дека
# (BC на ней проваливается — win-con=deck-out, но для PIMC-детерминизации оппонента
# нам нужна ТОЛЬКО дека для сэмплинга скрытой информации, не клонирование пилота).
# ~7% топ-100 (но ~30% топ-10).
CUBCHOO_DECK: list[int] = (
    [506] * 4    # Cubchoo
  + [117] * 1    # Ogerpon ex (wall)
  + [414] * 1    # TR Articuno (wall)
  + [3] * 6      # Basic {W} Energy
  + [1166] * 3   # Gravity Gemstone
  + [1264] * 2   # Battle Cage
  + [1247] * 1   # Neutralization Zone
  + [1256] * 2   # TR Watchtower
  + [1182] * 4   # Boss's Orders
  + [1219] * 4   # TR Petrel
  + [1227] * 4   # Lillie's Determination
  + [1194] * 2   # Colress's Tenacity
  + [1197] * 2   # Xerosic's Machinations
  + [1086] * 2   # Buddy-Buddy Poffin
  + [1120] * 4   # Crushing Hammer
  + [1081] * 4   # Enhanced Hammer
  + [1119] * 3   # Energy Search
  + [1097] * 3   # Night Stretcher
  + [1152] * 2   # Poké Pad
  + [1122] * 4   # Pokégear 3.0
  + [1137] * 1   # Tool Scrapper
  + [1123] * 1   # Switch
)
assert len(CUBCHOO_DECK) == 60, f"CUBCHOO_DECK must be 60 cards, got {len(CUBCHOO_DECK)}"


# ---------------------------------------------------------------------------
# Card IDs известных архетипов (для детектора)
# ---------------------------------------------------------------------------

_CRUSTLE_SIGNATURE_IDS: set[int]    = {344, 345}          # Dwebble, Crustle
_IONO_SIGNATURE_IDS: set[int]       = {265, 268, 269, 270, 271}  # Voltorb, Tadbulb, Bellibolt, Wattrel, Kilowattrel
_LUCARIO_SIGNATURE_IDS: set[int]    = {673, 674, 675, 676, 677, 678}  # Riolu family, Lucario ex (устарело, вне топ-меты)
_ALAKAZAM_SIGNATURE_IDS: set[int]   = {741, 742, 743}     # Abra, Kadabra, Alakazam
_GRIMMSNARL_SIGNATURE_IDS: set[int] = {646, 647, 648}     # Marnie's Impidimp/Morgrem/Grimmsnarl ex
_CUBCHOO_SIGNATURE_IDS: set[int]    = {506}               # Cubchoo

# Порядок проверки в identify_archetype() — сигнатуры почти не пересекаются между
# архетипами, порядок влияет только на редкие мультиколлизии (напр. чужая discard-карта
# с тем же ID по совпадению). Crustle первым для обратной совместимости со старым
# detect_opponent_deck().
_ARCHETYPE_SIGNATURES: dict[str, set[int]] = {
    "crustle": _CRUSTLE_SIGNATURE_IDS,
    "alakazam": _ALAKAZAM_SIGNATURE_IDS,
    "grimmsnarl": _GRIMMSNARL_SIGNATURE_IDS,
    "cubchoo": _CUBCHOO_SIGNATURE_IDS,
    "iono": _IONO_SIGNATURE_IDS,
}


# ---------------------------------------------------------------------------
# Мета-приоры (S2, 2026-07-05) — реальные доли топ-100 текущей меты.
# Archaludon-metal (~22% поля) НЕ реконструирован (нет данных для 60-card листа) —
# TODO(S2+): реконструировать Archaludon decklist из реплеев, как это было сделано
# для Alakazam/Grimmsnarl/Cubchoo (см. PLAN.md §F1-3..F1-5). До тех пор его доля
# (и остаток поля "generic": Lucario/random/нишевые архетипы) сэмплируется как
# CRUSTLE_DECK_APPROX — нейтральный midrange-плейсхолдер, а НЕ фейковая дека.
# ---------------------------------------------------------------------------

META_PRIORS: dict[str, float] = {
    "alakazam":   0.33,
    "archaludon": 0.22,  # TODO: нет реконструированного дека -> плейсхолдер CRUSTLE_DECK_APPROX
    "cubchoo":    0.07,
    "grimmsnarl": 0.07,
    "crustle":    0.05,
    "generic":    0.26,  # остаток поля -> плейсхолдер CRUSTLE_DECK_APPROX
}
assert abs(sum(META_PRIORS.values()) - 1.0) < 1e-9, "META_PRIORS должны суммироваться в 1.0"


def _deck_for_archetype(name: str | None) -> list[int]:
    """Дек-константа по имени архетипа.

    "archaludon", "generic", None и любой неопознанный ключ -> нейтральный
    плейсхолдер CRUSTLE_DECK_APPROX (см. TODO у META_PRIORS выше).
    """
    if name == "alakazam":
        return list(ALAKAZAM_DECK)
    if name == "grimmsnarl":
        return list(GRIMMSNARL_DECK)
    if name == "cubchoo":
        return list(CUBCHOO_DECK)
    if name == "crustle":
        return list(CRUSTLE_DECK_APPROX)
    if name == "iono":
        return list(IONO_DECK)
    return list(CRUSTLE_DECK_APPROX)


def identify_archetype(revealed: set[int]) -> str | None:
    """Определяет архетип оппонента по накопленному множеству revealed card ID.

    Args:
        revealed: множество card ID, вскрытых на стороне ОППОНЕНТА (накопленное
                  за партию, см. PIMCAgent._revealed_opp_ids).

    Returns:
        Имя архетипа ("alakazam", "grimmsnarl", "cubchoo", "crustle", "iono")
        при однозначном совпадении сигнатуры, иначе None (мета-микс).
    """
    if not revealed:
        return None
    for name, sig in _ARCHETYPE_SIGNATURES.items():
        if revealed & sig:
            return name
    return None


def sample_meta_archetype(rng: random.Random | None = None) -> str:
    """Сэмплит имя архетипа оппонента по взвешенным META_PRIORS."""
    names = list(META_PRIORS.keys())
    weights = list(META_PRIORS.values())
    if rng is not None:
        return rng.choices(names, weights=weights, k=1)[0]
    return random.choices(names, weights=weights, k=1)[0]


def sample_meta_deck(rng: random.Random | None = None) -> list[int]:
    """Сэмплит ПОЛНЫЙ дек оппонента (60 ID) по META_PRIORS — один мир детерминизации."""
    return _deck_for_archetype(sample_meta_archetype(rng))


# ---------------------------------------------------------------------------
# H3-1 (PLAN.md §H3-1) — rollout-политика ОППОНЕНТА по архетипу МИРА.
#
# S4b root cause: rollout ОБЕИХ сторон играл Crustle-IL → PIMC оценивало ходы
# оппонента так, будто он играет Crustle-style, что систематически неверно
# для Alakazam/Grimmsnarl (−0.40/−0.15 delta vs field). Фикс: rollout
# оппонента использует ДЕШЁВЫЙ СКРИПТОВЫЙ ПИЛОТ конкретного архетипа —
# ТОГО, что реально был сэмплирован ДЛЯ ЭТОГО МИРА детерминизации (не
# "detected"-архетип с прошлого хода, а деку, которую _build_search_inputs
# только что подставил в ИМЕННО ЭТОТ rollout, см. PIMCAgentV2._opp_archetype_ref
# и _make_dual_policy_rollout).
#
# Ключевой нюанс: "archaludon" и "generic" в META_PRIORS/_deck_for_archetype
# ПОКА не имеют собственного реконструированного дека (TODO там же) — им
# подставляется CRUSTLE_DECK_APPROX как нейтральный плейсхолдер. Раз ДЕКА,
# которую реально играет мир, физически совпадает с Crustle, то и
# rollout-политика оппонента для этих архетипов = v009a (тот пилот, что
# умеет играть именно эту деку) — а НЕ generic-эвристика, которая была бы
# "по имени архетипа", а не "по деке" (см. формулировку задачи H3-1).
# Cubchoo/Iono имеют СОБСТВЕННЫЕ реконструированные деки, но НЕТ скриптового
# пилота под них -> generic greedy-attack fallback (единственный вариант,
# который не требует деки-специфичной логики).
# ---------------------------------------------------------------------------

_ARCHETYPE_TO_ROLLOUT_POLICY: dict[str | None, str] = {
    "alakazam":   "alakazam",
    "grimmsnarl": "grimmsnarl",
    "crustle":    "crustle",
    "archaludon": "crustle",   # деку подставляет CRUSTLE_DECK_APPROX (плейсхолдер)
    "generic":    "crustle",   # деку подставляет CRUSTLE_DECK_APPROX (плейсхолдер)
    "cubchoo":    "greedy",    # своя дека (CUBCHOO_DECK), но нет скриптового пилота
    "iono":       "greedy",    # своя дека (IONO_DECK), но нет скриптового пилота
    None:         "greedy",    # архетип мира не определён (не должно случаться в auto/meta)
}

# submissions/<dir>/main.py для каждого скриптового rollout-пилота оппонента.
_FIELD_PILOT_SUBDIRS: dict[str, str] = {
    "alakazam":   "v017_alakazam_heur",
    "grimmsnarl": "v018_grimmsnarl",
    "crustle":    "v009a_crustle",
}


def _rollout_policy_key_for_archetype(name: str | None) -> str:
    """Архетип сэмплированного мира -> ключ rollout-политики оппонента
    ("alakazam" | "grimmsnarl" | "crustle" | "greedy"), см. блок-комментарий
    выше. Неизвестные/будущие имена архетипов падают в "greedy" (безопасный
    дефолт), а не кидают исключение."""
    return _ARCHETYPE_TO_ROLLOUT_POLICY.get(name, "greedy")


_FIELD_PILOT_LOAD_COUNTER: int = 0


def _load_field_pilot_module(sub_dir_name: str):
    """Загружает submissions/<sub_dir_name>/main.py СВЕЖИМ instance'ом модуля
    (уникальное имя через монотонный счётчик — тот же паттерн, что
    scripts/pimc_field_gauntlet.py._load_submission_module /
    src/eval/anchored_pool.py._load_agent). Вызывающий код (PIMCAgentV2)
    обязан закэшировать результат САМ (self._opp_policy_cache) — эта функция
    НЕ кэширует, чтобы гарантировать "свежий модуль на игру" (см. stateful-
    заметку у _reset_stateful_globals) при этом не тратясь на повторную
    загрузку LightGBM Booster'а на каждый мир/rollout (дорого — Booster
    грузится с диска при exec_module).
    """
    global _FIELD_PILOT_LOAD_COUNTER
    _FIELD_PILOT_LOAD_COUNTER += 1
    import importlib.util
    import sys as _sys

    sub_dir = os.path.join(_REPO, "submissions", sub_dir_name)
    main_path = os.path.join(sub_dir, "main.py")
    if not os.path.exists(main_path):
        raise FileNotFoundError(main_path)
    if sub_dir not in _sys.path:
        _sys.path.insert(0, sub_dir)
    mod_name = f"_pimc_v2_field_pilot_{sub_dir_name}_{_FIELD_PILOT_LOAD_COUNTER}"
    spec = importlib.util.spec_from_file_location(mod_name, main_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = main_path  # guarded _DIR resolution in their main.py relies on this
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        raise AttributeError(f"No agent() in {main_path}")
    return mod


def _reset_stateful_globals(mod, policy_key: str) -> None:
    """H3-1 stateful-аудит (ВНИМАНИЕ на stateful-агентов, см. задание):

    v017_alakazam_heur: module-level globals — ТОЛЬКО диагностические
      счётчики (_n_model_decisions, _n_ko_guard, _n_heuristic_fallback,
      _n_trivial_fallback, _exception_count, _rule_fires). Ни один из них НЕ
      читается для принятия решений (grep подтверждает: только `+= 1`) ->
      БЕЗВРЕДНО переиспользовать загруженный модуль между мирами/играми без
      сброса.

    v009a_crustle: та же картина — только диагностические счётчики
      (_n_model_decisions, _n_heuristic_fallback, _n_trivial_fallback,
      _exception_count). БЕЗВРЕДНО.

    v018_grimmsnarl: `_adrena_state: str | None` — РЕАЛЬНАЯ state-machine
      (None | "await_from" | "await_count" | "await_target"), кодирующая
      многоходовую forced-continuation последовательность Adrena-Brain snipe
      (_adrena_trigger_guard устанавливает её, _adrena_sequence_guard читает
      И меняет на каждый вызов agent()). Если модуль переиспользуется между
      РАЗНЫМИ мирами детерминизации (каждый мир = для движка "новая игра",
      но модуль об этом не узнаёт — нет select=None вызова внутри rollout),
      остаточное _adrena_state из мира N могло бы просочиться в мир N+1.
      НА ПРАКТИКЕ это самоограничено (_adrena_sequence_guard сбрасывает
      _adrena_state=None на ЛЮБОМ неожиданном context, "any other context
      while a snipe is pending is an unexpected desync ... reset" — код
      main.py:629-633), т.е. максимум ОДНО лишнее решение в новом мире
      деградирует к обычной guard-цепочке. Но чтобы не полагаться на этот
      побочный эффект, мы ЯВНО сбрасываем `_adrena_state=None` ПЕРЕД КАЖДЫМ
      МИРОМ, где v018 используется как rollout-политика оппонента (дёшево —
      один set атрибута, вызывается из _make_dual_policy_rollout).

    Перезагрузка МОДУЛЯ целиком на каждый мир была бы безопаснее "по
    построению", но недопустимо дорого: все три модуля грузят LightGBM
    Booster с диска при exec_module (десятки ms) — при сотнях миров/решение
    это убило бы бюджет партии. Reset одного атрибута — O(1).
    """
    if policy_key == "grimmsnarl" and hasattr(mod, "_adrena_state"):
        mod._adrena_state = None


# ---------------------------------------------------------------------------
# H3-1 — generic greedy-attack fallback (архетипы без скриптового пилота:
# cubchoo/iono/неизвестный) + первый-легальный-ход / случайный-легальный-ход
# helpers, используемые и rollout-политикой СВОЕЙ стороны (ε-greedy), и как
# универсальный "деградация" fallback при исключении внутри ЛЮБОЙ политики.
# Все работают на Observation (dataclass, как отдаёт search_step), НЕ на dict.
# ---------------------------------------------------------------------------

_ATTACK_DAMAGE_CACHE: dict[int, int] | None = None


def _get_attack_damage_map() -> dict[int, int]:
    """attackId -> damage, из all_attack() (кэшируется — движок уже native-
    загружен к моменту первого вызова, т.к. это часть rollout, а не
    cold-import path)."""
    global _ATTACK_DAMAGE_CACHE
    if _ATTACK_DAMAGE_CACHE is None:
        try:
            from cg_native.api import all_attack
        except ImportError:
            from cg.api import all_attack
        _ATTACK_DAMAGE_CACHE = {a.attackId: (a.damage or 0) for a in all_attack()}
    return _ATTACK_DAMAGE_CACHE


def _obs_first_legal_choice(obs) -> list[int]:
    """Абсолютно надёжный legal fallback (H3-1: "любое исключение rollout-
    политики -> первый легальный ход"). НЕ пытается быть хорошим ходом —
    только легальным. Не трогает _v012_decide/prior/поиск/движок."""
    sel = obs.select
    if sel is None:
        return []
    n = len(sel.option)
    if n == 0:
        return []
    if sel.type == _SELECT_TYPE_YES_NO:
        for i, o in enumerate(sel.option):
            if o.type == _OPTION_TYPE_YES:
                return [i]
        return [0]
    if sel.type == _SELECT_TYPE_COUNT:
        return [0]
    min_c = max(0, sel.minCount)
    max_c = min(sel.maxCount, n)
    if max_c <= 0:
        return []
    k = min_c if min_c > 0 else 1
    k = min(k, max_c, n)
    return list(range(k))


def _obs_random_legal_choice(obs, rng: random.Random) -> list[int]:
    """Равномерно случайный ЛЕГАЛЬНЫЙ выбор (H3-1: ε-greedy на своей стороне
    rollout'а — с вероятностью ε вместо Crustle-prior берём это)."""
    sel = obs.select
    if sel is None:
        return []
    n = len(sel.option)
    if n == 0:
        return []
    if sel.type in (_SELECT_TYPE_YES_NO, _SELECT_TYPE_COUNT):
        return [rng.randrange(n)]
    min_c = max(0, sel.minCount)
    max_c = min(sel.maxCount, n)
    if max_c <= 0:
        return []
    lo = min(min_c, max_c)
    k = rng.randint(lo, max_c)
    if k <= 0:
        return []
    return rng.sample(range(n), k)


def _greedy_attack_choice(obs) -> list[int]:
    """Generic rollout-политика оппонента для архетипов БЕЗ скриптового
    пилота (cubchoo/iono/неопознанный, см. _ARCHETYPE_TO_ROLLOUT_POLICY):
    атаковать максимальным доступным уроном; иначе первый ход по приоритету
    типа действия (ATTACK > ABILITY > EVOLVE > PLAY > ATTACH > RETREAT >
    CARD > END, тот же порядок что pimc_guided._simple_fallback); иначе
    первый легальный ход."""
    sel = obs.select
    if sel is None:
        return []
    opts = sel.option
    n = len(opts)
    if n == 0:
        return []
    if sel.type == _SELECT_TYPE_YES_NO:
        for i, o in enumerate(opts):
            if o.type == _OPTION_TYPE_YES:
                return [i]
        return [0]
    if sel.type == _SELECT_TYPE_COUNT:
        best = max(range(n), key=lambda i: (opts[i].number or 0))
        return [best]
    if sel.context == _SELECT_CONTEXT_MAIN:
        atk_idxs = [i for i, o in enumerate(opts) if int(o.type) == _OPTION_TYPE_ATTACK]
        if atk_idxs:
            dmg_map = _get_attack_damage_map()
            best = max(atk_idxs, key=lambda i: dmg_map.get(opts[i].attackId, 0))
            return [best]
        for ot in _GREEDY_TYPE_PRIORITY:
            idxs = [i for i, o in enumerate(opts) if int(o.type) == ot]
            if idxs:
                return [idxs[0]]
    return _obs_first_legal_choice(obs)


def _choose_first_move_if_applicable(sel: dict) -> list[int] | None:
    """H3-3 (PLAN.md §H3-3, форум 723591: первый ход +10пп, 55.2/44.8):
    если это выбор "кто ходит первым" (SelectContext.IS_FIRST=41, YES_NO) —
    всегда отвечаем YES (пойти первым). Возвращает None если sel НЕ этот
    контекст (обычный decision-gating/поиск идёт как раньше, нетронутым).

    Подтверждено эмпирически (2026-07-11, native движок, battle_start):
    первый select партии при выигрыше монетки = context=41, type=9(YES_NO),
    option=[{type=1(YES)}, {type=2(NO)}].
    """
    if sel.get("context") != _SELECT_CONTEXT_IS_FIRST:
        return None
    if sel.get("type") != _SELECT_TYPE_YES_NO:
        return None
    opts = sel.get("option") or []
    for i, o in enumerate(opts):
        t = o.get("type") if isinstance(o, dict) else getattr(o, "type", None)
        if t == _OPTION_TYPE_YES:
            return [i]
    return [0] if opts else None


def _make_dual_policy_rollout(
    pimc_mod,
    own_choice_fn: Callable,
    opp_policy_cache: dict,
    opp_archetype_ref: list,
    degrade_counts: dict,
):
    """H3-1/H3-2: заменяет pimc_guided._do_guided_rollout (которое гоняло
    ОБЕ стороны через одну и ту же v012/prior-эвристику — root cause S4b)
    на rollout, где каждая сторона играет СВОЕЙ политикой:

      - СВОЯ сторона (obs.current.yourIndex == your_index корня): ε-greedy
        над own_choice_fn (Crustle-prior с вероятностью 1-ε, случайный
        легальный ход с вероятностью ε — см. PIMCAgentV2._make_own_choice_fn).
      - ОППОНЕНТ: политика по архетипу ТЕКУЩЕГО МИРА (opp_archetype_ref[0] —
        читается ОДИН РАЗ в начале rollout, т.к. архетип мира фиксирован в
        момент _build_search_inputs и не меняется в течение одного rollout).

    Любое исключение внутри own_choice_fn/опп.-политики -> _obs_first_legal_choice
    + инкремент degrade_counts (репортится в bank_stats/смок-статистике).

    opp_policy_cache: dict policy_key -> (choice_fn, module|None), лениво
    заполняется и ПЕРЕИСПОЛЬЗУЕТСЯ на все миры/решения ОДНОЙ игры (см.
    _load_field_pilot_module — почему не грузим модуль на каждый мир).
    """
    search_step = pimc_mod.search_step

    def _resolve_opp_policy(policy_key: str):
        cached = opp_policy_cache.get(policy_key)
        if cached is not None:
            return cached
        if policy_key == "greedy":
            entry = (_greedy_attack_choice, None)
        else:
            sub_dir_name = _FIELD_PILOT_SUBDIRS[policy_key]
            mod = _load_field_pilot_module(sub_dir_name)
            entry = (_wrap_dict_agent_as_obs_choice(mod.agent), mod)
        opp_policy_cache[policy_key] = entry
        return entry

    def _dual_rollout(search_id: int, obs_after, your_index: int) -> float:
        policy_key = _rollout_policy_key_for_archetype(opp_archetype_ref[0])
        try:
            opp_fn, opp_mod = _resolve_opp_policy(policy_key)
        except Exception:
            # Не удалось загрузить скриптовый пилот (напр. файл отсутствует) —
            # деградируем на generic greedy для ВСЕЙ этой rollout-политики,
            # не только на первый ход.
            degrade_counts[f"opp_load_fail_{policy_key}"] = (
                degrade_counts.get(f"opp_load_fail_{policy_key}", 0) + 1
            )
            opp_policy_cache[policy_key] = (_greedy_attack_choice, None)
            opp_fn, opp_mod = _greedy_attack_choice, None
        if opp_mod is not None:
            _reset_stateful_globals(opp_mod, policy_key)

        cur_obs = obs_after
        cur_sid = search_id
        max_depth = pimc_mod.MAX_ROLLOUT_DEPTH

        for _ in range(max_depth):
            if cur_obs.current and cur_obs.current.result != -1:
                r = cur_obs.current.result
                if r == your_index:
                    return 1.0
                elif r == 2:
                    return 0.0
                else:
                    return -1.0

            is_our_turn = (cur_obs.current.yourIndex == your_index)
            if is_our_turn:
                try:
                    choice = own_choice_fn(cur_obs)
                except Exception:
                    degrade_counts["own_exc"] = degrade_counts.get("own_exc", 0) + 1
                    choice = None
            else:
                try:
                    choice = opp_fn(cur_obs)
                except Exception:
                    key = f"opp_exc_{policy_key}"
                    degrade_counts[key] = degrade_counts.get(key, 0) + 1
                    choice = None

            if choice is None:
                choice = _obs_first_legal_choice(cur_obs)

            try:
                nxt = search_step(cur_sid, choice)
            except Exception:
                return 0.0

            cur_obs = nxt.observation
            cur_sid = nxt.searchId

        # Rollout не завершился за MAX_ROLLOUT_DEPTH шагов
        return 0.0

    return _dual_rollout


# ---------------------------------------------------------------------------
# Вспомогательные функции для monkey-patching pimc_guided
# ---------------------------------------------------------------------------

# Диагностические счётчики вызовов rollout-prior (сбрасываются вручную из теста).
_PRIOR_CALL_COUNT: int = 0
_PRIOR_EXC_COUNT: int = 0


def reset_prior_counters() -> None:
    global _PRIOR_CALL_COUNT, _PRIOR_EXC_COUNT
    _PRIOR_CALL_COUNT = 0
    _PRIOR_EXC_COUNT = 0


def _dataclass_obs_to_dict(obj):
    """Рекурсивно конвертирует Observation (dataclass) → dict/list/примитивы.

    Общий helper, вынесенный из _make_prior_decide_wrapper (H3): используется
    и обёрткой над СВОИМ prior_agent (crustle-IL), и обёрткой над
    ОППОНЕНТСКИМИ полевыми пилотами (v017/v018/v009a — все ожидают dict-
    контракт agent(obs_dict), а search_step отдаёт Observation dataclass).
    """
    import dataclasses

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _dataclass_obs_to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    elif isinstance(obj, list):
        return [_dataclass_obs_to_dict(item) for item in obj]
    elif hasattr(obj, 'value'):  # IntEnum
        return obj.value
    return obj


def _make_prior_decide_wrapper(
    prior_agent_fn: Callable[[dict], list[int]],
) -> Callable:
    """Создаёт _decide-совместимый враппер для произвольного prior_agent.

    pimc_guided._guided_choice_from_obs вызывает _v012_decide(obs: Observation).
    Мы конвертируем Observation → dict и вызываем prior_agent_fn(obs_dict).
    Это точка интеграции WP3 ↔ WP4: BC-prior вместо v012 эвристики.
    """
    def _prior_decide(obs):
        global _PRIOR_CALL_COUNT, _PRIOR_EXC_COUNT
        _PRIOR_CALL_COUNT += 1
        try:
            obs_dict = _dataclass_obs_to_dict(obs)
            return prior_agent_fn(obs_dict)
        except Exception:
            # ДИАГНОСТИКА: если IL-prior часто кидает внутри роллаута, политика
            # молча деградирует в тупую эвристику _guided_choice_from_obs →
            # оценки миров становятся нерелевантны реальной игре IL.
            _PRIOR_EXC_COUNT += 1
            return None

    return _prior_decide


def _wrap_dict_agent_as_obs_choice(agent_fn: Callable[[dict], list[int]]) -> Callable:
    """Как _make_prior_decide_wrapper, но БЕЗ глобальных _PRIOR_* счётчиков и
    БЕЗ внутреннего try/except (H3-1: у ОППОНЕНТСКИХ rollout-политик своя,
    раздельная бухгалтерия деградации — см. PIMCAgentV2.rollout_degradation /
    _make_dual_policy_rollout — которая иначе конфликтовала бы с prior'ом
    нашей стороны, если бы оба использовали одни и те же _PRIOR_* счётчики).
    Исключения намеренно НЕ ловятся здесь — их ловит вызывающий код
    (_make_dual_policy_rollout), чтобы посчитать деградацию по archetype-ключу.
    """
    def _choice(obs):
        obs_dict = _dataclass_obs_to_dict(obs)
        return agent_fn(obs_dict)

    return _choice


def _make_patched_build_search_inputs(
    pimc_mod,
    our_deck: list[int],
    opp_deck_ref: list,             # list[list[int]] — mutable single-element container
    opp_sampler_ref: list | None = None,  # list[Callable[[], list[int]] | None] — per-world sampler
) -> Callable:
    """Создаёт patched _build_search_inputs с заданной deck-парой.

    opp_deck_ref — обёртка [deck_list], позволяющая менять модель оппонента
    на лету (для "auto"/"meta" режима без повторного monkey-patching), когда
    архетип оппонента УЖЕ зафиксирован (locked) или зафиксирован явно ("mirror"/"crustle").

    opp_sampler_ref — обёртка [callable_or_None]. Когда задан callable (архетип ещё
    НЕ опознан в "auto"/"meta" режиме), он вызывается ЗАНОВО НА КАЖДЫЙ МИР
    детерминизации (эта функция вызывается ровно раз на мир из
    pimc_guided._evaluate_candidates), чтобы разные миры получали разные деки
    согласно META_PRIORS вместо одного и того же фиксированного дека. Когда
    opp_sampler_ref[0] is None — используется фиксированный opp_deck_ref[0].

    Копирует логику из pimc_guided._build_search_inputs, но использует
    our_deck (ваш дек) и дек оппонента (fixed либо per-world sampled) вместо IONO_DECK.
    """
    _sample_hidden = pimc_mod._sample_hidden
    _opp_known = pimc_mod._opp_known
    if opp_sampler_ref is None:
        opp_sampler_ref = [None]

    def _patched(obs, your_index):
        my = obs.current.players[your_index]
        opp_idx = 1 - your_index
        opp = obs.current.players[opp_idx]

        # Ваш дек
        ydc = my.deckCount
        your_deck = (list(our_deck) + [4] * 10)[:max(ydc, 0)]
        your_prize = [p.id if p is not None else 4 for p in (my.prize or [])]

        # Оппонент: per-world сэмпл (мета-микс, архетип не зафиксирован) либо
        # фиксированный дек из mutable ref (mirror/crustle/locked-auto).
        sampler = opp_sampler_ref[0]
        opp_full = list(sampler()) if sampler is not None else list(opp_deck_ref[0])
        opp_kn = _opp_known(obs, opp_idx)
        opp_deck_sampled = _sample_hidden(opp_kn, opp_full, opp.deckCount)

        opp_prize = []
        for p in (opp.prize or []):
            if p is not None:
                opp_prize.append(p.id)
            else:
                pool = [x for x in opp_full if x not in opp_prize]
                opp_prize.append(random.choice(pool) if pool else 4)

        opp_hand = _sample_hidden(opp_kn, opp_full, opp.handCount)

        return your_deck, your_prize, opp_deck_sampled, opp_prize, opp_hand

    return _patched


# ---------------------------------------------------------------------------
# Детектор деки оппонента по revealed картам
# ---------------------------------------------------------------------------

def _extract_revealed_ids(obs_dict: dict) -> set[int]:
    """Извлекает revealed card ID СТОРОНЫ ОППОНЕНТА (не своей!) из текущего obs_dict.

    Смотрит на discard, active, bench, прикреплённые энергии/карты и открытые prize
    оппонента. ВАЖНО (PLAN.md F1-2 баг): в отличие от src/replays/parse.py
    revealed_cards(), который мешает карты ОБОИХ игроков, здесь мы явно берём
    только players[opp_idx] — это то, что нужно для честной детерминизации
    (нельзя "видеть" свои же карты как сигнатуру оппонента).

    Не бросает исключений — при любой ошибке формата возвращает пустое множество.
    """
    revealed: set[int] = set()
    try:
        if not obs_dict:
            return revealed

        current = obs_dict.get("current") or {}
        players = current.get("players") or []
        if len(players) < 2:
            return revealed

        your_index = current.get("yourIndex", 0)
        opp_idx = 1 - your_index
        if opp_idx >= len(players):
            return revealed

        opp = players[opp_idx]

        # Discard pile — полностью открыта
        for c in (opp.get("discard") or []):
            cid = _extract_card_id(c)
            if cid:
                revealed.add(cid)

        # Активный Покемон и bench
        active_list = opp.get("active") or []
        bench_list = opp.get("bench") or []
        for poke in list(active_list) + list(bench_list):
            if poke is None:
                continue
            cid = _extract_card_id(poke)
            if cid:
                revealed.add(cid)
            # Прикреплённые энергии и карты
            for ec in (poke.get("energyCards") or []):
                ec_id = _extract_card_id(ec)
                if ec_id:
                    revealed.add(ec_id)

        # Prize cards (открытые — напр. после Boss's Orders/переворотов)
        for c in (opp.get("prize") or []):
            if c is not None:
                cid = _extract_card_id(c)
                if cid:
                    revealed.add(cid)

    except Exception:
        pass

    return revealed


def detect_opponent_deck(obs_dict: dict, fallback_deck: list[int] | None = None) -> list[int]:
    """Определяет вероятный дек оппонента по revealed картам ТЕКУЩЕГО obs_dict.

    Однократная (без памяти между ходами) обёртка над _extract_revealed_ids +
    identify_archetype — сохранена для обратной совместимости и разовых lookup'ов.
    Для накопления revealed card ID ЗА ПАРТИЮ и лока на архетип используйте
    PIMCAgent._revealed_opp_ids / PIMCAgent._locked_archetype (см. ниже) — это
    даёт гораздо более надёжную детекцию, чем single-obs snapshot.

    Меtagame prior (S2, 2026-07-05): поле РОТИРОВАЛО (Alakazam 33%, Archaludon 22%,
    Cubchoo/Grimmsnarl по 7%, Crustle всего 5%) — Crustle-fallback больше НЕ отражает
    большинство поля. fallback_deck по умолчанию остаётся CRUSTLE_DECK_APPROX как
    нейтральный midrange-плейсхолдер (см. META_PRIORS/_deck_for_archetype для
    полноценного взвешенного мета-микса).

    Args:
        obs_dict:      Текущий obs_dict из движка.
        fallback_deck: Дек по умолчанию если архетип не опознан по ЭТОМУ obs_dict.
                       Если None — используется CRUSTLE_DECK_APPROX (нейтральный плейсхолдер).

    Returns:
        list[int] из 60 card IDs — предполагаемый полный дек оппонента.
    """
    if fallback_deck is None:
        fallback_deck = list(CRUSTLE_DECK_APPROX)

    revealed = _extract_revealed_ids(obs_dict)
    if not revealed:
        return list(fallback_deck)

    arch = identify_archetype(revealed)
    if arch is not None:
        return _deck_for_archetype(arch)

    if revealed & _LUCARIO_SIGNATURE_IDS:
        # TODO: добавить LUCARIO_DECK_APPROX когда будут данные (вне топ-меты 2026-07-05)
        return list(fallback_deck)

    return list(fallback_deck)


def _local_trivial_choice(
    sel: dict, n_options: int, min_count: int, max_count: int
) -> list[int]:
    """Тривиальный выбор БЕЗ зависимости от pimc_mod/_lazy_init (S3 дешёвый путь).

    Зеркалит pimc_guided._trivial_choice, но работает напрямую с raw obs_dict
    (sel — dict, не Observation dataclass) — так cheap-путь decision-gating'а
    не тянет за собой инициализацию движка/PIMC модуля вообще.
    """
    if n_options == 0:
        return []
    if n_options == 1:
        return [0]
    opts = sel.get("option") or []
    sel_type = sel.get("type")
    if sel_type == _SELECT_TYPE_YES_NO:
        for i, o in enumerate(opts):
            if (o.get("type") if isinstance(o, dict) else getattr(o, "type", None)) == _OPTION_TYPE_YES:
                return [i]
        return [0]
    if sel_type == _SELECT_TYPE_COUNT:
        def _num(o):
            return (o.get("number") if isinstance(o, dict) else getattr(o, "number", None)) or 0
        best = max(range(n_options), key=lambda i: _num(opts[i]))
        return [best]
    k = max(1, min_count)
    k = min(k, max_count, n_options)
    return list(range(k))


def _extract_card_id(card_or_dict) -> int | None:
    """Извлекает card ID из dict или объекта с атрибутом .id."""
    if card_or_dict is None:
        return None
    if isinstance(card_or_dict, dict):
        return card_or_dict.get("id") or None
    cid = getattr(card_or_dict, "id", None)
    return cid if cid else None


# ---------------------------------------------------------------------------
# PIMCAgent
# ---------------------------------------------------------------------------

class PIMCAgent:
    """PIMC-агент с настраиваемым бюджетом и rollout политикой.

    Параметризованная обёртка над src/pilot/pimc_guided.py (текущий лучший вариант).
    Предоставляет .agent callable для совместимости с Agent.policy контрактом.

    Args:
        budget:          Бюджет на ход в секундах (TIME_BUDGET_SEC).
        rollout_depth:   Максимальная глубина rollout (MAX_ROLLOUT_DEPTH).
        n_worlds_min:    Минимум миров на кандидата до ранней остановки (N_WORLDS_MIN).
        max_candidates:  Максимум кандидатов на MAIN select (MAX_CANDIDATES).
        prior_agent:     Опциональный BC-prior агент для rollout policy (WP3↔WP4).
                         Callable[[dict], list[int]] — тот же контракт что и agent().
                         Если None — использует эвристику v012.
        deck:            Дек кандидата (60 card IDs). Нужен для determinization.
                         Если None — используется IONO_DECK.
        opponent_model:  "mirror" | "crustle" | "auto" | "meta".
                         "mirror":  оппонент играет наш же дек.
                         "crustle": оппонент играет CRUSTLE_DECK_APPROX (фиксированно).
                         "auto":    накапливает revealed card ID оппонента ЗА ПАРТИЮ
                                    (self._revealed_opp_ids), лочится на архетип
                                    (ALAKAZAM/GRIMMSNARL/CUBCHOO/CRUSTLE_DECK) как только
                                    сигнатура однозначна; до лока — сэмплирует дек
                                    оппонента ПО КАЖДОМУ МИРУ детерминизации из
                                    META_PRIORS (реальные доли меты 2026-07-05).
                         "meta":    синоним "auto" (взвешенный мета-микс) — оставлен как
                                    явное имя режима по запросу S2.

    Атрибуты:
        agent:  Callable[[dict], list[int]] — entry point для движка (never-throw).
        stats:  dict — статистика последних вызовов (делегируется в pimc_guided).
        _revealed_opp_ids:  set[int] — накопленные за партию revealed ID стороны оппонента
                             (только "auto"/"meta").
        _locked_archetype:  str | None — архетип, на который залочились ("alakazam" и т.п.),
                             либо None пока не опознан.

    Пример:
        pimc = PIMCAgent(budget=3.0, opponent_model="auto")
        action = pimc.agent(obs_dict)
    """

    def __init__(
        self,
        budget: float = 3.0,
        rollout_depth: int = 150,
        n_worlds_min: int = 3,
        max_candidates: int = 8,
        prior_agent: Callable[[dict], list[int]] | None = None,
        deck: list[int] | None = None,
        opponent_model: str = "mirror",
        bank_total_sec: float | None = None,
        bank_safety: float | None = None,
        est_total_decisions: int | None = None,
        min_decision_budget: float | None = None,
        max_decision_budget: float | None = None,
    ):
        if opponent_model not in ("mirror", "crustle", "auto", "meta"):
            raise ValueError(
                f"opponent_model must be 'mirror', 'crustle', 'auto' or 'meta', got {opponent_model!r}"
            )

        self.budget = budget
        self.rollout_depth = rollout_depth
        self.n_worlds_min = n_worlds_min
        self.max_candidates = max_candidates
        self.prior_agent = prior_agent
        self.deck = deck if deck is not None else list(IONO_DECK)
        self.opponent_model = opponent_model

        # --- S3 bank-manager (см. блок констант вверху файла) ---------------
        # budget (аргумент выше) остаётся ПОТОЛКОМ/дефолтом для явного использования
        # (напр. смок-тестов с фикс. budget=0.3), а адаптивный per-decision бюджет
        # (_compute_budget) выставляется НА КАЖДЫЙ вызов .agent() и ограничен
        # [min_decision_budget, max_decision_budget] снизу/сверху.
        self.bank_total_sec = bank_total_sec if bank_total_sec is not None else BANK_TOTAL_SEC_DEFAULT
        self.bank_safety = bank_safety if bank_safety is not None else BANK_SAFETY_DEFAULT
        self.est_total_decisions = (
            est_total_decisions if est_total_decisions is not None else EST_TOTAL_DECISIONS_DEFAULT
        )
        self.min_decision_budget = (
            min_decision_budget if min_decision_budget is not None else MIN_DECISION_BUDGET_DEFAULT
        )
        self.max_decision_budget = (
            max_decision_budget if max_decision_budget is not None else MAX_DECISION_BUDGET_DEFAULT
        )
        # Метка времени начала партии (сбрасывается на deck-select, чисто
        # информационная — НЕ используется в budget-формуле, см. ниже) +
        # НАКОПЛЕННЫЙ расход банка (_bank_spent) — сумма wall-time, реально
        # потраченного ВНУТРИ НАШИХ СОБСТВЕННЫХ вызовов .agent() (как в
        # реальной игре: observation.remainingOverageTime — это МОЙ бак,
        # уменьшается только МОИМ overage, а не временем размышлений оппонента
        # между ходами). Поэтому budget_this считается из _bank_spent
        # (аккумулятор), а НЕ из time.time()-t_start (это включало бы и время
        # оппонента в последовательном раннере/на движке). _decisions_made —
        # счётчик вызовов .agent() после deck-select (включая дешёвые) —
        # используется в _compute_budget() для оценки est_remaining_decisions.
        self._t_start: float | None = None
        self._bank_spent: float = 0.0
        self._decisions_made: int = 0
        self._last_budget: float = 0.0
        # Счётчики для отчётности (S4 H2H): сколько решений реально ушло в поиск
        # vs дешёвый путь (gating), и максимальный/итоговый расход банка партии.
        self._n_search_decisions: int = 0
        self._n_cheap_decisions: int = 0

        # Накопление revealed card ID оппонента ЗА ПАРТИЮ + лок на архетип
        # (только используется в "auto"/"meta"; PIMCAgent-инстанс живёт всю партию,
        # так что это состояние переживает несколько вызовов .agent()).
        self._revealed_opp_ids: set[int] = set()
        self._locked_archetype: str | None = None

        # Mutable references, которые патченный _build_search_inputs читает НА КАЖДЫЙ
        # МИР детерминизации (см. _make_patched_build_search_inputs):
        #   _opp_deck_ref[0]    — фиксированный дек оппонента (mirror/crustle/locked-auto).
        #   _opp_sampler_ref[0] — Callable[[], list[int]] | None; если задан, вызывается
        #                         ЗАНОВО на каждый мир (мета-микс, пока архетип не залочен).
        if opponent_model == "mirror":
            self._opp_deck_ref: list = [list(self.deck)]
            self._opp_sampler_ref: list = [None]
        elif opponent_model == "crustle":
            self._opp_deck_ref = [list(CRUSTLE_DECK_APPROX)]
            self._opp_sampler_ref = [None]
        else:  # "auto" / "meta"
            self._opp_deck_ref = [list(CRUSTLE_DECK_APPROX)]  # плейсхолдер, перезаписывается сэмплером/локом
            self._opp_sampler_ref = [sample_meta_deck]  # не залочены -> мета-микс per-world

        # Lazy-инициализация (только Linux при первом .agent вызове)
        self._pimc_mod = None
        self._initialized = False

    def _lazy_init(self) -> None:
        """Инициализирует PIMC модуль лениво (требует Linux + libcg.so)."""
        if self._initialized:
            return

        import sys
        import importlib

        src_dir = os.path.join(_REPO, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        # Импортируем лучший доступный PIMC модуль
        # Порядок предпочтений: pimc_guided > pimc_hybrid > pimc_v2 > pimc
        for mod_name in ("pilot.pimc_guided", "pilot.pimc_hybrid", "pilot.pimc_v2", "pilot.pimc"):
            try:
                self._pimc_mod = importlib.import_module(mod_name)
                break
            except ImportError:
                continue

        if self._pimc_mod is None:
            raise RuntimeError(
                "Не удалось импортировать ни один PIMC модуль из src/pilot/. "
                "Убедитесь что src/ добавлен в PYTHONPATH."
            )

        # --- Применяем бюджет-параметры ---
        self._pimc_mod.TIME_BUDGET_SEC = self.budget
        self._pimc_mod.MAX_ROLLOUT_DEPTH = self.rollout_depth
        self._pimc_mod.N_WORLDS_MIN = self.n_worlds_min
        self._pimc_mod.MAX_CANDIDATES = self.max_candidates

        # --- Подключаем prior_agent как rollout policy (WP3 ↔ WP4) ---
        # prior_agent: Callable[[dict], list[int]] — заменяет v012 эвристику
        if self.prior_agent is not None and hasattr(self._pimc_mod, "_v012_decide"):
            prior_wrapper = _make_prior_decide_wrapper(self.prior_agent)
            # Патчим оба интерфейса: _decide (Observation) и agent (dict)
            self._pimc_mod._v012_decide = prior_wrapper
            self._pimc_mod._v012_agent = self.prior_agent

        # --- Патчим _build_search_inputs для opponent modeling ---
        # Используем _opp_deck_ref/_opp_sampler_ref (mutable) чтобы "auto"/"meta" режим
        # мог менять оппонента per-call (и per-world внутри одного вызова) без повторного
        # monkey-patching. Для "mirror" opp_deck_ref[0] = наш дек (mirror по умолчанию).
        # Для "crustle" и "auto"/"meta" — уже выставлено в __init__.
        if hasattr(self._pimc_mod, "_build_search_inputs"):
            patched = _make_patched_build_search_inputs(
                pimc_mod=self._pimc_mod,
                our_deck=self.deck,
                opp_deck_ref=self._opp_deck_ref,
                opp_sampler_ref=self._opp_sampler_ref,
            )
            self._pimc_mod._build_search_inputs = patched

        self._initialized = True

    # -- S3 bank-manager -----------------------------------------------------

    def _is_search_worthy(self, sel: dict) -> bool:
        """Decision-gating (S3): True только для осмысленных MAIN-развилок.

        Дешёвая проверка на raw obs_dict['select'] (никакого _lazy_init) —
        экономит банк, вызывая поиск ТОЛЬКО там, где есть реальный выбор.
        False для: <2 опций (форсировано), YES_NO/COUNT (тривиальные по типу),
        любой context != MAIN (setup/discard/to_active/guard и т.п.), а также
        min==max==0 (нечего выбирать).
        """
        opts = sel.get("option") or []
        if len(opts) < 2:
            return False
        if sel.get("type") in (_SELECT_TYPE_COUNT, _SELECT_TYPE_YES_NO):
            return False
        if sel.get("context") != _SELECT_CONTEXT_MAIN:
            return False
        min_c = int(sel.get("minCount") or 1)
        max_c = int(sel.get("maxCount") or 1)
        if min_c == 0 and max_c == 0:
            return False
        return True

    def _compute_budget(self) -> float:
        """Адаптивный бюджет на ТЕКУЩЕЕ решение из оставшегося банка (S3).

        budget_this = (bank_total*safety - spent) / est_remaining_decisions,
        зажатый в [min_decision_budget, max_decision_budget]. Возвращает 0.0
        если резерв банка уже исчерпан (SAFETY-часть) — сигнал "не искать
        вообще на этом ходу", см. agent() (bank-safety valve).

        spent = self._bank_spent — НАКОПЛЕННОЕ wall-time, потраченное
        ВНУТРИ НАШИХ СОБСТВЕННЫХ вызовов .agent() за партию (см. комментарий
        у self._bank_spent в __init__). Это осознанно НЕ time.time()-t_start:
        в реальном матче наш time-bank (observation.remainingOverageTime)
        уменьшается только НАШИМ overage, а не временем размышлений оппонента
        между нашими ходами — а в последовательном раннере (наш H2H-скрипт,
        и, судя по всему, сам движок) настенные часы тикают и во время хода
        соперника. Считаем по НАСТЕННОМУ времени НАШИХ решений (не по числу
        миров/rollout'ов) — поэтому в Docker (движок ~2.5x медленнее native)
        переоценка бюджета не нужна: та же секундная сетка, просто меньше
        миров/rollout'ов влезет в тот же budget_this.
        """
        reserve = self.bank_total_sec * self.bank_safety - self._bank_spent
        if reserve <= 0.0:
            return 0.0
        remaining_decisions = max(
            self.est_total_decisions - self._decisions_made, MIN_REMAINING_DECISIONS_FLOOR
        )
        budget = reserve / remaining_decisions
        return max(self.min_decision_budget, min(self.max_decision_budget, budget))

    def _sync_opponent_model_refs(self) -> None:
        """Синхронизирует _opp_deck_ref/_opp_sampler_ref (mutable refs, которые
        patched _build_search_inputs читает на каждый мир детерминизации) с
        текущим состоянием лока на архетип. Вызывается из agent() ПЕРЕД каждым
        поисковым решением (после _lazy_init, до self._pimc_mod.agent(...)).

        Вынесено в отдельный overridable метод (было инлайн-блоком в agent())
        специально ради PIMCAgentV2 (H3-1): базовая версия каждый раз
        перезаписывает _opp_sampler_ref[0] обратно на sample_meta_deck, что
        уничтожило бы archetype-tracking sampler V2 на СЛЕДУЮЩЕМ же вызове
        agent() — V2 переопределяет этот метод, сохраняя свой sampler.
        Поведение БАЗОВОГО класса (PIMCAgent) не изменилось.
        """
        if self.opponent_model in ("auto", "meta"):
            if self._locked_archetype is not None:
                self._opp_deck_ref[0] = _deck_for_archetype(self._locked_archetype)
                self._opp_sampler_ref[0] = None
            else:
                self._opp_sampler_ref[0] = sample_meta_deck

    def _cheap_decision(
        self, obs_dict: dict, sel: dict, n_options: int, min_count: int, max_count: int
    ) -> list[int]:
        """Дешёвый путь (S3): prior_agent, иначе локальный тривиальный выбор.

        НЕ трогает _lazy_init/pimc_mod вообще — используется и decision-gating'ом
        (форсированные/тривиальные контексты) и bank-safety valve'ом (резерв
        банка исчерпан -> перестаём искать, но доигрываем партию на prior).
        """
        self._n_cheap_decisions += 1
        if self.prior_agent is not None:
            try:
                result = self.prior_agent(obs_dict)
                if (result is not None
                        and all(0 <= i < n_options for i in result)
                        and len(result) >= min_count
                        and len(set(result)) == len(result)):
                    return result
            except Exception:
                pass
        return _local_trivial_choice(sel, n_options, min_count, max_count)

    @property
    def bank_stats(self) -> dict:
        """Отчётность bank-manager'а для одной партии (S3/S4 H2H).

        spent_sec:          НАШ реальный расход банка (self._bank_spent) —
                             сумма wall-time внутри НАШИХ .agent() вызовов
                             (это то, что решает budget_this и что нужно
                             сверять с 600s bank — критично < 300s при
                             SAFETY=0.5, и уж тем более < 600s).
        wall_clock_sec:      для сравнения — настенное время с начала партии
                             (включает и время оппонента между нашими ходами;
                             в реальном матче это НЕ наш бак, но полезно для
                             диагностики "партия сильно затянулась").
        decisions_made:      число вызовов .agent() после deck-select.
        n_search_decisions:  сколько ушло в поиск (search_worthy И budget>0).
        n_cheap_decisions:   сколько ушло дешёвым путём (gating ИЛИ резерв=0).
        last_budget:         budget_this последнего вычисленного решения.
        """
        wall_clock = (time.time() - self._t_start) if self._t_start is not None else 0.0
        return {
            "spent_sec": self._bank_spent,
            "wall_clock_sec": wall_clock,
            "decisions_made": self._decisions_made,
            "n_search_decisions": self._n_search_decisions,
            "n_cheap_decisions": self._n_cheap_decisions,
            "last_budget": self._last_budget,
        }

    def agent(self, obs_dict: dict) -> list[int]:
        """Публичный entry point — совместим с Agent.policy контрактом.

        Первый вызов (obs_dict["select"] is None) возвращает self.deck и
        сбрасывает bank-manager (новая партия). Последующие вызовы:
          1. Decision-gating (_is_search_worthy) — форсированные/1-опционные/
             YES_NO/COUNT/не-MAIN контексты идут дешёвым путём (prior/trivial),
             ни разу не трогая _lazy_init/pimc_mod.
          2. Для осмысленных MAIN-развилок — адаптивный budget_this
             (_compute_budget); если резерв банка уже исчерпан (budget==0),
             тоже уходим дешёвым путём (bank-safety valve).
          3. Иначе — поиск с budget_this секунд.

        Никогда не выбрасывает исключение (never-throw wrapper).
        """
        # NOTE: в реальном Kaggle-раннере первый вызов агента всегда идёт с
        # select=None (см. CLAUDE.md контракт) -> ветка ниже ловит начало
        # партии надёжно. НО локальные движковые раннеры (cg_native.game.
        # battle_start(deck0, deck1) и cg.game аналогично) передают деки
        # НАПРЯМУЮ аргументами и НЕ вызывают agent() с select=None вообще —
        # тогда этот фолбэк ловит "начало партии" на первом реальном решении
        # (иначе self._t_start остался бы None всю партию -> wall_clock_sec
        # диагностика была бы всегда 0; _bank_spent это НЕ затрагивает, он не
        # зависит от _t_start).
        if self._t_start is None:
            self._t_start = time.time()

        # Deck selection (первый вызов партии) — сбрасываем bank-manager.
        sel = obs_dict.get("select") if obs_dict else None
        if sel is None:
            self._t_start = time.time()
            self._bank_spent = 0.0
            self._decisions_made = 0
            self._n_search_decisions = 0
            self._n_cheap_decisions = 0
            self._last_budget = 0.0
            return list(self.deck)

        # call_start: измеряем СВОЙ реальный расход времени на это решение
        # (все return ниже проходят через call_start -> self._bank_spent),
        # см. комментарий у self._bank_spent в __init__/_compute_budget.
        call_start = time.time()

        n_options = len(sel.get("option") or [])
        min_count = int(sel.get("minCount") or 1)
        max_count = int(sel.get("maxCount") or 1)

        if n_options == 0:
            self._bank_spent += time.time() - call_start
            return []

        self._decisions_made += 1

        # Накопление revealed card ID оппонента — дёшево (только set-операции),
        # делаем на КАЖДОМ вызове (не только search-worthy), не требует
        # _lazy_init: сигнатура архетипа может проявиться и вне MAIN-развилок
        # (напр. discard/reveal эффекты).
        if self.opponent_model in ("auto", "meta"):
            self._revealed_opp_ids |= _extract_revealed_ids(obs_dict)
            if self._locked_archetype is None:
                arch = identify_archetype(self._revealed_opp_ids)
                if arch is not None:
                    self._locked_archetype = arch

        search_worthy = self._is_search_worthy(sel)
        budget_this = self._compute_budget() if search_worthy else 0.0
        self._last_budget = budget_this

        if not search_worthy or budget_this <= 0.0:
            result = self._cheap_decision(obs_dict, sel, n_options, min_count, max_count)
            self._bank_spent += time.time() - call_start
            return result

        try:
            self._lazy_init()

            # Синхронизируем opponent-model refs, которые patched
            # _build_search_inputs читает на каждый мир детерминизации.
            self._sync_opponent_model_refs()

            # Адаптивный per-decision бюджет (S3) вместо статичного self.budget.
            self._pimc_mod.TIME_BUDGET_SEC = budget_this

            result = self._pimc_mod.agent(obs_dict)
            if (result
                    and all(0 <= i < n_options for i in result)
                    and len(result) >= min_count
                    and len(set(result)) == len(result)):
                self._n_search_decisions += 1
                self._bank_spent += time.time() - call_start
                return result
        except NotImplementedError:
            raise
        except Exception:
            pass  # fallback ниже

        result = self._cheap_decision(obs_dict, sel, n_options, min_count, max_count)
        self._bank_spent += time.time() - call_start
        return result

    @property
    def stats(self) -> dict:
        """Статистика из PIMC модуля (если инициализирован).

        Включает: rollouts_total, decisions_pimc, decisions_heuristic,
        decisions_trivial, time_total, avg_worlds_per_decision, min/max_worlds.
        """
        if self._pimc_mod is None or not hasattr(self._pimc_mod, "get_stats"):
            return {}
        return self._pimc_mod.get_stats()


# ---------------------------------------------------------------------------
# PIMCAgentV2 — H3: раздельные rollout-политики + первый ход (PLAN.md §H3).
# ---------------------------------------------------------------------------

class PIMCAgentV2(PIMCAgent):
    """PIMC v2 — фикс root cause S4b (PLAN.md §H3-1..H3-3).

    Наследует ВСЁ от PIMCAgent (bank-manager S3, decision-gating,
    детерминизатор с META_PRIORS, archetype lock-on из revealed cards) и
    добавляет:

      H3-1 раздельные rollout-политики:
        - СВОЯ сторона: Crustle-prior (prior_agent) с ε-greedy (ε=epsilon,
          дефолт 0.3 — Scopone-литература ε∈[0.3,0.7]).
        - ОППОНЕНТ: дешёвый скриптовый пилот по архетипу ТЕКУЩЕГО МИРА
          детерминизации (не detected-архетип с прошлого хода — архетип
          ИМЕННО ЭТОГО rollout'а): alakazam->v017, grimmsnarl->v018,
          crustle/archaludon/generic->v009a (все три подставляют
          CRUSTLE_DECK_APPROX как деку — см. _ARCHETYPE_TO_ROLLOUT_POLICY),
          cubchoo/iono/неопознан->generic greedy-attack.
        - Любое исключение внутри любой rollout-политики -> первый легальный
          ход + счётчик в self.rollout_degradation.

      H3-2 (уже было структурно в pimc_guided._evaluate_candidates —
        _build_search_inputs + search_begin + rollout на КАЖДУЮ итерацию
        while-цикла = редетерминизация на каждый rollout): не тронуто,
        PIMCAgentV2 только заменяет ЧТО rollout делает на каждом шаге, не
        КОГДА определяется мир.

      H3-3: SelectContext.IS_FIRST (=41, "хотите ходить первым?") -> всегда
        YES, в обход decision-gating/поиска/prior (см. agent()).

    Дополнительные параметры (сверх PIMCAgent.__init__):
        epsilon:   вероятность случайного легального хода на СВОЕЙ стороне
                   rollout вместо Crustle-prior (дефолт 0.3).
        rng_seed:  seed для внутреннего random.Random (ε-greedy + сэмплинг
                   архетипа мира с трекингом). None -> несидированный.

    Атрибуты (сверх PIMCAgent):
        rollout_degradation: dict[str, int] — счётчики деградации
            rollout-политик за ТЕКУЩУЮ партию (ключи: "own_exc",
            "opp_exc_<policy_key>", "opp_load_fail_<policy_key>"). Сбрасывается
            неявно — экземпляр PIMCAgentV2 живёт одну партию (как и базовый
            класс), поэтому счётчик и так "за партию" без явного reset.
    """

    def __init__(
        self,
        *args,
        epsilon: float = 0.3,
        rng_seed: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self._rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

        # H3-1: архетип мира, который patched _build_search_inputs РЕАЛЬНО
        # подставил в ТЕКУЩИЙ мир детерминизации (читается _do_guided_rollout
        # ОДИН раз в начале rollout — архетип мира не меняется в течение
        # одного rollout, только между мирами). mirror/crustle -> фиксирован
        # на всю партию; auto/meta -> обновляется _tracking_sampler на
        # каждый НЕзалоченный мир, либо фиксируется на self._locked_archetype
        # после лока (см. _sync_opponent_model_refs override ниже).
        if self.opponent_model == "mirror":
            # Оппонент играет НАШУ же деку -> его архетип = архетип нашей деки.
            self._opp_archetype_ref: list = [identify_archetype(set(self.deck)) or "generic"]
        elif self.opponent_model == "crustle":
            self._opp_archetype_ref = ["crustle"]
        else:  # "auto" / "meta"
            self._opp_archetype_ref = [None]
            # Base __init__ уже создал self._opp_sampler_ref=[sample_meta_deck]
            # для auto/meta — заменяем ЭЛЕМЕНТ (не сам mutable-list, чтобы
            # patched _build_search_inputs, который уже держит ссылку на этот
            # список, увидел новую функцию) на трекающую версию, которая
            # запоминает сэмплированный архетип в self._opp_archetype_ref[0].
            self._opp_sampler_ref[0] = self._tracking_sampler

        # H3-1: кэш загруженных скриптовых пилотов-оппонентов
        # (policy_key -> (choice_fn, module|None)) — ленивый, живёт ОДНУ
        # партию (свежий экземпляр PIMCAgentV2 = свежий кэш = свежие модули,
        # никакой утечки состояния МЕЖДУ играми; см. _reset_stateful_globals
        # для утечки МЕЖДУ мирами внутри одной игры).
        self._opp_policy_cache: dict = {}
        self.rollout_degradation: dict = {}
        self._v2_rollout_patched: bool = False

    def _tracking_sampler(self) -> list[int]:
        """Как sample_meta_deck(), но ПОБОЧНЫМ ЭФФЕКТОМ записывает имя
        сэмплированного архетипа в self._opp_archetype_ref[0] — это то, что
        позволяет _do_guided_rollout узнать, КАКОЙ архетип реально играется
        в мире, который только что подставил _build_search_inputs (вызывается
        оттуда как `sampler()`, без аргументов — сигнатура должна совпадать).
        """
        name = sample_meta_archetype(self._rng)
        self._opp_archetype_ref[0] = name
        return _deck_for_archetype(name)

    def _sync_opponent_model_refs(self) -> None:
        """Override PIMCAgent._sync_opponent_model_refs (H3-1): сохраняет
        _tracking_sampler вместо того, чтобы каждый вызов agent() сбрасывать
        _opp_sampler_ref[0] обратно на "голый" sample_meta_deck (который не
        пишет в _opp_archetype_ref). При локе на архетип — фиксирует и
        _opp_archetype_ref[0] на locked-архетип (иначе он остался бы на
        последнем сэмплированном значении до лока)."""
        if self.opponent_model in ("auto", "meta"):
            if self._locked_archetype is not None:
                self._opp_deck_ref[0] = _deck_for_archetype(self._locked_archetype)
                self._opp_sampler_ref[0] = None
                self._opp_archetype_ref[0] = self._locked_archetype
            else:
                self._opp_sampler_ref[0] = self._tracking_sampler

    def _make_own_choice_fn(self) -> Callable:
        """H3-1: ε-greedy обёртка над Crustle-prior для СВОЕЙ стороны
        rollout'а. base_prior = pimc_mod._v012_decide, уже monkey-patched на
        self.prior_agent базовым PIMCAgent._lazy_init (если prior_agent
        задан в конструкторе) — переиспользуем как есть, ничего не дублируем.
        """
        base_prior = getattr(self._pimc_mod, "_v012_decide", None)
        eps = self.epsilon
        rng = self._rng

        def _own_choice(obs):
            if eps > 0.0 and rng.random() < eps:
                return _obs_random_legal_choice(obs, rng)
            if base_prior is not None:
                result = base_prior(obs)
                if result is not None:
                    return result
            return _obs_first_legal_choice(obs)

        return _own_choice

    def _lazy_init(self) -> None:
        """Расширяет PIMCAgent._lazy_init: после базовой инициализации
        (импорт pimc_guided, budget/candidates, _v012_decide patch,
        _build_search_inputs patch) ДОПОЛНИТЕЛЬНО патчит
        pimc_mod._do_guided_rollout на dual-policy версию (H3-1). Патчится
        РОВНО ОДИН РАЗ на экземпляр (self._v2_rollout_patched) — повторные
        вызовы _lazy_init (base уже идемпотентен через self._initialized)
        не должны каждый раз пересобирать замыкание own_choice_fn.
        """
        super()._lazy_init()
        if self._v2_rollout_patched:
            return

        own_choice_fn = self._make_own_choice_fn()
        dual_rollout = _make_dual_policy_rollout(
            pimc_mod=self._pimc_mod,
            own_choice_fn=own_choice_fn,
            opp_policy_cache=self._opp_policy_cache,
            opp_archetype_ref=self._opp_archetype_ref,
            degrade_counts=self.rollout_degradation,
        )
        self._pimc_mod._do_guided_rollout = dual_rollout
        self._v2_rollout_patched = True

    def agent(self, obs_dict: dict) -> list[int]:
        """H3-3: перехватывает SelectContext.IS_FIRST ДО decision-gating/
        bank/поиска — гарантированно отвечает YES (пойти первым) независимо
        от того, что решил бы cheap-путь/prior/поиск на этом контексте.
        Всё остальное делегируется PIMCAgent.agent() без изменений."""
        if self._t_start is None:
            self._t_start = time.time()

        sel = obs_dict.get("select") if obs_dict else None
        if sel is not None:
            first_move = _choose_first_move_if_applicable(sel)
            if first_move is not None:
                call_start = time.time()
                self._decisions_made += 1
                self._n_cheap_decisions += 1
                self._bank_spent += time.time() - call_start
                self._last_budget = 0.0
                return first_move

        return super().agent(obs_dict)


# ---------------------------------------------------------------------------
# Удобный конструктор с предустановленными профилями
# ---------------------------------------------------------------------------

def make_pimc_agent(
    profile: str = "balanced",
    opponent_model: str = "auto",
    deck: list[int] | None = None,
    prior_agent: Callable[[dict], list[int]] | None = None,
) -> PIMCAgent:
    """Фабрика PIMCAgent с предустановленными профилями.

    Args:
        profile:
          "fast"     — budget=1.0s, depth=100, worlds_min=2 (для smoke-тестов)
          "balanced" — budget=3.0s, depth=150, worlds_min=3 (текущий v014 equivalent)
          "deep"     — budget=5.0s, depth=200, worlds_min=5 (для ценных решений)
        opponent_model:
          "mirror" | "crustle" | "auto" | "meta" (default: "auto" — меtagame-aware)
        deck:
          Дек кандидата. Если None — IONO_DECK.
        prior_agent:
          BC-prior агент для rollout (из WP4). Если None — эвристика v012.

    Returns:
        Готовый PIMCAgent.

    Примеры:
        pimc = make_pimc_agent("fast")                     # smoke-тест
        pimc = make_pimc_agent("balanced", "crustle")      # против Crustle-поля
        pimc = make_pimc_agent("deep", prior_agent=pilot)  # с BC-prior из WP4
    """
    profiles = {
        "fast":     dict(budget=1.0, rollout_depth=100, n_worlds_min=2, max_candidates=6),
        "balanced": dict(budget=3.0, rollout_depth=150, n_worlds_min=3, max_candidates=8),
        "deep":     dict(budget=5.0, rollout_depth=200, n_worlds_min=5, max_candidates=8),
    }
    if profile not in profiles:
        raise ValueError(f"Unknown profile: {profile!r}. Choose from: {list(profiles)}")
    return PIMCAgent(
        **profiles[profile],
        opponent_model=opponent_model,
        deck=deck,
        prior_agent=prior_agent,
    )


def make_pimc_agent_v2(
    profile: str = "balanced",
    opponent_model: str = "auto",
    deck: list[int] | None = None,
    prior_agent: Callable[[dict], list[int]] | None = None,
    epsilon: float = 0.3,
    rng_seed: int | None = None,
) -> "PIMCAgentV2":
    """Фабрика PIMCAgentV2 (H3) — та же профильная система, что
    make_pimc_agent, но конструирует PIMCAgentV2 (раздельные rollout-политики
    + первый ход, см. класс PIMCAgentV2 docstring). deck по умолчанию —
    CRUSTLE_DECK_APPROX (H3: наша сторона всегда Crustle), НЕ IONO_DECK как
    в make_pimc_agent (Iono мёртв, см. PLAN.md «Отвергнутые пути»).

    Примеры:
        pimc = make_pimc_agent_v2("fast")   # smoke
        pimc = make_pimc_agent_v2("balanced", prior_agent=v009a_agent)
    """
    profiles = {
        "fast":     dict(budget=1.0, rollout_depth=100, n_worlds_min=2, max_candidates=6),
        "balanced": dict(budget=3.0, rollout_depth=150, n_worlds_min=3, max_candidates=8),
        "deep":     dict(budget=5.0, rollout_depth=200, n_worlds_min=5, max_candidates=8),
    }
    if profile not in profiles:
        raise ValueError(f"Unknown profile: {profile!r}. Choose from: {list(profiles)}")
    return PIMCAgentV2(
        **profiles[profile],
        opponent_model=opponent_model,
        deck=deck if deck is not None else list(CRUSTLE_DECK_APPROX),
        prior_agent=prior_agent,
        epsilon=epsilon,
        rng_seed=rng_seed,
    )
