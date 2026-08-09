#!/usr/bin/env python3
"""featurize_corpus_2b.py — превращает out/corpus_2b/decisions.jsonl (одна строка = одно
РЕШЕНИЕ эксперта) в таблицу "одна строка = одна ОПЦИЯ одного решения" для обучения ранкера
по схеме tetsutani (main.py:103-114 у стороннего сабмишена — модель скорит каждую опцию,
ответ = top-k по скору; см. MEMORY reference_kaggle_recon_20260808_code_discussion).

ПОЧЕМУ построчно по опциям, а не по решениям: BC-классификатор "какой индекс выбрать" не
переносится между решениями с разным n_options (индекс 3 в одном решении и индекс 3 в другом
— РАЗНЫЕ вещи). Ранкер решает это, скоря КАЖДУЮ опцию как отдельный пример с одинаковым
набором фич независимо от n_options; y=1 у опций, попавших в action эксперта, y=0 у остальных.

ВХОД:
  out/corpus_2b/decisions.jsonl   — per-decision метаданные + action эксперта
  out/corpus_2b/obs_shards/<episode_id>.jsonl — obs_dict; obs_ref = номер строки (0-based)
  out/corpus_2b/split/split.json  — episode_id -> train/test (сплит по ЭПИЗОДАМ, не решениям)

ВЫХОД (out/corpus_2b/features/):
  <group>.parquet     — по одному файлу на группу роутинга (main/c7/low/mid/easy/rest)
  feature_spec.json   — УПОРЯДОЧЕННЫЙ список имён X-колонок. Это часть контракта: тот же
                         порядок нужно воспроизвести байт-в-байт при live-инференсе (T1.3),
                         иначе индексы весов модели разъедутся с фичами на гейте.
  stats.json           — сводка (строк/группа, доли train/test, доля y=1, ...)

СХЕМА СТРОКИ:
  decision_uid (str "episode_id:agent_index:obs_ref"), option_idx (int, позиция опции в
  select.option), y (0/1), group (str), split ('train'/'test'), затем X-фичи (см.
  feature_spec.json), затем МЕТА-колонки НЕ входящие в X (см. META_COLUMNS ниже) —
  episode_id/agent_index/turn_idx/step_idx/expert/opp_archetype/opp_has_crustle/
  subj_archetype/deck_key/game_won/sel_context_name/n_options/action_len — оставлены для
  аудита/страт-анализа, но модель их видеть не должна (это и есть проверка #4 ниже).

СЕНТИНЕЛ -1: вместо NaN/inf используется явный -1 для "значения не существует по смыслу"
(нет своего/чужого активного покемона -> его hp/maxHp/damage/hp_frac/id = -1; опция не
атака -> opt_attack_id/opt_attack_damage/opt_damage_minus_opp_hp = -1; карта не
резолвится -> opt_card_id = -1 и т.д.). ИСКЛЮЧЕНИЕ: чистые СЧЁТчики (bench_n, discard_n,
active_n_energies, active_n_tools, bench_energy_sum, n_lines_*) используют натуральный 0
при отсутствии предмета счёта — там 0 не сентинел, а корректное значение ("0 энергий на
несуществующем моне" неотличимо по смыслу от "0 энергий на реальном моне без энергий").
ЛОВУШКА: opt_damage_minus_opp_hp у реальной атаки теоретически может случайно равняться -1
(урон на 1 меньше HP цели) — тогда сентинел и настоящее значение неразличимы; чтобы модель
могла разрулить эту коллизию, рядом всегда лежит opt_is_attack (1/0), не зависящий от знака.

ОДНОГОРЯЧЕЕ КОДИРОВАНИЕ card_id (по заданию): card_id-поля, которые прямо называют КАКОГО
покемона нашей 19-карточной деки Grimmsnarl-toolbox видно на столе или разыгрывается опцией
— active_id (свой актив), opp_active_id (чужой актив), opt_card_id (резолвленная карта
опции) — кодируются и как сырой int id, и как one-hot по 19 картам + флаг *_not_in_deck (1,
если id есть, но не входит в 19; 0 и при отсутствии карты, и при принадлежности деке).
19-card список — карты деки Luca/szlachetny snieg (out/corpus_2b/ref_decks_2b.json),
ОСОЗНАННЫЙ ВЫБОР: у Dominic Peel/__Taichicchi__ (peel_handfan) вместо {Munkidori=112,
Rock Inn=1137} стоит {1161}; 1161 попадёт в *_not_in_deck=1, а не в отдельный 20-й one-hot
— иначе список card_id не соответствовал бы формулировке задания ("19 карт"). stadium_id и
context_card_id one-hot НЕ получают (только сырой id) — это метаданные контекста, а не
"чья это карта на столе"; раздувать 19 доп. колонок ради редкого stadium_id избыточно.

ЛОВУШКА резолва опции: players[i] не содержит ключа 'deck' (только deckCount), а prize —
список из null-плейсхолдеров, пока приз не взят — резолв в зону deck(1) ИЛИ prize(6)
специально возвращает None (обе зоны рубашкой вверх для ОБЕИХ сторон по правилам TCG),
это НЕ баг. На 07-20-корпусе (fidelity_expert_topk.py) это уже описано как "TO_HAND слеп".

ЛОВУШКА игрока-перспективы: obs_ref уникально указывает на РОВНО ОДНУ строку шарда;
current.yourIndex в этой строке ВСЕГДА равен decision['agent_index'] (проверено на 18
эпизодах-зеркалках, где оба игрока — отслеживаемые эксперты, obs_ref-диапазоны двух агентов
не пересекаются). "me" в резолве берётся из obs, не из decision, но они тождественны.

Порт логики резолва опции -> card_id — ПОРТ (не копия) _resolve_ref/_resolve_hand_card_id
из scripts/fidelity_expert_topk.py, упрощённый до одного возвращаемого card_id на опцию
(там же порт ZONE-кодов area: 1=deck,2=hand,3=discard,4=active,5=bench,6=prize).

Запуск:
    .venv/bin/python scripts/featurize_corpus_2b.py --limit 50      # смоук, 50 эпизодов
    .venv/bin/python scripts/featurize_corpus_2b.py                 # полный корпус
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

WORKTREE = Path(__file__).resolve().parent.parent
SRC = WORKTREE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(WORKTREE / "scripts") not in sys.path:
    sys.path.insert(0, str(WORKTREE / "scripts"))

from prize_values_2b import prize_value  # noqa: E402

# ── Native engine shim (macOS arm64), только для таблицы урона атак ──────────
import cg_native  # noqa: E402
import cg_native.api  # noqa: E402

sys.modules.setdefault("cg", cg_native)
sys.modules.setdefault("cg.api", cg_native.api)
from cg.api import all_attack, all_card_data  # noqa: E402

SENTINEL = -1.0

# ── OptionType / area / zone константы (см. submissions/v076_final/main.py:365-423) ──
OPT_NUMBER, OPT_YES, OPT_NO, OPT_CARD = 0, 1, 2, 3
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY = 7, 8, 9, 10
OPT_RETREAT, OPT_ATTACK, OPT_END = 12, 13, 14
AREA_HAND, AREA_ACTIVE, AREA_BENCH = 2, 4, 5

# ── LogType константы, нужные только БЛОКУ 2 (см. cg.api.LogType) — играют роль "тип
# события" истории хода. Значения совпадают с числами из cg.api.LogType (10/11/12/15); в
# буфер истории они перекодируются в OPT_*-числа (см. _extract_turn_log_events) так, чтобы
# prevN_type жил в ТОЙ ЖЕ числовой шкале, что и opt_type у остальных фич.
LOG_PLAY, LOG_ATTACH, LOG_EVOLVE, LOG_ATTACK = 10, 11, 12, 15

# ── EnergyType константы, нужные только БЛОКУ 1 (энергия) — см. cg.api.EnergyType.
ENERGY_COLORLESS, ENERGY_DARKNESS = 0, 7
MUNKIDORI_ID = 112
BASIC_D_ENERGY_ID = 7
AREA_LOOKING = 12  # НЕ путать с OPT_RETREAT=12 (тот же int, другое пространство имён —
# area-код опции vs option.type). Найдено эмпирически на корпусе 2B: Pokégear 3.0
# ("Look at the top 7 cards of your deck...") даёт TO_HAND(7)-опции с area=12, индекс —
# позиция в obs['current']['looking'] (временный "показ" карт колоды/сброса ИГРОКУ,
# а не постоянная зона players[i][...]). В отличие от deck(1)/prize(6) эта зона
# РЕЗОЛВИРУЕТСЯ — карты в looking видны нам напрямую.
ZONE = {1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench", 6: "prize"}
CARD_RESOLVABLE_TYPES = {OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY, OPT_CARD}

# 19 карт деки Grimmsnarl-toolbox (Luca / szlachetny snieg — большинство корпуса),
# см. докстринг выше про peel_handfan-вариант.
DECK19 = sorted([7, 104, 112, 646, 647, 648, 860, 1079, 1080, 1086, 1097, 1122, 1137, 1152,
                  1182, 1219, 1227, 1231, 1259])
assert len(DECK19) == 19

# ── БЛОК 3 (T1.5, состав руки/бенча/сброса + видимый борд оппонента) — ДАННЫЕ, посчитаны/
# проверены разово по cg.api.all_card_data()/корпусу 2B, здесь вшиты литералом (та же
# конвенция, что MUNKIDORI_ID/BASIC_D_ENERGY_ID выше — маленькие таблицы не тянут card_db
# в live, см. докстринг _FEATURES_TEMPLATE в export_bc_2b.py про "ноль внешних зависимостей").
#
# Эволюционная линия (cg.api.CardData.evolvesFrom, проверено на живых данных движка):
# Impidimp(646) -> Morgrem(647) -> Grimmsnarl ex(648) (Marnie-линия, cardType=0/Pokemon,
# stage1/stage2 флаги); отдельно Snorunt(860) -> Froslass(104) (обычная линия, НЕ Marnie).
# EVOLUTION_NEXT: card_id стоящего на столе моно -> card_id карты, эволюционирующей ИЗ него.
EVOLUTION_NEXT = {646: 647, 647: 648, 860: 104}
STAGE_NAMES = {646: "impidimp", 647: "morgrem", 648: "grimmsnarl", 860: "snorunt", 104: "froslass"}

# cardType-классификация внутри DECK19 (cg.api.CardData.cardType: 0=Pokemon, 1=Item,
# 3=Supporter, 4=Stadium, 5=Energy) — нужна только для агрегатов "есть ли в руке
# supporter/item", проверено на живых данных движка (см. вызов в разработке задачи).
SUPPORTER_IDS = frozenset({1182, 1219, 1227, 1231})
ITEM_IDS = frozenset({1079, 1080, 1086, 1097, 1122, 1137, 1152})

# Топ-24 самых частых card_id на видимом борде (актив+бенч) ОППОНЕНТА по корпусу 2B —
# компактная кодировка вместо one-hot по всем ~76 встречающимся id. Посчитано разовым
# скриптом по out/corpus_2b/obs_shards (каждая 7-я строка каждого из 738 шардов,
# 10660 сэмплов) 2026-08-09; включает и карты нашей же 19-карточной деки (частые
# зеркальные матчи — mirror_grimmsnarl ~59-64%, см. MEMORY FIELD_RECON), и карты чужих
# архетипов (Alakazam-линия 743/741/742, Cynthia's Garchomp ex 381, Crustle 345,
# Fezandipiti ex 140, Lopunny ex 849 и т.д. — совпадает с известной метой). Список НЕ
# меняется при ретренах (фиксированная кодировка = часть контракта FEATURE_ORDER), только
# опп_board_other_cnt ловит всё, что вне списка.
OPP_BOARD_TOPN = [112, 648, 305, 646, 743, 741, 647, 742, 140, 381, 860, 345, 342, 401,
                  343, 400, 104, 341, 414, 431, 756, 344, 380, 379]
assert len(OPP_BOARD_TOPN) == 24 == len(set(OPP_BOARD_TOPN))

# ── Роутинг групп (порт схемы tetsutani main.py:103-114 — ДАННЫЕ, не код) ────────────────
def route_group(ctx) -> str:
    ctx = int(ctx) if ctx is not None else -1
    if ctx == 0:
        return "main"
    if ctx == 7:
        return "c7"
    if ctx in (3, 5, 8):
        return "low"
    if ctx in (13, 15, 16, 21, 22, 40, 43):
        return "mid"
    if ctx in (1, 2, 4, 27, 30, 34, 37, 38, 41):
        return "easy"
    return "rest"


def onehot_deck19(prefix: str, card_id) -> dict:
    """{prefix}_is_<cid> для каждой из 19 карт + {prefix}_not_in_deck. card_id может быть
    None/-1 (нет карты) -> всё 0 (не путать "нет карты" с "чужая карта")."""
    out = {f"{prefix}_is_{cid}": 0.0 for cid in DECK19}
    not_in_deck = 0.0
    if card_id is not None and card_id != -1:
        cid = int(card_id)
        if cid in DECK19:
            out[f"{prefix}_is_{cid}"] = 1.0
        else:
            not_in_deck = 1.0
    out[f"{prefix}_not_in_deck"] = not_in_deck
    return out


def _card_id_counts(cards) -> Counter:
    """★ БЛОК 3. Список карт-словарей (obs['...']['hand'|'bench'|'discard'], каждый
    элемент {'id':..., ...}) -> Counter{card_id: count}. Общая для руки/бенча/сброса/
    видимого борда оппонента — там везде одна и та же форма (список dict с ключом 'id'),
    включая bench-мон-словари (у них тоже есть 'id', не только у карт руки/сброса)."""
    c = Counter()
    for card in (cards or []):
        if isinstance(card, dict) and card.get("id") is not None:
            try:
                c[int(card["id"])] += 1
            except (TypeError, ValueError):
                pass
    return c


def hand_bench_discard_features(mine: dict, own_active_id) -> dict:
    """★ БЛОК 3 (T1.5) — СОСТАВ РУКИ/БЕНЧА/СБРОСА (своих). Возвращает X-фичи (own_hand_cnt_*/
    own_bench_cnt_*/own_discard_cnt_*/_other_cnt/эволюция) ПЛЮС две приватные (не-X, по
    конвенции "_me"/"_players" в decision_base_features) сырые Counter-таблицы —
    "_hand_counts"/"_discard_counts" — нужны дальше в option_features для (5) "опция в
    контексте руки" (opt_card_hand_remaining_n/opt_card_discard_n), чтобы не пересчитывать
    руку/сброс дважды на каждую опцию одного решения.
    ПОЧЕМУ full-19-table для сброса, а не только 5 "ключевых карт" из задания: те 5 (648
    Grimmsnarl ex/1182 Boss's Orders/112 Munkidori/104 Froslass/7 энергия) — просто ПОДМНОЖЕСТВО
    этой таблицы (own_discard_cnt_648 и т.д.), а полная таблица ещё и покрывает пункт (5) для
    ЛЮБОЙ карты опции, не только этих пяти — те же 19 счётчиков решают обе задачи разом."""
    hand_counts = _card_id_counts(mine.get("hand"))
    bench_counts = _card_id_counts(mine.get("bench"))
    discard_counts = _card_id_counts(mine.get("discard"))

    feats = {}
    hand_in_deck = 0
    for cid in DECK19:
        n = hand_counts.get(cid, 0)
        feats[f"own_hand_cnt_{cid}"] = float(n)
        hand_in_deck += n
    feats["own_hand_other_cnt"] = float(sum(hand_counts.values()) - hand_in_deck)

    bench_in_deck = 0
    for cid in DECK19:
        n = bench_counts.get(cid, 0)
        feats[f"own_bench_cnt_{cid}"] = float(n)
        bench_in_deck += n
    feats["own_bench_other_cnt"] = float(sum(bench_counts.values()) - bench_in_deck)

    discard_in_deck = 0
    for cid in DECK19:
        n = discard_counts.get(cid, 0)
        feats[f"own_discard_cnt_{cid}"] = float(n)
        discard_in_deck += n
    feats["own_discard_other_cnt"] = float(sum(discard_counts.values()) - discard_in_deck)

    # эволюционная линия: сколько своих монов НА СТОЛЕ (актив+бенч, НЕ рука) в каждой стадии.
    board_counts = Counter(bench_counts)
    if own_active_id is not None:
        board_counts[int(own_active_id)] += 1
    for cid, name in STAGE_NAMES.items():
        feats[f"own_board_n_{name}"] = float(board_counts.get(cid, 0))

    # готовность эволюции: скольким своим монам НА СТОЛЕ доступна следующая стадия В РУКЕ
    # (EVOLUTION_NEXT[cid] > 0 копий в hand_counts) — считаем по board_counts (актив+бенч).
    n_ready = 0
    for cid, cnt in board_counts.items():
        nxt = EVOLUTION_NEXT.get(cid)
        if nxt is not None and hand_counts.get(nxt, 0) > 0:
            n_ready += cnt
    feats["own_n_ready_to_evolve"] = float(n_ready)
    active_nxt = EVOLUTION_NEXT.get(int(own_active_id)) if own_active_id is not None else None
    feats["own_active_ready_to_evolve"] = (
        1.0 if (active_nxt is not None and hand_counts.get(active_nxt, 0) > 0) else 0.0)

    feats["own_hand_has_supporter"] = (
        1.0 if any(hand_counts.get(c, 0) > 0 for c in SUPPORTER_IDS) else 0.0)
    feats["own_hand_has_item"] = (
        1.0 if any(hand_counts.get(c, 0) > 0 for c in ITEM_IDS) else 0.0)

    feats["_hand_counts"] = hand_counts
    feats["_discard_counts"] = discard_counts
    return feats


def opp_board_features(theirs: dict) -> dict:
    """★ БЛОК 3 (T1.5) — ВИДИМЫЙ БОРД ОППОНЕНТА (актив+бенч, всегда видны обеим сторонам по
    правилам TCG — не fog-of-war, в отличие от чужой руки/деки/призов). Кодировка — счётчики
    по OPP_BOARD_TOPN (см. её докстринг про метод отбора) + один общий "прочее"-счётчик."""
    opp_list = list(theirs.get("active") or []) + list(theirs.get("bench") or [])
    counts = _card_id_counts(opp_list)
    known = 0
    feats = {}
    for cid in OPP_BOARD_TOPN:
        n = counts.get(cid, 0)
        feats[f"opp_board_cnt_{cid}"] = float(n)
        known += n
    feats["opp_board_other_cnt"] = float(sum(counts.values()) - known)
    return feats


def card_at(players: list, pi, area, idx):
    """card_id по (playerIndex, area-код, index-в-зоне) или None, если зона скрыта /
    индекс не резолвится (в т.ч. area=deck(1) — своя/чужая колода НЕ видна по правилам)."""
    if area not in ZONE or not isinstance(idx, int):
        return None
    if not isinstance(pi, int) or not (0 <= pi < len(players)):
        return None
    zone = players[pi].get(ZONE[area])
    if not isinstance(zone, list) or not (0 <= idx < len(zone)):
        return None
    c = zone[idx]
    if not isinstance(c, dict):
        return None
    cid = c.get("id")
    return int(cid) if cid is not None else None


def mon_at(players: list, pi, area, idx):
    """★ БЛОК 1 (энергия). Как card_at, но возвращает ВЕСЬ словарь покемона (нужны его
    energies/hp), а не только id. area здесь всегда active(4)/bench(5) — используется ТОЛЬКО
    для резолва цели ATTACH-опции (inPlayArea/inPlayIndex, всегда на СВОЕЙ стороне)."""
    if area not in ZONE or not isinstance(idx, int):
        return None
    if not isinstance(pi, int) or not (0 <= pi < len(players)):
        return None
    zone = players[pi].get(ZONE[area])
    if not isinstance(zone, list) or not (0 <= idx < len(zone)):
        return None
    c = zone[idx]
    return c if isinstance(c, dict) else None


def _find_own_mon(mine: dict, card_id: int):
    """★ БЛОК 1. Первый свой моно (active ИЛИ bench) с данным card_id -> (mon_dict, is_active),
    либо (None, False), если такого на борту нет. Нужен для Munkidori-guard-сигналов (у
    старого пилота был отдельный guard именно под "Munkidori с 0 энергии", см. main.py)."""
    active_mon = _mon(mine.get("active"))
    if active_mon is not None and int(active_mon.get("id") or 0) == card_id:
        return active_mon, True
    for m in (mine.get("bench") or []):
        if isinstance(m, dict) and int(m.get("id") or 0) == card_id:
            return m, False
    return None, False


def compute_attack_ready_cost(card_db: dict, attack_db: dict) -> dict:
    """★ БЛОК 1. card_id -> минимальное число энергий среди атак этой карты, полностью
    оплачиваемых ЕДИНСТВЕННЫМ типом энергии в этой деке (Basic {D} Darkness=7): каждый слот
    стоимости атаки должен быть COLORLESS(0, оплачивается любой энергией) или DARKNESS(7) —
    иначе атака физически некастуема в этой деке (0 энергии другого типа никогда не
    появится на столе). Карты без такой атаки (Munkidori/Froslass/Snorunt — их единственные
    атаки требуют Psychic/Water) ОТСУТСТВУЮТ в словаре (см. вызывающий код: .get(...) -> None
    -> SENTINEL). Проверено эмпирически на живых данных движка (cg_native.all_card_data/
    all_attack): Grimmsnarl ex(648)/Morgrem(647) Shadow Bullet/Corkscrew Punch = {D}{D} (2);
    Impidimp(646) имеет ДВЕ кастуемые атаки Filch(0 dmg, {C}) и Corkscrew Punch(10 dmg, {D}) —
    min=1; Munkidori(112)/Froslass(104)/Snorunt(860) — 0 кастуемых атак (Mind Bend {P}/
    Frost Smash {W}/Chilly {W}), отсутствуют в таблице."""
    out = {}
    for cid, cd in card_db.items():
        costs = []
        for aid in (getattr(cd, "attacks", None) or []):
            atk = attack_db.get(int(aid))
            if atk is None:
                continue
            energies = list(getattr(atk, "energies", None) or [])
            if all(int(e) in (ENERGY_COLORLESS, ENERGY_DARKNESS) for e in energies):
                costs.append(len(energies))
        if costs:
            out[int(cid)] = min(costs)
    return out


# ── БЛОК 4 (H1, 08-09 итерация 4: "модель не видит угрозу оппонента") ────────────────────
# ПРОБЛЕМА (из ручного разбора 196 промахов): признаков про атаки ОППОНЕНТА нет вообще —
# только own_active_min_attack_cost/own_active_attack_ready (БЛОК 1 выше). retreat
# предлагается в 59% MAIN-решений, выбирается экспертом в 2.9%, модель промахивается в 51.9%
# ВСЕГДА в сторону недооценки (никогда не переоценивает), медианный ранг правильного retreat
# = 3 — классический сигнал "не видит, что активу вот-вот прилетит".
# ПОЧЕМУ own_active_min_attack_cost НЕЛЬЗЯ переиспользовать для оппонента: та таблица
# (compute_attack_ready_cost) — упрощение "наша дека кастует только {D}", учитывает ТОЛЬКО
# число энергий, не тип (для нашей дека тип гарантирован). Чужой актив — ЛЮБАЯ карта с ЛЮБЫМ
# типом энергии; "может ли он атаковать прямо сейчас" ТРЕБУЕТ настоящей проверки типов
# ПРОТИВ реально прикреплённых энергий (obs[...]['active'][0]['energies'] — список int
# EnergyType, ВИДИМ ОБЕИМ сторонам, борд не fog-of-war) — иначе цветной cost задаёт ложное
# "не готов" (или ложное "готов", если бы просто сравнивали count).
CTX_DAMAGE_COUNTER, CTX_REMOVE_DAMAGE_COUNTER, CTX_REMOVE_DAMAGE_COUNTER_COUNT = 13, 16, 40
# ENERGY_COLORLESS(=0) уже объявлен в БЛОКЕ 1 выше (compute_attack_ready_cost) — переиспользуем.
ENERGY_RAINBOW, ENERGY_TEAM_ROCKET = 10, 11  # cg.api.EnergyType: RAINBOW платит за ЛЮБОЙ тип,
# TEAM_ROCKET — за PSYCHIC(5) ИЛИ DARKNESS(7) (энергии, специфичные для некоторых чужих
# архетипов меты, напр. tr_mewtwo — "TR" = Team Rocket). Проверено эмпирически на выборке
# ~15.5k видимых энергий корпуса: RAINBOW(10) НЕ встретился ни разу, TEAM_ROCKET(11) — 184
# раза (1.2%), остальные — обычные однотипные энергии.
_TEAM_ROCKET_PAYS = frozenset({5, 7})


def compute_card_attack_costs(card_db: dict, attack_db: dict) -> dict:
    """★ БЛОК 4 (H1). card_id -> tuple((cost_tuple, damage), ...) для ВСЕХ атак карты — в
    отличие от compute_attack_ready_cost (БЛОК 1, только D-only-кастуемые атаки НАШЕЙ деки),
    здесь ничего не фильтруется: нужна таблица для ЛЮБОГО чужого актива. cost_tuple — tuple
    int'ов EnergyType этой атаки в исходном порядке attack.energies (0=COLORLESS — любой тип
    годится, см. energies_can_pay). Карты без единой атаки (энергии/предметы/стадиумы/карты
    без данных об атаках) отсутствуют в словаре -> .get(...) -> None -> mon_threat трактует
    как SENTINEL, та же конвенция, что compute_attack_ready_cost."""
    out = {}
    for cid, cd in card_db.items():
        entries = []
        for aid in (getattr(cd, "attacks", None) or []):
            atk = attack_db.get(int(aid))
            if atk is None:
                continue
            cost = tuple(int(e) for e in (getattr(atk, "energies", None) or []))
            dmg = int(getattr(atk, "damage", 0) or 0)
            entries.append((cost, dmg))
        if entries:
            out[int(cid)] = tuple(entries)
    return out


def energies_can_pay(cost: tuple, avail: list) -> bool:
    """★ БЛОК 4 (H1). Может ли мультимножество прикреплённых энергий `avail` (list int
    EnergyType, порядок неважен) оплатить cost-лист ОДНОЙ атаки `cost` (tuple int EnergyType,
    0=COLORLESS — годится любой тип). Точный (не жадный/не аппроксимация) бэктрекинг-мэтчинг:
    цветные слоты матчатся 1:1 по типу либо через wildcard-энергии (RAINBOW — любой тип,
    TEAM_ROCKET — PSYCHIC/DARKNESS), COLORLESS-слоты проверяются последними — любой остаток
    энергий (не пошедший в цветные слоты) годится. Числа малы на практике корпуса (<=5
    слотов стоимости, <=8 прикреплённых энергий) — полный перебор дёшев (не более чем
    5!≈120 веток), не приближение."""
    colored = sorted(c for c in cost if c != ENERGY_COLORLESS)
    n_colorless = len(cost) - len(colored)
    if len(avail) < len(cost):
        return False
    avail = list(avail)
    used = [False] * len(avail)

    def can_supply(etype, energy) -> bool:
        if energy == etype:
            return True
        if energy == ENERGY_RAINBOW:
            return True
        if energy == ENERGY_TEAM_ROCKET:
            return etype in _TEAM_ROCKET_PAYS
        return False

    def backtrack(i: int) -> bool:
        if i == len(colored):
            return True
        etype = colored[i]
        for j, e in enumerate(avail):
            if not used[j] and can_supply(etype, e):
                used[j] = True
                if backtrack(i + 1):
                    return True
                used[j] = False
        return False

    if not backtrack(0):
        return False
    remaining = len(avail) - sum(used)
    return remaining >= n_colorless


def mon_threat(card_id, energies: list, card_attack_costs: dict) -> tuple:
    """★ БЛОК 4 (H1). -> (min_attack_cost, attack_ready, max_damage_ready) для мона `card_id`
    с прикреплёнными энергиями `energies` (list int EnergyType). min_attack_cost — минимум
    среди ВСЕХ атак карты (сколько энергии нужно самой дешёвой, безотносительно готовности
    прямо сейчас, SENTINEL если карты/атак нет). attack_ready/max_damage_ready — среди атак,
    ОПЛАЧИВАЕМЫХ ТЕКУЩИМИ energies (energies_can_pay): attack_ready=0.0 и
    max_damage_ready=0.0, если карта есть, но НИ ОДНА атака сейчас не оплачивается (натуральный
    0 — "0 урона он может нанести прямо сейчас", не сентинел, та же конвенция, что счётчики
    БЛОКА 3); card_id=None или карта без атак в таблице -> (SENTINEL, 0.0, SENTINEL) — та же
    тройка значений, что own_active_min_attack_cost/own_active_attack_ready используют для
    "нет активного мона" (см. decision_base_features)."""
    if card_id is None:
        return SENTINEL, 0.0, SENTINEL
    entries = card_attack_costs.get(int(card_id))
    if not entries:
        return SENTINEL, 0.0, SENTINEL
    min_cost = min(len(cost) for cost, _dmg in entries)
    ready = 0.0
    max_dmg = None
    for cost, dmg in entries:
        if energies_can_pay(cost, energies):
            ready = 1.0
            if max_dmg is None or dmg > max_dmg:
                max_dmg = dmg
    return float(min_cost), ready, (float(max_dmg) if max_dmg is not None else 0.0)


# ── БЛОК 2 (история хода): события извлекаются ИСКЛЮЧИТЕЛЬНО из obs['logs'] ("Events that
# have occurred since the last selection", cg.api.Observation.logs) — НЕ из
# decisions.jsonl['action']/y. Это намеренно: build_features() в live НЕ имеет доступа к
# тому, что main.py в итоге выбрал (main.py вызывает его для КАЖДОЙ опции решения ДО того,
# как посчитаны скоры и выбран pick — сам pick никогда не передаётся обратно фиче-билдеру, и
# трогать main.py запрещено заданием). obs['logs'] же — часть СЫРОГО obs, доступная И offline
# (тот же obs_shards), И live (тот же obs_dict) идентично, так что оба берут историю из ОДНОГО
# и того же источника одной и той же функцией (verbatim-порт, как decision_base_features).
# ЛОВУШКА (проверено на реальных решениях корпуса, эпизод 87556853 ref=23..27): активация
# Munkidori Adrena-Brain НЕ имеет своего LogType (cg.api.LogType не содержит "ABILITY") —
# видна только как HP_CHANGE(putDamageCounter=True/False), КОТОРЫЙ неотличим от тиков
# poison/burn на чекапе (та же сигнатура на реальных логах). Поэтому "способность" сюда
# сознательно НЕ включена (см. is_ability-эксперимент в истории задачи) — только 4 типа с
# однозначным LogType: PLAY/ATTACH/EVOLVE/ATTACK. is_attach добавлен вместо is_ability (даёт
# те же 3 булевых флага, что просил список-подсказка, но на надёжном сигнале).
def _extract_turn_log_events(obs: dict) -> list:
    """obs['logs'] -> список (type, card_id, attack_id) СВОИХ (playerIndex==me) событий
    PLAY/ATTACH/EVOLVE/ATTACK, в порядке журнала. type перекодирован в OPT_*-шкалу (не
    LogType), card_id — ПОРТ конвенции resolve_opt_card_id: у ATTACH это cardIdTarget (кого
    усиливает энергия), у остальных — cardId (сыгранная/эволюционировавшая/атакующая карта)."""
    cur = obs.get("current") or {}
    me = int(cur.get("yourIndex") or 0)
    events = []
    for entry in (obs.get("logs") or []):
        if not isinstance(entry, dict) or entry.get("playerIndex") != me:
            continue
        t = entry.get("type")
        if t == LOG_PLAY:
            events.append((OPT_PLAY, entry.get("cardId"), None))
        elif t == LOG_ATTACH:
            events.append((OPT_ATTACH, entry.get("cardIdTarget"), None))
        elif t == LOG_EVOLVE:
            events.append((OPT_EVOLVE, entry.get("cardId"), None))
        elif t == LOG_ATTACK:
            events.append((OPT_ATTACK, entry.get("cardId"), entry.get("attackId")))
    return events


def _history_features_from_buf(buf: list) -> dict:
    """buf[-1]=prev1 (самое недавнее) .. buf[-3]=prev3. Короче 3 -> SENTINEL/0 на недостающих
    слотах (натуральные 0 для булевых is_*-флагов — та же конвенция, что opt_is_attack/
    opt_is_end нигде не используют SENTINEL)."""
    feats = {}
    for k in range(3):
        p = f"prev{k + 1}"
        if k < len(buf):
            typ, cid, aid = buf[-(k + 1)]
            feats[f"{p}_type"] = float(typ) if typ is not None else SENTINEL
            feats[f"{p}_card_id"] = float(cid) if cid is not None else SENTINEL
            feats[f"{p}_attack_id"] = float(aid) if aid is not None else SENTINEL
            feats[f"{p}_is_attack"] = 1.0 if typ == OPT_ATTACK else 0.0
            feats[f"{p}_is_attach"] = 1.0 if typ == OPT_ATTACH else 0.0
            feats[f"{p}_is_play"] = 1.0 if typ == OPT_PLAY else 0.0
        else:
            feats[f"{p}_type"] = SENTINEL
            feats[f"{p}_card_id"] = SENTINEL
            feats[f"{p}_attack_id"] = SENTINEL
            feats[f"{p}_is_attack"] = 0.0
            feats[f"{p}_is_attach"] = 0.0
            feats[f"{p}_is_play"] = 0.0
    return feats


def turn_history_step(hist_state: dict, obs: dict) -> dict:
    """★ OFFLINE-сторона стейтового буфера БЛОКА 2. `hist_state` — dict{me: {"turn","buf",
    "ability_used"}}, ЛОКАЛЬНЫЙ на один эпизод (создаётся заново в process_episode -> граница
    игры это просто "новый вызов", отдельная сброс-логика не нужна). Буфер per-agent
    (mirror-эпизоды гоняют ДВУХ отслеживаемых экспертов через ОДИН и тот же episode_id, см.
    докстринг модуля). Смена turn -> сброс буфера и ability_used этого агента (граница хода).
    Вызывается РОВНО ОДИН раз на решение (в process_episode, где каждая decisions.jsonl-строка
    обрабатывается один раз) — в отличие от live-версии (bc_features_2b.py), которой из-за
    сигнатуры build_features() приходится вызываться на КАЖДУЮ опцию решения и себя
    дедуплицировать (см. докстринг _FEATURES_TEMPLATE про сигнатуру-дедуп) — единственное
    намеренное отличие offline/live реализаций этого стейта; сама формула (turn-сброс +
    _extract_turn_log_events + _history_features_from_buf + ability_used-детект) — ТА ЖЕ
    функция, дословно продублированная.

    ★ БЛОК 5 (H3, "способность уже использовалась в этом ходу") — ЧЕСТНЫЙ ПРОКСИ, не точный
    признак. Задание указало (и БЛОК 2 выше это подтверждает): активация способности Munkidori
    Adrena-Brain НЕ имеет своего LogType — обычным путём (obs['logs']) она НЕОТЛИЧИМА от тиков
    poison/burn на чекапе. Прокси: после активации способности игра ВСЕГДА проводит игрока
    через sub-select'ы CTX_DAMAGE_COUNTER(13)/CTX_REMOVE_DAMAGE_COUNTER(16)/
    CTX_REMOVE_DAMAGE_COUNTER_COUNT(40) ("adrena-target"-контексты, см. HEADLINE в
    fidelity_expert_topk.py) — это НЕ прямое чтение события активации, а вывод из наблюдения
    ПОСЛЕДСТВИЯ (какие select-контексты нам показал движок ЭТОГО хода). ПРОВЕРЕНО эмпирически
    на всём корпусе: 12528/12528 (100%) решений с sel.context в {13,16,40} происходят при
    own_munkidori_on_board==1 — необходимое условие держится железно. Это НЕ логическая
    гарантия: (а) в деке ТОЛЬКО у Munkidori есть способность такой формы (сейчас единственный
    источник этих контекстов), но если бы в дальнейшем деке добавилась другая карта с похожей
    "положи/убери damage counter" способностью — прокси стал бы неразличим между ними
    (флаг остался бы "какая-то способность этого типа сработала", не именно Munkidori); (б)
    признак называется "_this_turn", но фактически инвертирован по времени: он ЧИТАЕТ состояние
    ДО текущего решения (какие adrena-контексты УЖЕ встречались раньше в этом ходу) и
    ОБНОВЛЯЕТСЯ ПОСЛЕ — так что на самой ПЕРВОЙ MAIN-развилке хода, где Munkidori ещё не был
    активирован, флаг корректно 0.0, а начиная со следующей развилки после активации — 1.0,
    вплоть до смены хода (сброс). offline/live считают ОДИНАКОВО: оба читают ТОЛЬКО
    obs['select']['context'] текущего решения (часть сырого obs, доступная идентично в обеих
    средах — та же симметрия, что и БЛОК 2), не действие эксперта/пилота."""
    cur = obs.get("current") or {}
    sel = obs.get("select") or {}
    me = int(cur.get("yourIndex") or 0)
    st = hist_state.get(me)
    turn = cur.get("turn")
    if st is None or st["turn"] != turn:
        st = {"turn": turn, "buf": [], "ability_used": False}
        hist_state[me] = st
    st["buf"].extend(_extract_turn_log_events(obs))
    if len(st["buf"]) > 3:
        del st["buf"][:-3]
    feats = _history_features_from_buf(st["buf"])
    feats["own_munkidori_ability_used_this_turn"] = 1.0 if st["ability_used"] else 0.0
    if sel.get("context") in (CTX_DAMAGE_COUNTER, CTX_REMOVE_DAMAGE_COUNTER,
                               CTX_REMOVE_DAMAGE_COUNTER_COUNT):
        st["ability_used"] = True
    return feats


def _card_at_looking(looking: list, idx):
    if not isinstance(looking, list) or not isinstance(idx, int) or not (0 <= idx < len(looking)):
        return None
    c = looking[idx]
    if not isinstance(c, dict):
        return None
    cid = c.get("id")
    return int(cid) if cid is not None else None


def _card_at_sel_deck(sel_deck: list | None, idx):
    """★ ГЛАВНАЯ ПРАВКА 08-09 (оркестратор): при area=1 (deck) карта НЕ скрыта —
    движок кладёт полный список карт-кандидатов в obs['select']['deck'], а opt['index']
    это индекс ИМЕННО В ЭТОМ СПИСКЕ (проверено: 3866/3866 опций area=1 резолвятся, и
    выбор эксперта осмыслен — 648 Grimmsnarl ex / 647 Morgrem / 112 Munkidori / 104
    Froslass, т.е. тьютор своей линии). Раньше эти опции считались fog-of-war и уходили
    в 'hidden_deck_or_prize_zone', из-за чего группа c7 (TO_HAND, поиск по деке) училась
    БЕЗ признака «какую карту берём» и дала top-1 всего 0.2419.
    Ловушка «TO_HAND слеп» из памяти относится к ЧУЖОЙ руке/призам (area=6 действительно
    скрыт — проверено, select.deck там пуст), а не к собственному тьютору."""
    if not isinstance(sel_deck, list) or not isinstance(idx, int) or not (0 <= idx < len(sel_deck)):
        return None
    c = sel_deck[idx]
    if not isinstance(c, dict):
        return None
    cid = c.get("id")
    return int(cid) if cid is not None else None


def resolve_opt_card_id(players: list, me: int, opt: dict, looking: list | None = None,
                        sel_deck: list | None = None):
    """Порт (упрощённый) _resolve_ref из fidelity_expert_topk.py -> один card_id на опцию,
    None если опция не про конкретную карту (ATTACK/NUMBER/RETREAT/END/YES/NO/unknown).
    `looking` — obs['current']['looking'] (см. AREA_LOOKING выше);
    `sel_deck` — obs['select']['deck'] (см. _card_at_sel_deck)."""
    if not isinstance(opt, dict):
        return None
    if opt.get("area") == 1:
        cid = _card_at_sel_deck(sel_deck, opt.get("index"))
        if cid is not None:
            return cid
    if opt.get("cardId") is not None:
        try:
            return int(opt["cardId"])
        except (TypeError, ValueError):
            return None
    t = opt.get("type")
    if t == OPT_PLAY:
        return card_at(players, me, AREA_HAND, opt.get("index"))
    if t == OPT_ATTACH:
        # кого энергия усиливает (не саму энергию — она фунгибельна {D})
        return card_at(players, me, opt.get("inPlayArea"), opt.get("inPlayIndex"))
    if t == OPT_EVOLVE:
        # какая карта эволюционирует (из руки/зоны area)
        return card_at(players, me, opt.get("area", AREA_HAND), opt.get("index"))
    if t == OPT_ABILITY:
        return card_at(players, me, opt.get("area"), opt.get("index"))
    if t == OPT_CARD:
        area = opt.get("area")
        if area == AREA_LOOKING:
            return _card_at_looking(looking, opt.get("index"))
        pi = opt.get("playerIndex")
        pi = int(pi) if isinstance(pi, int) else me
        return card_at(players, pi, area, opt.get("index"))
    return None


# ── Порядок X-фич — КОНТРАКТ (feature_spec.json = этот список) ──────────────────────────
def _onehot_names(prefix: str) -> list:
    return [f"{prefix}_is_{cid}" for cid in DECK19] + [f"{prefix}_not_in_deck"]


FEATURE_ORDER: list = (
    # (a) глобальные
    ["turn", "turn_action_count", "is_first_player", "energy_attached", "supporter_played",
     "stadium_played", "retreated", "stadium_id", "n_logs"]
    # (b) свои
    + ["deck_count", "hand_count", "n_prize_left", "bench_n", "discard_n", "active_id"]
    + _onehot_names("active")
    + ["active_hp", "active_maxhp", "active_damage", "active_hp_frac", "active_n_energies",
       "active_n_tools", "own_asleep", "own_burned", "own_confused", "own_paralyzed",
       "own_poisoned", "bench_energy_sum", "n_lines_own"]
    # (c) оппонент
    + ["opp_deck_count", "opp_hand_count", "opp_n_prize_left", "opp_bench_n", "opp_discard_n",
       "opp_active_id"]
    + _onehot_names("opp_active")
    + ["opp_active_hp", "opp_active_maxhp", "opp_active_damage", "opp_active_hp_frac",
       "opp_active_n_energies", "opp_active_n_tools", "opp_asleep", "opp_burned",
       "opp_confused", "opp_paralyzed", "opp_poisoned", "opp_bench_energy_sum", "n_lines_opp",
       "opp_active_prize_value"]
    # (d) производные
    + ["prize_diff", "hp_diff", "bench_diff", "energy_diff", "turn_parity"]
    # (д) контекст решения
    + ["sel_context", "sel_type", "min_count", "max_count", "n_options",
       "remain_damage_counter", "remain_energy_cost", "effect_id", "context_card_id"]
    # (е) по опции
    + ["opt_type", "opt_index", "opt_area", "opt_in_play_area", "opt_in_play_index",
       "opt_player_rel", "opt_position", "opt_position_frac", "opt_card_id"]
    + _onehot_names("opt_card")
    + ["opt_card_prize_value", "opt_is_attack", "opt_attack_id", "opt_attack_damage",
       "opt_kills_opp_active", "opt_damage_minus_opp_hp", "opt_is_end"]
    # (ж) БЛОК 1 (T1.4, энергия) — своя часть решения: готовность активного к атаке +
    # Munkidori-guard-сигнал (см. compute_attack_ready_cost/_find_own_mon).
    + ["own_active_min_attack_cost", "own_active_attack_ready", "own_munkidori_on_board",
       "own_munkidori_energy_n", "own_munkidori_is_active", "own_munkidori_zero_energy",
       "own_hand_energy_n", "own_discard_energy_n"]
    # (ж) БЛОК 1 — по опции: цель ATTACH (см. mon_at/compute_attack_ready_cost).
    + ["opt_is_attach", "opt_attach_target_energy_n", "opt_attach_target_min_attack_cost",
       "opt_attach_target_ready_after"]
    # (з) БЛОК 2 (T1.4, история хода) — последние 1-3 события ЭТОГО ХОДА из obs['logs']
    # (см. _extract_turn_log_events/turn_history_step). prevN_type в шкале OPT_* (не LogType).
    + ["prev1_type", "prev1_card_id", "prev1_attack_id", "prev1_is_attack", "prev1_is_attach",
       "prev1_is_play"]
    + ["prev2_type", "prev2_card_id", "prev2_attack_id", "prev2_is_attack", "prev2_is_attach",
       "prev2_is_play"]
    + ["prev3_type", "prev3_card_id", "prev3_attack_id", "prev3_is_attack", "prev3_is_attach",
       "prev3_is_play"]
    # (и) БЛОК 3 (T1.5) — состав руки/бенча/сброса (свои) + видимый борд оппонента, см.
    # hand_bench_discard_features/opp_board_features. Закрывает "модель не видит состав руки/
    # бенча/сброса, только их размеры" — задание 08-09.
    + [f"own_hand_cnt_{cid}" for cid in DECK19]
    + ["own_hand_other_cnt", "own_hand_has_supporter", "own_hand_has_item"]
    + [f"own_bench_cnt_{cid}" for cid in DECK19]
    + ["own_bench_other_cnt"]
    + [f"own_board_n_{STAGE_NAMES[cid]}" for cid in (646, 647, 648, 860, 104)]
    + ["own_n_ready_to_evolve", "own_active_ready_to_evolve"]
    + [f"own_discard_cnt_{cid}" for cid in DECK19]
    + ["own_discard_other_cnt"]
    + [f"opp_board_cnt_{cid}" for cid in OPP_BOARD_TOPN]
    + ["opp_board_other_cnt"]
    # (к) БЛОК 3 — по опции: карта X, которую опция играет/эволюционирует/привязывает/берёт —
    # сколько ЕЩЁ копий X доступно в руке (см. resolved_from_hand в option_features) и сколько
    # уже в сбросе (не путать с opt_card_prize_value/opt_card_is_* — те про "что это за карта",
    # это — про "сколько таких карт ещё осталось у меня").
    + ["opt_card_hand_remaining_n", "opt_card_discard_n"]
    # (л) БЛОК 4 (H1, 08-09 итерация 4) — угроза оппонента + производные, см. mon_threat/
    # compute_card_attack_costs и докстринг decision_base_features.
    + ["opp_active_min_attack_cost", "opp_active_attack_ready", "opp_active_max_damage_to_own",
       "opp_lethal_to_own_active", "own_kills_opp_active_this_turn",
       "own_n_prize_left_after_exchange", "opp_n_prize_left_after_exchange"]
    # (м) БЛОК 5 (H3, 08-09 итерация 4) — "способность уже использовалась в этом ходу",
    # честный прокси (не точный признак), см. turn_history_step докстринг про ограничения.
    + ["own_munkidori_ability_used_this_turn"]
)
assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER)), "дубликат имени фичи в FEATURE_ORDER"

# Поля НЕ входящие в X — только для аудита/страт-анализа. Проверка #4 ниже требует, чтобы
# ни одно из них (и ничего производного от action/game_won/expert/opp_archetype) не попало
# в FEATURE_ORDER.
META_COLUMNS = ["decision_uid", "option_idx", "y", "group", "split", "episode_id",
                "agent_index", "turn_idx", "step_idx", "expert", "opp_archetype",
                "opp_has_crustle", "subj_archetype", "deck_key", "game_won",
                "sel_context_name", "n_options_meta", "action_len"]
# проверка #4 (утечка): X-фича не должна ТОЧНО совпадать с одним из этих имён — это
# единственные поля, что прямо кодируют выбор/итог эксперта. Раньше тут была substring-
# проверка на "action", которая ложно палила turn_action_count (легитимное поле
# current.turnActionCount, к action эксперта отношения не имеющее) — заменено на точное
# совпадение имени.
LEAKAGE_EXACT_NAMES = {"action", "action_len", "game_won", "won", "expert", "opp_archetype"}


def _mon(lst):
    """Первый непустой моно-словарь из active/bench-списка ([], [None], [{...}]) или None."""
    if isinstance(lst, list) and lst and isinstance(lst[0], dict):
        return lst[0]
    return None


def _status(players, pi, key):
    if 0 <= pi < len(players) and isinstance(players[pi], dict):
        return 1.0 if players[pi].get(key) else 0.0
    return 0.0


def decision_base_features(obs: dict, attack_ready_cost: dict, card_attack_costs: dict) -> dict:
    """Фичи (a)+(b)+(c)+(d)+(д)+(ж-своя часть)+(л, БЛОК 4/H1), общие для ВСЕХ опций одного
    решения. `attack_ready_cost` — см. compute_attack_ready_cost (БЛОК 1). `card_attack_costs`
    — см. compute_card_attack_costs (БЛОК 4)."""
    cur = obs.get("current") or {}
    me = int(cur.get("yourIndex") or 0)
    opp = 1 - me
    players = cur.get("players") or []
    mine = players[me] if me < len(players) else {}
    theirs = players[opp] if opp < len(players) else {}
    sel = obs.get("select") or {}

    stadium = cur.get("stadium") or []
    stadium_card = stadium[0] if stadium and isinstance(stadium[0], dict) else None
    stadium_id = stadium_card.get("id") if stadium_card else None

    def board(p):
        active_mon = _mon(p.get("active"))
        bench = [m for m in (p.get("bench") or []) if isinstance(m, dict)]
        bench_n = len(bench)
        bench_energy = sum(len(m.get("energies") or []) for m in bench)
        if active_mon is not None:
            hp, mx = active_mon.get("hp"), active_mon.get("maxHp")
            hp = hp if isinstance(hp, (int, float)) else None
            mx = mx if isinstance(mx, (int, float)) else None
            a_id = active_mon.get("id")
            a_id = int(a_id) if a_id is not None else None
            a_hp = float(hp) if hp is not None else SENTINEL
            a_mx = float(mx) if mx is not None else SENTINEL
            a_dmg = float(mx - hp) if (hp is not None and mx is not None) else SENTINEL
            a_frac = float(hp / mx) if (hp is not None and mx not in (None, 0)) else SENTINEL
            a_en = float(len(active_mon.get("energies") or []))
            a_tools = float(len(active_mon.get("tools") or []))
            n_lines = bench_n + 1
        else:
            a_id, a_hp, a_mx, a_dmg, a_frac = None, SENTINEL, SENTINEL, SENTINEL, SENTINEL
            a_en, a_tools = 0.0, 0.0
            n_lines = bench_n
        return {
            "deck_count": float(p.get("deckCount") if p.get("deckCount") is not None else SENTINEL),
            "hand_count": float(p.get("handCount") if p.get("handCount") is not None else SENTINEL),
            "n_prize_left": float(len(p.get("prize") or [])),
            "bench_n": float(bench_n),
            "discard_n": float(len(p.get("discard") or [])),
            "active_id": a_id, "active_hp": a_hp, "active_maxhp": a_mx,
            "active_damage": a_dmg, "active_hp_frac": a_frac,
            "active_n_energies": a_en, "active_n_tools": a_tools,
            "bench_energy_sum": float(bench_energy), "n_lines": float(n_lines),
        }

    own = board(mine)
    opp_b = board(theirs)

    feats = {
        "turn": float(cur.get("turn") if cur.get("turn") is not None else SENTINEL),
        "turn_action_count": float(cur.get("turnActionCount") if cur.get("turnActionCount") is not None else SENTINEL),
        "is_first_player": 1.0 if cur.get("firstPlayer") == me else 0.0,
        "energy_attached": 1.0 if cur.get("energyAttached") else 0.0,
        "supporter_played": 1.0 if cur.get("supporterPlayed") else 0.0,
        "stadium_played": 1.0 if cur.get("stadiumPlayed") else 0.0,
        "retreated": 1.0 if cur.get("retreated") else 0.0,
        "stadium_id": float(stadium_id) if stadium_id is not None else SENTINEL,
        "n_logs": float(len(obs.get("logs") or [])),

        "deck_count": own["deck_count"], "hand_count": own["hand_count"],
        "n_prize_left": own["n_prize_left"], "bench_n": own["bench_n"],
        "discard_n": own["discard_n"], "active_id": float(own["active_id"]) if own["active_id"] is not None else SENTINEL,
        "active_hp": own["active_hp"], "active_maxhp": own["active_maxhp"],
        "active_damage": own["active_damage"], "active_hp_frac": own["active_hp_frac"],
        "active_n_energies": own["active_n_energies"], "active_n_tools": own["active_n_tools"],
        "own_asleep": _status(players, me, "asleep"), "own_burned": _status(players, me, "burned"),
        "own_confused": _status(players, me, "confused"), "own_paralyzed": _status(players, me, "paralyzed"),
        "own_poisoned": _status(players, me, "poisoned"),
        "bench_energy_sum": own["bench_energy_sum"], "n_lines_own": own["n_lines"],

        "opp_deck_count": opp_b["deck_count"], "opp_hand_count": opp_b["hand_count"],
        "opp_n_prize_left": opp_b["n_prize_left"], "opp_bench_n": opp_b["bench_n"],
        "opp_discard_n": opp_b["discard_n"],
        "opp_active_id": float(opp_b["active_id"]) if opp_b["active_id"] is not None else SENTINEL,
        "opp_active_hp": opp_b["active_hp"], "opp_active_maxhp": opp_b["active_maxhp"],
        "opp_active_damage": opp_b["active_damage"], "opp_active_hp_frac": opp_b["active_hp_frac"],
        "opp_active_n_energies": opp_b["active_n_energies"], "opp_active_n_tools": opp_b["active_n_tools"],
        "opp_asleep": _status(players, opp, "asleep"), "opp_burned": _status(players, opp, "burned"),
        "opp_confused": _status(players, opp, "confused"), "opp_paralyzed": _status(players, opp, "paralyzed"),
        "opp_poisoned": _status(players, opp, "poisoned"),
        "opp_bench_energy_sum": opp_b["bench_energy_sum"], "n_lines_opp": opp_b["n_lines"],
        "opp_active_prize_value": (
            float(prize_value(opp_b["active_id"])) if opp_b["active_id"] is not None else SENTINEL),

        "sel_context": float(sel.get("context") if sel.get("context") is not None else SENTINEL),
        "sel_type": float(sel.get("type") if sel.get("type") is not None else SENTINEL),
        "min_count": float(sel.get("minCount") if sel.get("minCount") is not None else SENTINEL),
        "max_count": float(sel.get("maxCount") if sel.get("maxCount") is not None else SENTINEL),
        "n_options": float(len(sel.get("option") or [])),
        "remain_damage_counter": float(sel.get("remainDamageCounter") if sel.get("remainDamageCounter") is not None else SENTINEL),
        "remain_energy_cost": float(sel.get("remainEnergyCost") if sel.get("remainEnergyCost") is not None else SENTINEL),
        "effect_id": (float((sel.get("effect") or {}).get("id"))
                       if isinstance(sel.get("effect"), dict) and sel["effect"].get("id") is not None
                       else SENTINEL),
        "context_card_id": (float((sel.get("contextCard") or {}).get("id"))
                             if isinstance(sel.get("contextCard"), dict) and sel["contextCard"].get("id") is not None
                             else SENTINEL),
    }
    feats.update(onehot_deck19("active", own["active_id"]))
    feats.update(onehot_deck19("opp_active", opp_b["active_id"]))

    # (d) производные
    feats["prize_diff"] = own["n_prize_left"] - opp_b["n_prize_left"]
    feats["hp_diff"] = own["active_hp"] - opp_b["active_hp"]
    feats["bench_diff"] = own["bench_n"] - opp_b["bench_n"]
    feats["energy_diff"] = ((own["active_n_energies"] + own["bench_energy_sum"])
                             - (opp_b["active_n_energies"] + opp_b["bench_energy_sum"]))
    turn = cur.get("turn")
    feats["turn_parity"] = float(int(turn) % 2) if isinstance(turn, int) else SENTINEL

    # (ж) БЛОК 1 (T1.4, энергия) — своя часть: готовность активного к атаке + Munkidori-guard.
    own_active_mon = _mon(mine.get("active"))
    if own_active_mon is not None:
        _cost = attack_ready_cost.get(int(own_active_mon.get("id") or 0))
        feats["own_active_min_attack_cost"] = float(_cost) if _cost is not None else SENTINEL
        feats["own_active_attack_ready"] = (
            1.0 if (_cost is not None and own["active_n_energies"] >= _cost) else 0.0)
    else:
        feats["own_active_min_attack_cost"] = SENTINEL
        feats["own_active_attack_ready"] = 0.0

    munk_mon, munk_is_active = _find_own_mon(mine, MUNKIDORI_ID)
    if munk_mon is not None:
        _munk_en = float(len(munk_mon.get("energies") or []))
        feats["own_munkidori_on_board"] = 1.0
        feats["own_munkidori_energy_n"] = _munk_en
        feats["own_munkidori_is_active"] = 1.0 if munk_is_active else 0.0
        feats["own_munkidori_zero_energy"] = 1.0 if _munk_en == 0 else 0.0
    else:
        feats["own_munkidori_on_board"] = 0.0
        feats["own_munkidori_energy_n"] = SENTINEL
        feats["own_munkidori_is_active"] = 0.0
        feats["own_munkidori_zero_energy"] = 0.0

    feats["own_hand_energy_n"] = float(sum(
        1 for c in (mine.get("hand") or [])
        if isinstance(c, dict) and int(c.get("id") or 0) == BASIC_D_ENERGY_ID))
    feats["own_discard_energy_n"] = float(sum(
        1 for c in (mine.get("discard") or [])
        if isinstance(c, dict) and int(c.get("id") or 0) == BASIC_D_ENERGY_ID))

    # (л) БЛОК 4 (H1, 08-09 итерация 4) — угроза оппонента + производные (симметрия к "своей"
    # части БЛОКА 1 выше, но type-aware через mon_threat/energies_can_pay — см. их докстринги
    # про то, почему own_active_min_attack_cost/compute_attack_ready_cost сюда не годятся).
    own_active_mon_full = _mon(mine.get("active"))
    opp_active_mon_full = _mon(theirs.get("active"))
    own_energies = list(own_active_mon_full.get("energies") or []) if own_active_mon_full else []
    opp_energies = list(opp_active_mon_full.get("energies") or []) if opp_active_mon_full else []

    _opp_min_cost, _opp_ready, _opp_max_dmg_to_own = mon_threat(
        opp_b["active_id"], opp_energies, card_attack_costs)
    feats["opp_active_min_attack_cost"] = _opp_min_cost
    feats["opp_active_attack_ready"] = _opp_ready
    feats["opp_active_max_damage_to_own"] = _opp_max_dmg_to_own
    # "он убивает мой актив в следующий ход" — ОЦЕНКА ПО ТЕКУЩИМ видимым энергиям (см.
    # mon_threat докстринг): не знаем, что оппонент довесит энергии/усилит атаку ДО своего
    # хода — честная нижняя граница угрозы прямо сейчас, не гарантия будущего.
    feats["opp_lethal_to_own_active"] = (
        1.0 if (own["active_id"] is not None and opp_b["active_id"] is not None
                and own["active_hp"] != SENTINEL and _opp_max_dmg_to_own > 0
                and _opp_max_dmg_to_own >= own["active_hp"]) else 0.0)

    _own_min_cost2, _own_ready2, _own_max_dmg_to_opp = mon_threat(
        own["active_id"], own_energies, card_attack_costs)
    # "я убиваю его актив в этот ход" — признак УРОВНЯ РЕШЕНИЯ (opt_kills_opp_active в
    # option_features ниже — только по конкретной опции атаки; этот — "есть ли ВООБЩЕ такая
    # атака среди наших опций сейчас", не зависит от того, какая именно опция скорится).
    feats["own_kills_opp_active_this_turn"] = (
        1.0 if (opp_b["active_id"] is not None and opp_b["active_hp"] != SENTINEL
                and _own_max_dmg_to_opp > 0 and _own_max_dmg_to_opp >= opp_b["active_hp"])
        else 0.0)

    # "сколько призов останется у каждой стороны после этого размена" — проекция ОБОИХ
    # KO (наш этим ходом / чужой следующим), floor на 0 (n_prize_left не может уйти в минус).
    feats["own_n_prize_left_after_exchange"] = max(0.0, own["n_prize_left"] - (
        float(prize_value(opp_b["active_id"]))
        if feats["own_kills_opp_active_this_turn"] == 1.0 else 0.0))
    feats["opp_n_prize_left_after_exchange"] = max(0.0, opp_b["n_prize_left"] - (
        float(prize_value(own["active_id"]))
        if feats["opp_lethal_to_own_active"] == 1.0 else 0.0))

    # (и)+(к) БЛОК 3 (T1.5) — состав руки/бенча/сброса + видимый борд оппонента, см.
    # hand_bench_discard_features/opp_board_features. hbd_feats несёт также приватные
    # "_hand_counts"/"_discard_counts" (сырые Counter, не X-фичи) — нужны option_features
    # для (5) "опция в контексте руки"; сливаем в feats и оставляем приватные ключи внутри,
    # они не попадут в X, т.к. их нет в FEATURE_ORDER (та же защита, что у "_me"/"_players").
    hbd_feats = hand_bench_discard_features(mine, own["active_id"])
    feats.update(hbd_feats)
    feats.update(opp_board_features(theirs))

    # добавочные поля, нужные option_features но не входящие в X напрямую
    feats["_me"] = me
    feats["_players"] = players
    feats["_looking"] = cur.get("looking")
    feats["_sel_deck"] = sel.get("deck")
    feats["_opp_active_hp"] = opp_b["active_hp"]
    return feats


def option_features(base: dict, opt: dict, opt_idx: int, n_options: int, attack_dmg: dict,
                     attack_ready_cost: dict) -> dict:
    """Фичи (е)+(ж-по опции) для одной опции. `base` — результат decision_base_features (даёт
    _me/_players/_opp_active_hp, не входящие в X). `attack_ready_cost` — см. БЛОК 1."""
    me = base["_me"]
    players = base["_players"]
    looking = base["_looking"]
    opp_active_hp = base["_opp_active_hp"]

    t = opt.get("type") if isinstance(opt, dict) else None
    area = opt.get("area") if isinstance(opt, dict) else None
    idx = opt.get("index") if isinstance(opt, dict) else None
    ipa = opt.get("inPlayArea") if isinstance(opt, dict) else None
    ipi = opt.get("inPlayIndex") if isinstance(opt, dict) else None
    pidx = opt.get("playerIndex") if isinstance(opt, dict) else None

    opt_player_rel = 0.0
    if isinstance(pidx, int):
        opt_player_rel = 0.0 if pidx == me else 1.0

    card_id = resolve_opt_card_id(players, me, opt if isinstance(opt, dict) else {}, looking,
                                  base.get("_sel_deck"))
    is_attack = (t == OPT_ATTACK)
    attack_id = opt.get("attackId") if (isinstance(opt, dict) and is_attack) else None
    if is_attack and attack_id is not None:
        dmg = attack_dmg.get(int(attack_id), SENTINEL)
    else:
        dmg = SENTINEL
    if is_attack and dmg != SENTINEL and opp_active_hp != SENTINEL:
        kills = 1.0 if dmg >= opp_active_hp else 0.0
        dmg_minus_hp = dmg - opp_active_hp
    else:
        kills = 0.0
        dmg_minus_hp = SENTINEL

    feats = {
        "opt_type": float(t) if t is not None else SENTINEL,
        "opt_index": float(idx) if isinstance(idx, int) else SENTINEL,
        "opt_area": float(area) if isinstance(area, int) else SENTINEL,
        "opt_in_play_area": float(ipa) if isinstance(ipa, int) else SENTINEL,
        "opt_in_play_index": float(ipi) if isinstance(ipi, int) else SENTINEL,
        "opt_player_rel": opt_player_rel,
        "opt_position": float(opt_idx),
        "opt_position_frac": float(opt_idx) / n_options if n_options else SENTINEL,
        "opt_card_id": float(card_id) if card_id is not None else SENTINEL,
        "opt_card_prize_value": float(prize_value(card_id)) if card_id is not None else SENTINEL,
        "opt_is_attack": 1.0 if is_attack else 0.0,
        "opt_attack_id": float(attack_id) if (is_attack and attack_id is not None) else SENTINEL,
        "opt_attack_damage": float(dmg),
        "opt_kills_opp_active": kills,
        "opt_damage_minus_opp_hp": float(dmg_minus_hp),
        "opt_is_end": 1.0 if t == OPT_END else 0.0,
    }
    feats.update(onehot_deck19("opt_card", card_id))

    # (ж) БЛОК 1 (T1.4, энергия) — по опции: цель ATTACH (кого усиливаем), см. mon_at/
    # compute_attack_ready_cost. ATTACH всегда бьёт по СВОЕЙ стороне (inPlayArea/inPlayIndex
    # на `me` — та же конвенция, что resolve_opt_card_id уже применяет для OPT_ATTACH).
    is_attach = (t == OPT_ATTACH)
    target_mon = mon_at(players, me, ipa, ipi) if is_attach else None
    if target_mon is not None:
        _ten = float(len(target_mon.get("energies") or []))
        _tcost = attack_ready_cost.get(int(target_mon.get("id") or 0))
        feats["opt_is_attach"] = 1.0
        feats["opt_attach_target_energy_n"] = _ten
        feats["opt_attach_target_min_attack_cost"] = float(_tcost) if _tcost is not None else SENTINEL
        feats["opt_attach_target_ready_after"] = (
            1.0 if (_tcost is not None and (_ten + 1.0) >= _tcost) else 0.0)
    else:
        feats["opt_is_attach"] = 1.0 if is_attach else 0.0
        feats["opt_attach_target_energy_n"] = SENTINEL
        feats["opt_attach_target_min_attack_cost"] = SENTINEL
        feats["opt_attach_target_ready_after"] = 0.0

    # (к) БЛОК 3 (T1.5) — "опция в контексте руки": если card_id опции входит в DECK19,
    # сколько ЕЩЁ копий доступно в руке (декремент на 1, если ИМЕННО ЭТА опция резолвится
    # ИЗ руки — PLAY всегда area=hand, EVOLVE по умолчанию тоже area=hand если не указано
    # иное; см. resolve_opt_card_id) и сколько уже в сбросе (без декремента — сброс от этой
    # опции не меняется). Для ATTACH/ABILITY/CARD card_id — не сама карта руки, а мон на
    # столе/в looking/в тюторе; счётчик там читается БЕЗ декремента ("сколько ещё таких же
    # монов у меня в руке", тоже полезный сигнал, просто не "потрачу одну").
    hand_counts = base.get("_hand_counts") or {}
    discard_counts = base.get("_discard_counts") or {}
    if card_id is not None and int(card_id) in DECK19:
        cid_i = int(card_id)
        eff_area = area if isinstance(area, int) else AREA_HAND
        resolved_from_hand = (t == OPT_PLAY) or (t == OPT_EVOLVE and eff_area == AREA_HAND)
        hand_n = hand_counts.get(cid_i, 0)
        if resolved_from_hand:
            hand_n = max(0, hand_n - 1)
        feats["opt_card_hand_remaining_n"] = float(hand_n)
        feats["opt_card_discard_n"] = float(discard_counts.get(cid_i, 0))
    else:
        feats["opt_card_hand_remaining_n"] = SENTINEL
        feats["opt_card_discard_n"] = SENTINEL

    unresolved_reason = None
    if t in CARD_RESOLVABLE_TYPES and card_id is None:
        if area in (1, 6):
            # своя/чужая колода (1) и свои/чужие призы (6) не видны по fog-of-war ни одной
            # из сторон (призы лежат рубашкой вверх по правилам TCG до момента взятия) —
            # TO_HAND-ловушка, см. докстринг модуля и MEMORY feedback_fidelity_instrument_traps
            # ("TO_HAND слеп") — уже известная, не новая, дыра прибора.
            unresolved_reason = "hidden_deck_or_prize_zone"
        elif t == OPT_CARD and area == AREA_HAND and isinstance(pidx, int) and pidx != me:
            unresolved_reason = "hidden_opp_hand"
        elif t == OPT_ABILITY and area not in ZONE:
            # area-код вне {1..6} (напр. 7) у ABILITY-опций — тот же пробел уже есть в
            # эталонном резолвере fidelity_expert_topk.py::_resolve_ref (тоже card_at с
            # тем же ZONE), порт сохраняет это поведение, а не "чинит" его сам.
            unresolved_reason = "ability_area_outside_zone_table"
        else:
            unresolved_reason = "other_unexpected"
    return feats, unresolved_reason


def process_episode(ep_id: int, decs: list, obs_dir: Path, split_map: dict, attack_dmg: dict,
                     attack_ready_cost: dict, card_attack_costs: dict, group_rows: dict,
                     dec_y_sums: dict, dec_action_len: dict, unresolved_reasons: Counter,
                     oob_counter: list) -> int:
    shard_path = obs_dir / f"{ep_id}.jsonl"
    with open(shard_path, "r") as f:
        obs_list = [json.loads(line) for line in f]

    n_rows = 0
    split_label = split_map.get(int(ep_id), "UNKNOWN")
    # ★ БЛОК 2 (история хода): dict{me: {"turn","buf"}}, ЛОКАЛЬНЫЙ на этот эпизод (см.
    # turn_history_step) — mirror-эпизоды держат буфер РАЗДЕЛЬНО на agent_index 0/1.
    hist_state: dict = {}
    for d in decs:
        ref = d["obs_ref"]
        if not (0 <= ref < len(obs_list)):
            continue
        obs = obs_list[ref]
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        n_options = len(opts)
        action = list(d.get("action") or [])
        ctx = d.get("sel_context")
        group = route_group(ctx)

        decision_uid = f"{ep_id}:{d['agent_index']}:{ref}"
        dec_action_len[decision_uid] = len(action)

        base = decision_base_features(obs, attack_ready_cost, card_attack_costs)
        base.update(turn_history_step(hist_state, obs))

        y_sum = 0
        for i, opt in enumerate(opts):
            y = 1 if i in action else 0
            y_sum += y
            opt_feats, unresolved_reason = option_features(base, opt, i, n_options, attack_dmg,
                                                             attack_ready_cost)
            if unresolved_reason is not None:
                unresolved_reasons[unresolved_reason] += 1

            row = {
                "decision_uid": decision_uid, "option_idx": i, "y": y,
                "group": group, "split": split_label,
                "episode_id": int(ep_id), "agent_index": int(d["agent_index"]),
                "turn_idx": d.get("turn_idx"), "step_idx": d.get("step_idx"),
                "expert": d.get("expert"), "opp_archetype": d.get("opp_archetype"),
                "opp_has_crustle": d.get("opp_has_crustle"),
                "subj_archetype": d.get("subj_archetype"), "deck_key": d.get("deck_key"),
                "game_won": d.get("game_won"), "sel_context_name": d.get("sel_context_name"),
                "n_options_meta": n_options, "action_len": len(action),
            }
            for k in FEATURE_ORDER:
                if k in base:
                    row[k] = base[k]
                elif k in opt_feats:
                    row[k] = opt_feats[k]
                else:
                    row[k] = SENTINEL  # défensif, не должно случаться
            group_rows[group].append(row)
            n_rows += 1
        # для action-индексов вне диапазона (не должно случаться, но считаем явно)
        oob = sum(1 for a in action if not (0 <= a < n_options))
        oob_counter[0] += oob
        dec_y_sums[decision_uid] = y_sum
    return n_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", default=str(WORKTREE / "out" / "corpus_2b" / "decisions.jsonl"))
    ap.add_argument("--obs-dir", default=str(WORKTREE / "out" / "corpus_2b" / "obs_shards"))
    ap.add_argument("--split", default=str(WORKTREE / "out" / "corpus_2b" / "split" / "split.json"))
    ap.add_argument("--out-dir", default=str(WORKTREE / "out" / "corpus_2b" / "features"))
    ap.add_argument("--limit", type=int, default=0, help="0 = все эпизоды; иначе — только первые N эпизодов (смоук)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    obs_dir = Path(args.obs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_raw = json.loads(Path(args.split).read_text())
    split_map = {}
    for e in split_raw.get("train_episode_ids", []):
        split_map[int(e)] = "train"
    for e in split_raw.get("test_episode_ids", []):
        split_map[int(e)] = "test"

    attack_dmg = {int(a.attackId): int(a.damage) for a in all_attack()}
    card_db = {int(c.cardId): c for c in all_card_data()}
    attack_db = {int(a.attackId): a for a in all_attack()}
    attack_ready_cost = compute_attack_ready_cost(card_db, attack_db)
    print(f"[featurize] attack_ready_cost: {len(attack_ready_cost)} карт с кастуемой атакой "
          f"({attack_ready_cost})", file=sys.stderr)
    card_attack_costs = compute_card_attack_costs(card_db, attack_db)
    print(f"[featurize] card_attack_costs (БЛОК 4/H1): {len(card_attack_costs)} карт с атаками",
          file=sys.stderr)

    # decisions.jsonl подряд сгруппирован по episode_id (проверено заранее) — читаем стримингом.
    decs_path = Path(args.decisions)
    episodes_order = []
    by_ep = defaultdict(list)
    cur_ep = None
    with open(decs_path, "r") as f:
        for line in f:
            d = json.loads(line)
            ep = d["episode_id"]
            if ep != cur_ep:
                if args.limit and len(episodes_order) >= args.limit:
                    break
                episodes_order.append(ep)
                cur_ep = ep
            by_ep[ep].append(d)

    print(f"[featurize] эпизодов к обработке: {len(episodes_order)} "
          f"(limit={args.limit or 'ALL'})", file=sys.stderr)

    group_rows = defaultdict(list)
    dec_y_sums, dec_action_len = {}, {}
    unresolved_reasons = Counter()
    oob_counter = [0]
    total_rows = 0
    unknown_split_eps = 0

    for i, ep in enumerate(episodes_order):
        if int(ep) not in split_map:
            unknown_split_eps += 1
        total_rows += process_episode(ep, by_ep[ep], obs_dir, split_map, attack_dmg,
                                       attack_ready_cost, card_attack_costs, group_rows,
                                       dec_y_sums, dec_action_len, unresolved_reasons, oob_counter)
        if (i + 1) % 100 == 0:
            print(f"[featurize] {i+1}/{len(episodes_order)} эпизодов, {total_rows} строк...",
                  file=sys.stderr)

    id_cols = ["decision_uid", "option_idx", "y", "group", "split"]
    meta_only = [c for c in META_COLUMNS if c not in id_cols]
    full_col_order = id_cols + FEATURE_ORDER + meta_only

    stats = {"groups": {}, "checks": {}}
    check1_fail, check2_fail = [], []
    nan_inf_cols = defaultdict(int)

    for group, rows in group_rows.items():
        df = pd.DataFrame(rows)
        for c in full_col_order:
            if c not in df.columns:
                df[c] = SENTINEL
        df = df[full_col_order]

        # проверка #3: NaN/inf в X
        x = df[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce")
        nan_mask = x.isna()
        inf_mask = np.isinf(x.to_numpy(dtype="float64"))
        if nan_mask.to_numpy().any():
            for col in x.columns[nan_mask.any(axis=0)]:
                nan_inf_cols[f"{group}:{col}:nan"] += int(nan_mask[col].sum())
        if inf_mask.any():
            inf_cols = x.columns[inf_mask.any(axis=0)]
            for col in inf_cols:
                nan_inf_cols[f"{group}:{col}:inf"] += int(inf_mask[:, list(x.columns).index(col)].sum())

        out_path = out_dir / f"{group}.parquet"
        df.to_parquet(out_path, engine="pyarrow", index=False)

        n = len(df)
        n_dec = df["decision_uid"].nunique()
        stats["groups"][group] = {
            "n_rows": n, "n_decisions": n_dec,
            "train_share": round(float((df["split"] == "train").mean()), 4) if n else None,
            "test_share": round(float((df["split"] == "test").mean()), 4) if n else None,
            "unknown_split_share": round(float((df["split"] == "UNKNOWN").mean()), 4) if n else None,
            "y1_share": round(float(df["y"].mean()), 4) if n else None,
            "mean_n_options": round(n / n_dec, 3) if n_dec else None,
        }

        # проверка #1: сумма строк группы == сумма n_options решений этой группы
        expected = int(df.groupby("decision_uid")["n_options_meta"].first().sum())
        if expected != n:
            check1_fail.append((group, expected, n))

        # проверка #2: sum(y) по решению == len(action)
        by_dec = df.groupby("decision_uid")["y"].sum()
        for uid, ysum in by_dec.items():
            if int(ysum) != dec_action_len.get(uid, -999):
                check2_fail.append(uid)

        print(f"[featurize] {group}: {n} строк, {n_dec} решений -> {out_path}", file=sys.stderr)

    # feature_spec.json — контракт порядка X-колонок
    forbidden_hits = [k for k in FEATURE_ORDER if k in LEAKAGE_EXACT_NAMES]
    (out_dir / "feature_spec.json").write_text(json.dumps({
        "feature_order": FEATURE_ORDER,
        "n_features": len(FEATURE_ORDER),
        "deck19": DECK19,
        "sentinel": SENTINEL,
        "id_columns": id_cols,
        "meta_columns": meta_only,
    }, indent=2, ensure_ascii=False))

    stats["checks"] = {
        "check1_rows_eq_sum_noptions": {"pass": len(check1_fail) == 0, "fail_groups": check1_fail},
        "check2_sum_y_eq_len_action": {"pass": len(check2_fail) == 0,
                                        "n_fail": len(check2_fail),
                                        "sample_fail_uids": check2_fail[:10]},
        "check3_no_nan_inf": {"pass": len(nan_inf_cols) == 0, "offenders": dict(nan_inf_cols)},
        "check4_no_leakage_in_feature_order": {"pass": len(forbidden_hits) == 0,
                                                "hits": forbidden_hits},
        "action_indices_out_of_range": oob_counter[0],
        "opt_card_id_unresolved_when_expected": {
            "total": int(sum(unresolved_reasons.values())),
            "by_reason": dict(unresolved_reasons),
            "note": ("hidden_deck_or_prize_zone/hidden_opp_hand — не баг, а fog-of-war (карта "
                     "в закрытой колоде/призах/чужой руке физически не видна в obs, "
                     "известная 'TO_HAND слеп'-ловушка из MEMORY "
                     "feedback_fidelity_instrument_traps); ability_area_outside_zone_table — "
                     "тот же пробел уже есть в эталонном _resolve_ref "
                     "(fidelity_expert_topk.py), порт сохраняет поведение; other_unexpected — "
                     "единственная категория, требующая разбора."),
        },
        "episodes_missing_from_split_json": unknown_split_eps,
    }
    stats["total_rows"] = total_rows
    stats["total_decisions"] = len(dec_y_sums)
    stats["wall_sec"] = round(time.perf_counter() - t0, 1)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\n[featurize] done in {stats['wall_sec']}s, {total_rows} rows -> {out_dir}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
