#!/usr/bin/env python3
"""export_bc_2b.py — экспортирует 6 LightGBM-моделей out/corpus_2b/models/<group>.txt (T1.2,
train_bc_2b.py) в LIVE-SAFE чистый-Python артефакт: пара файлов, которые submissions/*/main.py
сможет импортировать БЕЗ numpy/pandas/lightgbm:

  out/corpus_2b/export/bc_model_2b.py    — деревья как кортежи чисел + predict_group(group,
                                            feat)->float, обычный цикл по узлам, без eval/exec/
                                            pickle.
  out/corpus_2b/export/bc_features_2b.py — build_features(obs_dict, option, option_idx,
                                            n_options)->list[float] (порядок = feature_spec.
                                            json, это контракт) + route_group(ctx)->str, читает
                                            СЫРОЙ obs_dict движка, без pandas.

ЗАЧЕМ pure-python: submissions/v076_final/main.py сознательно не тянет numpy/pandas/lightgbm
("три меньше точек отказа live" — см. его докстринг) — риск подтверждён на практике (MEMORY
project_kaggle_live_safety: __file__/ленивый import/numpy ABI на Kaggle молча роняли пилот в
greedy-fallback). BC-гибрид (PLAN_WR60 T1.3) обязан сохранить это свойство submission'а, а не
просто "работать у меня локально".

ЧТО ДЕЛАЕТ ЭТОТ СКРИПТ:
  1. Экспорт деревьев (export_group_trees): booster.dump_model()['tree_info'] (вложенный dict
     на дерево) -> ПЛОСКИЙ pre-order список узлов (feature_idx, value, left, right,
     default_left). feature_idx==-1 -> лист (value=leaf_value); иначе value=threshold, left/
     right — индексы детей в ТОМ ЖЕ списке. Проверено эмпирически (см. equivalence-тест ниже
     и ручную проверку в разработке): у всех 6 моделей objective=lambdarank,
     average_output=False, ВСЕ 138 фич числовые (num_cat=0, decision_type — всегда "<="),
     shrinkage=0.05 одинаков по всем деревьям и leaf_value из dump_model() УЖЕ включает
     shrinkage — booster.predict(X) == сумма leaf_value по всем деревьям БЕЗ доп. множителя
     и БЕЗ sigmoid на выходе (lambdarank не применяет link-функцию к предсказанию, в отличие
     от objective=binary).
  2. Экспорт фич (render_features_module): decision_base_features/option_features/
     resolve_opt_card_id/route_group и их вспомогательные функции — ПОЧТИ ДОСЛОВНЫЙ порт из
     scripts/featurize_corpus_2b.py (эти функции там и так уже чистый Python, не трогают
     numpy/pandas). Единственное отличие — таблицы PRIZE_VALUES (scripts/prize_values_2b.py)
     и attack_id->damage (из cg.api.all_attack(), доступно только через движок) ВШИВАЮТСЯ как
     Python-литералы, чтобы live-сабмит не тянул ни prize_values_2b.py, ни cg-движок ради
     самого скоринга фич (движок и так грузится сабмитом для игры — но фиче-билдер обязан
     работать и без него, тестируемо и изолированно).
  3. build_features НЕ кэширует decision_base_features между опциями одного решения по
     id(obs_dict): id() в CPython может переиспользоваться после GC у короткоживущих словарей
     — ложное попадание в кэш means подмена фич решения тихо и без ошибки. Бюджет live —
     600с/эпизод на ~150-200 решений (~3-4с/решение) — с огромным запасом даже при
     пересчёте base на каждую опцию (см. замер времени ниже), выигрыш кэша не стоит риска.
  4. ТЕСТ ЭКВИВАЛЕНТНОСТИ (главная приёмка, --n-test строк): решения сэмплируются ЦЕЛИКОМ
     (не отдельные опции вразнобой — иначе top-1 внутри решения нельзя проверить) из TEST-
     сплита, пропорционально числу test-строк каждой группы, пока суммарное число опций не
     достигнет --n-test. Для каждого решения: сырой obs берётся из obs_shards (episode_id:
     agent_index:obs_ref из decision_uid), опции прогоняются через build_features() СГЕНЕРИ-
     РОВАННОГО bc_features_2b.py и predict_group() СГЕНЕРИРОВАННОГО bc_model_2b.py (тестируем
     ИМЕННО артефакт, не параллельную копию логики). Три проверки:
       a) feat-vector от build_features() поэлементно совпадает с X-строкой из parquet
          (та же decision_uid+option_idx) — это проверяет bc_features_2b.py против
          featurize_corpus_2b.py (независимый эталон) с допуском 1e-9 (структурный порт,
          не переобучение — числа обязаны совпасть почти побитово).
       b) predict_group(group, feat) сравнивается с booster.predict(feat) той же LightGBM-
          модели на ТОМ ЖЕ feat-векторе — допуск max|diff|<1e-4 (задание).
       c) top-k (k=minCount решения, tie-break — меньший option_idx раньше, та же конвенция,
          что train_bc_2b.py::decision_level_eval) по pure-python-скорам сравнивается с
          top-k по lightgbm-скорам ТОГО ЖЕ решения — обязано совпасть в 100% случаев.
     ★ T1.4 (БЛОК 2, история хода): decision_base_features теперь стейтовая (модуль-левел
     буфер _turn_hist_* в bc_features_2b.py, см. его докстринг) — решения сэмплируются
     ВРАЗНОБОЙ из случайных эпизодов, так что изолированный вызов build_features() видел бы
     пустой/чужой буфер. _prime_turn_history() перед каждым решением ставит буфер в состояние
     "как будто честно доиграли досюда", используя ГОТОВЫЕ prev1-3 X-колонки этой же строки
     (независимый эталон — featurize_corpus_2b.py::turn_history_step, настоящий обход
     decisions.jsonl) — так check (a)/(b)/(c) продолжают тестировать ПЛАМБИНГ (buffer ->
     FEATURE_ORDER, SENTINEL), а не падают по методологии теста.
  5. ЗАМЕР ВРЕМЕНИ: build_features()+predict_group() на ВСЕ опции одного решения (то, что
     реально нужно на гейте — обойти select.option и выбрать k лучших), чистый Python,
     time.perf_counter() вокруг каждого решения из того же equivalence-сэмпла; печатает
     среднее/p99 и сравнивает с бюджетом 600с/~150-200 решений/эпизод.
  6. РАЗМЕР + ПРУНИНГ: bytes обоих файлов; если bc_model_2b.py > 5MB — prune_group_trees()
     обрезает КАЖДУЮ группу до первых N деревьев (LightGBM boosting аддитивен и
     последователен — обрезка С КОНЦА, не с начала, сохраняет самые ранние/крупные
     поправки), файл перегенерируется, top-1 на test пересчитывается ЗАНОВО (train_bc_2b.py-
     совместимая decision_level_eval) и печатается потерянный top-1 по каждой группе — так
     цена прунинга видна, а не скрыта.

Запуск:
    .venv/bin/python scripts/export_bc_2b.py                     # полный экспорт + тест (2000 строк)
    .venv/bin/python scripts/export_bc_2b.py --n-test 200         # смоук, быстрый прогон
    .venv/bin/python scripts/export_bc_2b.py --max-bytes 500000   # форсировать прунинг для проверки ветки
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

WORKTREE = Path(__file__).resolve().parent.parent
SCRIPTS = WORKTREE / "scripts"
SRC = WORKTREE / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prize_values_2b import PRIZE_VALUES as _PRIZE_VALUES_STR, DEFAULT_PRIZE_VALUE  # noqa: E402

# ── Native engine shim (macOS arm64) — dev-time ТОЛЬКО, для damage-таблицы атак.
# Сгенерированный bc_features_2b.py эту таблицу получает уже вшитой литералом и cg НЕ импортирует.
import cg_native  # noqa: E402
import cg_native.api  # noqa: E402

sys.modules.setdefault("cg", cg_native)
sys.modules.setdefault("cg.api", cg_native.api)
from cg.api import all_attack, all_card_data  # noqa: E402

GROUPS = ["main", "c7", "low", "mid", "easy", "rest"]

# ── БЛОК 1 (энергия) — DEV-TIME вычисление ATTACK_READY_COST (порт
# featurize_corpus_2b.py::compute_attack_ready_cost, см. её докстринг про диагностику).
# ДАННЫЕ (не код): нужны только здесь, чтобы посчитать таблицу ДО того, как она вошьётся
# литералом в bc_features_2b.py — сам вычислитель в живой артефакт не попадает.
_ENERGY_COLORLESS, _ENERGY_DARKNESS = 0, 7


def compute_attack_ready_cost(card_db: dict, attack_db: dict) -> dict:
    out = {}
    for cid, cd in card_db.items():
        costs = []
        for aid in (getattr(cd, "attacks", None) or []):
            atk = attack_db.get(int(aid))
            if atk is None:
                continue
            energies = list(getattr(atk, "energies", None) or [])
            if all(int(e) in (_ENERGY_COLORLESS, _ENERGY_DARKNESS) for e in energies):
                costs.append(len(energies))
        if costs:
            out[int(cid)] = min(costs)
    return out


# ── БЛОК 4 (H1) — DEV-TIME вычисление CARD_ATTACK_COSTS (порт
# featurize_corpus_2b.py::compute_card_attack_costs, см. её докстринг про то, почему
# ATTACK_READY_COST/compute_attack_ready_cost выше не годятся для чужого актива).
def compute_card_attack_costs(card_db: dict, attack_db: dict) -> dict:
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


def render_card_attack_costs(d: dict) -> str:
    """dict[int, tuple[(tuple[int,...], int), ...]] -> Python-литерал. Все значения уже
    builtin int/tuple -- repr() безопасен и даёт валидный литерал напрямую (в отличие от
    render_int_dict, которому нужен JSON-совместимый обход)."""
    ordered = {int(k): tuple((tuple(int(e) for e in cost), int(dmg)) for cost, dmg in v)
               for k, v in sorted(d.items())}
    return repr(ordered)


# ═══════════════════════════ 1. Экспорт деревьев ═══════════════════════════

def flatten_tree(node: dict) -> list:
    """dump_model()['tree_structure'] (вложенный dict, один узел = один dict с
    left_child/right_child) -> плоский pre-order список узлов (feature_idx, value, left,
    right, default_left). Индекс 0 всегда корень. feature_idx=-1 -> лист (value=leaf_value,
    left/right/default_left не используются, поставлены -1/-1/0)."""
    nodes: list = []

    def rec(n: dict) -> int:
        if "split_index" not in n:  # лист: {'leaf_index', 'leaf_value', 'leaf_weight', 'leaf_count'}
            nodes.append((-1, float(n["leaf_value"]), -1, -1, 0))
            return len(nodes) - 1
        my_idx = len(nodes)
        nodes.append(None)  # placeholder — left/right детей узнаем только после рекурсии
        left_idx = rec(n["left_child"])
        right_idx = rec(n["right_child"])
        nodes[my_idx] = (
            int(n["split_feature"]), float(n["threshold"]),
            left_idx, right_idx, 1 if n.get("default_left") else 0,
        )
        return my_idx

    rec(node)
    return nodes


def export_group_trees(booster: lgb.Booster) -> list:
    dump = booster.dump_model()
    if dump.get("objective") != "lambdarank":
        raise RuntimeError(
            f"неожиданный objective={dump.get('objective')!r} — predict_group() в bc_model_2b.py "
            "предполагает lambdarank (сумма leaf_value без sigmoid); нужна ревизия, если это изменится."
        )
    if dump.get("average_output"):
        raise RuntimeError("average_output=True неожидан для lambdarank — ревизия формулы predict_group()")
    return [flatten_tree(t["tree_structure"]) for t in dump["tree_info"]]


def render_node(node: tuple) -> str:
    f, v, l, r, dl = node
    return f"({f},{v!r},{l},{r},{dl})"


def render_tree(nodes: list) -> str:
    return "(" + ",".join(render_node(n) for n in nodes) + ",)"


def render_models_module(group_trees: dict, feature_spec: dict) -> str:
    lines = []
    lines.append(_MODEL_HEADER.replace("@@N_FEATURES@@", str(feature_spec["n_features"])))
    lines.append("MODELS = {")
    for group in GROUPS:
        trees = group_trees[group]
        lines.append(f'  "{group}": (')
        for nodes in trees:
            lines.append(f"    {render_tree(nodes)},")
        lines.append("  ),")
    lines.append("}")
    lines.append("")
    lines.append(_MODEL_FOOTER)
    return "\n".join(lines)


_MODEL_HEADER = '''"""bc_model_2b.py — LIVE-SAFE чистый-Python инференс BC-гибрида (T1.3a, PLAN_WR60 §T1.3).
СГЕНЕРИРОВАНО scripts/export_bc_2b.py из out/corpus_2b/models/<group>.txt — НЕ РЕДАКТИРОВАТЬ
РУКАМИ, перезапустить экспорт при переобучении моделей (train_bc_2b.py).

Ноль внешних зависимостей (не numpy/lightgbm/pandas/eval/exec/pickle) — submissions/
v076_final/main.py сознательно устроен так же (см. его докстринг "три меньше точек отказа
live"); BC-гибрид обязан сохранить это свойство.

ФОРМАТ ДЕРЕВА: каждая модель группы — tuple деревьев; дерево — tuple узлов в pre-order;
узел = (feature_idx, value, left, right, default_left):
  feature_idx == -1  -> ЛИСТ. value = leaf_value (финальный вклад в скор). left/right/
                         default_left не используются (placeholder -1/-1/0).
  feature_idx >= 0    -> ВНУТРЕННИЙ узел. value = threshold сравнения "feat[feature_idx] <=
                         threshold". left/right — индексы детей ВНУТРИ ТОГО ЖЕ tuple дерева.
                         default_left — куда идти, если feat[feature_idx] отсутствует/NaN
                         (в live такого не бывает — фичи всегда см. bc_features_2b.py с
                         сентинелом -1.0 вместо NaN/None — но узел хранит default_left для
                         точного повторения поведения LightGBM, если контракт когда-нибудь
                         нарушится).

@@N_FEATURES@@ фич на вектор, порядок = out/corpus_2b/features/feature_spec.json
["feature_order"] — контракт с bc_features_2b.py::build_features(), индексы должны совпадать
байт-в-байт, иначе веса дерева применятся не к тем фичам.

СЕМАНТИКА СКОРА (проверено экспортёром при обучении — см. export_bc_2b.py::
export_group_trees и отчёт эквивалентности T1.3a): все 6 моделей — objective="lambdarank"
(LambdaMART, NDCG@1, обученный в train_bc_2b.py). LightGBM НЕ применяет sigmoid/link-функцию
к выходу lambdarank-модели при predict() (в отличие от objective="binary", где predict()
даёт вероятность через sigmoid) — предсказание ЭТО сумма leaf_value по всем деревьям, и
leaf_value из dump_model() уже включает shrinkage (learning_rate=0.05 у всех моделей) —
никакого дополнительного умножения не нужно. predict_group() ниже — просто эта сумма.

ВЫБОР ОПЦИИ: скор — это НЕ вероятность, а ранжирующий балл (выше = эксперт вероятнее выбрал
бы эту опцию). Чтобы получить решение: посчитать predict_group() для КАЖДОЙ опции обс.select.
option, взять top-k по убыванию скора (k = obs.select.minCount, тай-брейк — меньший
option_idx раньше — та же конвенция, что train_bc_2b.py::decision_level_eval, чтобы offline-
измеренная top-1 fidelity была тем самым числом, что увидит гейт).
"""
from __future__ import annotations

'''

_MODEL_FOOTER = '''

def predict_group(group: str, feat) -> float:
    """Сырой скор lambdarank-модели группы `group` на векторе фич `feat` (list/tuple длины
    N_FEATURES, порядок = feature_spec.json). Сумма leaf_value по всем деревьям группы —
    ТО ЖЕ САМОЕ число, что booster.predict(X) у LightGBM (см. докстринг модуля и отчёт
    эквивалентности T1.3a: max|diff|<1e-4 на held-out строках, top-1 решений 100% совпадает).
    KeyError, если group не из {"main","c7","low","mid","easy","rest"} — намеренно: молчаливый
    fallback здесь замаскировал бы баг роутинга (см. MEMORY project_kaggle_live_safety)."""
    total = 0.0
    for nodes in MODELS[group]:
        node = nodes[0]
        f = node[0]
        while f != -1:
            v = feat[f]
            # v == v ложно только при NaN (стандартный no-import трюк) — в контракте
            # bc_features_2b.py NaN не производит (сентинел -1.0), но узел хранит
            # default_left на случай, если контракт когда-нибудь нарушится.
            go_left = (v <= node[1]) if v == v else bool(node[4])
            node = nodes[node[2] if go_left else node[3]]
            f = node[0]
        total += node[1]
    return total
'''


def prune_group_trees(group_trees: dict, keep_by_group: dict) -> dict:
    """Обрезка КАЖДОЙ группы до первых keep_by_group[group] деревьев (аддитивный бустинг —
    более поздние деревья дают всё меньшую поправку при early-stopping по NDCG, обрезка с
    КОНЦА последовательности сохраняет самые крупные/ранние поправки)."""
    return {g: trees[: keep_by_group.get(g, len(trees))] for g, trees in group_trees.items()}


# ═══════════════════════════ 2. Экспорт фич ═══════════════════════════

def render_int_dict(d: dict) -> str:
    """dict[int,int] -> Python-литерал с INT-ключами (json.dumps тут не годится — JSON
    object requires string keys, а нам нужны int-ключи для O(1) lookup без str(int(...)))."""
    items = ", ".join(f"{int(k)}: {int(v)}" for k, v in sorted(d.items()))
    return "{" + items + "}"


def render_features_module(feature_spec: dict, prize_values_int: dict, attack_damage: dict,
                            attack_ready_cost: dict, card_attack_costs: dict) -> str:
    src = _FEATURES_TEMPLATE
    src = src.replace("@@SENTINEL@@", repr(float(feature_spec["sentinel"])))
    src = src.replace("@@N_FEATURES@@", str(feature_spec["n_features"]))
    src = src.replace("@@DECK19@@", json.dumps(feature_spec["deck19"]))
    src = src.replace("@@FEATURE_ORDER@@", json.dumps(feature_spec["feature_order"], indent=2))
    src = src.replace("@@PRIZE_VALUES@@", render_int_dict(prize_values_int))
    src = src.replace("@@DEFAULT_PRIZE_VALUE@@", str(int(DEFAULT_PRIZE_VALUE)))
    src = src.replace("@@ATTACK_DAMAGE@@", render_int_dict(attack_damage))
    src = src.replace("@@ATTACK_READY_COST@@", render_int_dict(attack_ready_cost))
    src = src.replace("@@CARD_ATTACK_COSTS@@", render_card_attack_costs(card_attack_costs))
    return src


# NOTE: тело функций ниже — ПОЧТИ ДОСЛОВНЫЙ порт scripts/featurize_corpus_2b.py
# (decision_base_features/option_features/resolve_opt_card_id/card_at/_card_at_looking/
# onehot_deck19/_mon/_status/route_group) — те функции там УЖЕ чистый Python (numpy/pandas
# используются только в featurize вокруг них, для сборки DataFrame). Расхождение с
# featurize_corpus_2b.py допустимо ТОЛЬКО в способе получения PRIZE_VALUES/attack_dmg
# (там — импорт/движок, здесь — вшитые литералы) — проверяется тестом эквивалентности (a).
_FEATURES_TEMPLATE = '''"""bc_features_2b.py — LIVE-SAFE чистый-Python извлечение фич BC-гибрида (T1.3a, PLAN_WR60
§T1.3). СГЕНЕРИРОВАНО scripts/export_bc_2b.py — таблицы PRIZE_VALUES/ATTACK_DAMAGE вшиты
литералами (порт scripts/prize_values_2b.py и cg.api.all_attack() соответственно), логика
build_features — почти дословный порт scripts/featurize_corpus_2b.py::decision_base_features/
option_features/resolve_opt_card_id (та часть featurize и так уже была чистым Python).

Ноль внешних зависимостей: не numpy/pandas/cg. Работает НАПРЯМУЮ на сыром obs_dict, каким
его отдаёт движок (см. CLAUDE.md "СТРУКТУРА obs"), без to_observation_class().

КОНТРАКТ: build_features(obs_dict, option, option_idx, n_options) -> list[float] длины
N_FEATURES, порядок ТОЧНО = FEATURE_ORDER (= out/corpus_2b/features/feature_spec.json
["feature_order"]) — это то же самое, что видела модель при обучении (train_bc_2b.py читает
те же parquet-колонки в том же порядке). Разъезд порядка -> веса дерева применятся не к тем
фичам молча (без исключения) — единственная защита сейчас — тест эквивалентности T1.3a
(поэлементное сравнение с parquet X-строками), перегенерировать этот файл, если feature_spec
когда-нибудь изменится, а не редактировать руками.

СЕНТИНЕЛ: @@SENTINEL@@ вместо NaN/inf для "значения не существует по смыслу" (порт решения
featurize_corpus_2b.py — см. его докстринг про сентинел -1 и счётчики-исключения bench_n/
discard_n/active_n_energies/active_n_tools/bench_energy_sum/n_lines_* — там 0 корректное
значение, не сентинел).

БЕЗ КЭШИРОВАНИЯ base-фич между опциями одного решения по id(obs_dict): id() в CPython может
переиспользоваться после GC у короткоживущих словарей — ложное попадание в кэш подменило бы
фичи решения тихо. Бюджет live 600с/эпизод на ~150-200 решений даёт огромный запас даже с
пересчётом base на каждую опцию (см. замер времени export_bc_2b.py).
"""
from __future__ import annotations

SENTINEL = @@SENTINEL@@
N_FEATURES = @@N_FEATURES@@

# ── OptionType / area / zone константы (см. submissions/v076_final/main.py:365-423 /
# scripts/featurize_corpus_2b.py) ──────────────────────────────────────────────────────
OPT_NUMBER, OPT_YES, OPT_NO, OPT_CARD = 0, 1, 2, 3
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY = 7, 8, 9, 10
OPT_RETREAT, OPT_ATTACK, OPT_END = 12, 13, 14
AREA_HAND, AREA_ACTIVE, AREA_BENCH = 2, 4, 5
AREA_LOOKING = 12  # area-код у TO_HAND-опций из obs['current']['looking'] (Pokégear 3.0 и
# подобные) — НЕ путать с OPT_RETREAT=12 (то же число, другое пространство имён).
ZONE = {1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench", 6: "prize"}
CARD_RESOLVABLE_TYPES = {OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY, OPT_CARD}

# LogType-константы (см. cg.api.LogType) — нужны только БЛОКУ 2 (история хода).
LOG_PLAY, LOG_ATTACH, LOG_EVOLVE, LOG_ATTACK = 10, 11, 12, 15
MUNKIDORI_ID = 112
BASIC_D_ENERGY_ID = 7

# ── БЛОК 4 (H1)/БЛОК 5 (H3) константы — см. featurize_corpus_2b.py (та же секция там).
CTX_DAMAGE_COUNTER, CTX_REMOVE_DAMAGE_COUNTER, CTX_REMOVE_DAMAGE_COUNTER_COUNT = 13, 16, 40
ENERGY_RAINBOW, ENERGY_TEAM_ROCKET = 10, 11
_TEAM_ROCKET_PAYS = frozenset({5, 7})

# 19 карт деки Grimmsnarl-toolbox (порядок = feature_spec.json["deck19"], сортированный).
DECK19: list = @@DECK19@@
assert len(DECK19) == 19

# ── БЛОК 3 (T1.5, состав руки/бенча/сброса + видимый борд оппонента) — те же вшитые
# литералом ДАННЫЕ, что featurize_corpus_2b.py (см. её докстринг там про метод отбора
# OPP_BOARD_TOPN/эволюционную линию/cardType-классификацию). Не через @@...@@-подстановку —
# это фиксированные числа, известные на момент написания кода, а не производные от
# card_db/корпуса на каждом ретрейне (та же конвенция, что MUNKIDORI_ID/BASIC_D_ENERGY_ID).
EVOLUTION_NEXT = {646: 647, 647: 648, 860: 104}
STAGE_NAMES = {646: "impidimp", 647: "morgrem", 648: "grimmsnarl", 860: "snorunt", 104: "froslass"}
SUPPORTER_IDS = frozenset({1182, 1219, 1227, 1231})
ITEM_IDS = frozenset({1079, 1080, 1086, 1097, 1122, 1137, 1152})
OPP_BOARD_TOPN = [112, 648, 305, 646, 743, 741, 647, 742, 140, 381, 860, 345, 342, 401,
                  343, 400, 104, 341, 414, 431, 756, 344, 380, 379]
assert len(OPP_BOARD_TOPN) == 24 == len(set(OPP_BOARD_TOPN))

# Порядок X-фич — КОНТРАКТ, см. докстринг модуля. Порт feature_spec.json["feature_order"].
FEATURE_ORDER: list = @@FEATURE_ORDER@@
assert len(FEATURE_ORDER) == N_FEATURES == len(set(FEATURE_ORDER))

# ── Вшитые таблицы (порт данных, не кода — см. докстринг модуля) ──────────────────────
# scripts/prize_values_2b.py::PRIZE_VALUES, ключи переведены в int (там были str) — O(1)
# без str(int(...)) на каждый вызов. default = @@DEFAULT_PRIZE_VALUE@@ у карт не в таблице.
PRIZE_VALUES: dict = @@PRIZE_VALUES@@
DEFAULT_PRIZE_VALUE = @@DEFAULT_PRIZE_VALUE@@

# cg.api.all_attack(): {attackId: damage}. Вшито литералом — build_features не грузит cg.
ATTACK_DAMAGE: dict = @@ATTACK_DAMAGE@@

# ★ БЛОК 1 (энергия): card_id -> мин. энергии среди атак, кастуемых {D}-only декой. Порт
# featurize_corpus_2b.py::compute_attack_ready_cost (вшито литералом, та же причина, что
# ATTACK_DAMAGE — build_features не грузит cg). Карты без кастуемой атаки ОТСУТСТВУЮТ в dict.
ATTACK_READY_COST: dict = @@ATTACK_READY_COST@@

# ★ БЛОК 4 (H1): card_id -> tuple((cost_tuple, damage), ...) для ВСЕХ атак ЛЮБОЙ карты (не
# только D-only-кастуемых, как ATTACK_READY_COST выше) — порт
# featurize_corpus_2b.py::compute_card_attack_costs, нужна для честной оценки угрозы от
# ЛЮБОГО чужого актива (см. energies_can_pay/mon_threat ниже).
CARD_ATTACK_COSTS: dict = @@CARD_ATTACK_COSTS@@


def energies_can_pay(cost: tuple, avail: list) -> bool:
    """★ БЛОК 4 (H1). Порт featurize_corpus_2b.py::energies_can_pay (дословно) — точный
    бэктрекинг-мэтчинг cost-листа одной атаки против мультимножества прикреплённых энергий."""
    colored = sorted(c for c in cost if c != 0)
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


def mon_threat(card_id, energies: list) -> tuple:
    """★ БЛОК 4 (H1). Порт featurize_corpus_2b.py::mon_threat (дословно, CARD_ATTACK_COSTS —
    модульная константа здесь, там был параметр функции)."""
    if card_id is None:
        return SENTINEL, 0.0, SENTINEL
    entries = CARD_ATTACK_COSTS.get(int(card_id))
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


def prize_value(card_id, default: int = DEFAULT_PRIZE_VALUE) -> int:
    """Число призов за KO карты card_id (int/float/None). None -> default. Порт
    prize_values_2b.prize_value (там str-ключ str(int(card_id)); здесь int-ключ int(card_id),
    те же значения — таблица идентична, см. докстринг модуля)."""
    if card_id is None:
        return default
    try:
        return PRIZE_VALUES.get(int(card_id), default)
    except (TypeError, ValueError):
        return default


def route_group(ctx) -> str:
    """Роутинг select.context -> группа модели. Порт featurize_corpus_2b.py::route_group
    (ПОРТ ДАННЫХ схемы tetsutani main.py:103-114, см. его докстринг)."""
    try:
        ctx = int(ctx) if ctx is not None else -1
    except (TypeError, ValueError):
        ctx = -1
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
    """{prefix}_is_<cid> для каждой из 19 карт + {prefix}_not_in_deck. Порт
    featurize_corpus_2b.py::onehot_deck19."""
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


def _card_id_counts(cards) -> dict:
    """Список карт-словарей -> {card_id: count}. Порт featurize_corpus_2b.py::_card_id_counts
    (там Counter, здесь обычный dict — .get(key, 0) ведёт себя идентично для наших нужд, без
    импорта collections)."""
    c: dict = {}
    for card in (cards or []):
        if isinstance(card, dict) and card.get("id") is not None:
            try:
                cid = int(card["id"])
            except (TypeError, ValueError):
                continue
            c[cid] = c.get(cid, 0) + 1
    return c


def hand_bench_discard_features(mine: dict, own_active_id) -> dict:
    """★ БЛОК 3 (T1.5). Порт featurize_corpus_2b.py::hand_bench_discard_features (дословно,
    Counter -> dict, см. _card_id_counts)."""
    hand_counts = _card_id_counts(mine.get("hand"))
    bench_counts = _card_id_counts(mine.get("bench"))
    discard_counts = _card_id_counts(mine.get("discard"))

    feats: dict = {}
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

    board_counts = dict(bench_counts)
    if own_active_id is not None:
        board_counts[int(own_active_id)] = board_counts.get(int(own_active_id), 0) + 1
    for cid, name in STAGE_NAMES.items():
        feats[f"own_board_n_{name}"] = float(board_counts.get(cid, 0))

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
    """★ БЛОК 3 (T1.5). Порт featurize_corpus_2b.py::opp_board_features."""
    opp_list = list(theirs.get("active") or []) + list(theirs.get("bench") or [])
    counts = _card_id_counts(opp_list)
    known = 0
    feats: dict = {}
    for cid in OPP_BOARD_TOPN:
        n = counts.get(cid, 0)
        feats[f"opp_board_cnt_{cid}"] = float(n)
        known += n
    feats["opp_board_other_cnt"] = float(sum(counts.values()) - known)
    return feats


def card_at(players: list, pi, area, idx):
    """card_id по (playerIndex, area-код, index-в-зоне) или None. Порт
    featurize_corpus_2b.py::card_at (deck(1)/prize(6) НЕ резолвятся по fog-of-war — см.
    докстринг featurize_corpus_2b.py про "TO_HAND слеп")."""
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
    energies/hp), а не только id. Порт featurize_corpus_2b.py::mon_at."""
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
    либо (None, False). Порт featurize_corpus_2b.py::_find_own_mon."""
    active_mon = _mon(mine.get("active"))
    if active_mon is not None and int(active_mon.get("id") or 0) == card_id:
        return active_mon, True
    for m in (mine.get("bench") or []):
        if isinstance(m, dict) and int(m.get("id") or 0) == card_id:
            return m, False
    return None, False


# ── БЛОК 2 (история хода) — события извлекаются ИСКЛЮЧИТЕЛЬНО из obs['logs'] ("Events that
# have occurred since the last selection"), НЕ из того, что main.py в итоге выбрал (main.py
# вызывает build_features() для КАЖДОЙ опции решения ДО выбора pick — сам pick сюда никогда
# не передаётся). Порт featurize_corpus_2b.py::_extract_turn_log_events/
# _history_features_from_buf (там же — докстринг про ability-ловушку: cg.api.LogType не
# содержит "ABILITY", Adrena-Brain неотличим по логам от чекапа poison/burn, поэтому
# "способность" сюда сознательно не включена).
_turn_hist_turn = None       # номер хода, которому принадлежит _turn_hist_buf
_turn_hist_buf: list = []    # [(type, card_id, attack_id), ...] хронологически, макс. 3
_turn_hist_last_sig = None   # сигнатура последнего РЕШЕНИЯ, чьи logs уже собраны (дедуп —
# build_features() вызывается на КАЖДУЮ опцию решения, харвест должен случиться РОВНО раз)
_turn_hist_ability_used = False  # ★ БЛОК 5 (H3) — см. _turn_history_step докстринг про
# честный прокси "способность уже использовалась в этом ходу" и его ограничения.


def _extract_turn_log_events(obs: dict) -> list:
    """obs['logs'] -> [(type, card_id, attack_id), ...] СВОИХ (playerIndex==me) событий
    PLAY/ATTACH/EVOLVE/ATTACK. Порт featurize_corpus_2b.py::_extract_turn_log_events."""
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
    """buf[-1]=prev1 .. buf[-3]=prev3. Порт featurize_corpus_2b.py::_history_features_from_buf."""
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


def _turn_history_step(obs: dict) -> dict:
    """★ LIVE-сторона стейтового буфера БЛОКА 2 (модуль-левел globals _turn_hist_*, см. выше).
    ОТЛИЧИЕ от offline-версии (featurize_corpus_2b.py::turn_history_step): там эта функция
    вызывается РОВНО ОДИН раз на решение (цикл по decisions.jsonl); здесь build_features()
    вызывается ОДИН РАЗ НА КАЖДУЮ ОПЦИЮ решения (main.py: `for i, opt in enumerate(options):
    ...build_features(obs_dict, opt, i, n_options)`), поэтому нужна сигнатура-дедуп
    (`_turn_hist_last_sig`), чтобы харвест obs['logs'] произошёл РОВНО один раз на решение, а
    не n_options раз подряд. Сама формула (сброс по смене turn + _extract_turn_log_events +
    _history_features_from_buf) — ТА ЖЕ, дословно.
    Сброс на границе ИГРЫ: build_features() никогда не вызывается на deck-selection шаге
    (obs['select'] is None -> main.py обрабатывает это ДО _bc_decide и сюда не доходит), так
    что явного "новая игра" сигнала тут нет — но turn у новой игры начинается с малого числа
    (0/1/2, см. cg.api.State.turn), поэтому уже существующий сброс "turn изменился" срабатывает
    и на границе игры тоже (единственный конструктивный риск — редкое совпадение turn-номера
    на стыке двух игр, не более чем на 1 решение "залипшая" история — не корректность, не
    исключение).
    ★ БЛОК 5 (H3, "способность уже использовалась в этом ходу") — та же сигнатура-дедуп
    охраняет и _turn_hist_ability_used: он читается ДО апдейта (значение, накопленное
    решениями СТРОГО ДО этого) и обновляется ПОСЛЕ, ровно как офлайн-версия
    (featurize_corpus_2b.py::turn_history_step) — см. её докстринг про честный прокси
    (CTX_DAMAGE_COUNTER/CTX_REMOVE_DAMAGE_COUNTER/CTX_REMOVE_DAMAGE_COUNTER_COUNT — не прямое
    чтение события активации, LogType для ability отсутствует) и его ограничения."""
    global _turn_hist_turn, _turn_hist_buf, _turn_hist_last_sig, _turn_hist_ability_used
    cur = obs.get("current") or {}
    sel = obs.get("select") or {}
    turn = cur.get("turn")
    if turn != _turn_hist_turn:
        _turn_hist_turn = turn
        _turn_hist_buf = []
        _turn_hist_last_sig = None
        _turn_hist_ability_used = False
    sig = (turn, cur.get("turnActionCount"), len(obs.get("logs") or []), sel.get("context"),
           sel.get("type"), sel.get("minCount"), sel.get("maxCount"), len(sel.get("option") or []))
    if sig != _turn_hist_last_sig:
        _turn_hist_last_sig = sig
        _turn_hist_buf.extend(_extract_turn_log_events(obs))
        if len(_turn_hist_buf) > 3:
            del _turn_hist_buf[:-3]
        feats = _history_features_from_buf(_turn_hist_buf)
        feats["own_munkidori_ability_used_this_turn"] = 1.0 if _turn_hist_ability_used else 0.0
        if sel.get("context") in (CTX_DAMAGE_COUNTER, CTX_REMOVE_DAMAGE_COUNTER,
                                   CTX_REMOVE_DAMAGE_COUNTER_COUNT):
            _turn_hist_ability_used = True
        return feats
    feats = _history_features_from_buf(_turn_hist_buf)
    feats["own_munkidori_ability_used_this_turn"] = 1.0 if _turn_hist_ability_used else 0.0
    return feats


def _card_at_looking(looking, idx):
    if not isinstance(looking, list) or not isinstance(idx, int) or not (0 <= idx < len(looking)):
        return None
    c = looking[idx]
    if not isinstance(c, dict):
        return None
    cid = c.get("id")
    return int(cid) if cid is not None else None


def _card_at_sel_deck(sel_deck, idx):
    """area=1 (deck) НЕ скрыта: движок кладёт карты-кандидаты в obs['select']['deck'],
    а opt['index'] — индекс в этом списке. Порт featurize_corpus_2b.py::_card_at_sel_deck
    (правка 08-09: без неё группа c7/TO_HAND училась вслепую, top-1 0.24 вместо 0.57)."""
    if not isinstance(sel_deck, list) or not isinstance(idx, int) or not (0 <= idx < len(sel_deck)):
        return None
    c = sel_deck[idx]
    if not isinstance(c, dict):
        return None
    cid = c.get("id")
    return int(cid) if cid is not None else None


def resolve_opt_card_id(players: list, me: int, opt: dict, looking=None, sel_deck=None):
    """card_id опции или None, если опция не про конкретную карту. Порт
    featurize_corpus_2b.py::resolve_opt_card_id (упрощённый порт _resolve_ref из
    scripts/fidelity_expert_topk.py)."""
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
        return card_at(players, me, opt.get("inPlayArea"), opt.get("inPlayIndex"))
    if t == OPT_EVOLVE:
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


def _mon(lst):
    if isinstance(lst, list) and lst and isinstance(lst[0], dict):
        return lst[0]
    return None


def _status(players, pi, key):
    if 0 <= pi < len(players) and isinstance(players[pi], dict):
        return 1.0 if players[pi].get(key) else 0.0
    return 0.0


def decision_base_features(obs: dict) -> dict:
    """Фичи, общие для ВСЕХ опций одного решения. Порт
    featurize_corpus_2b.py::decision_base_features (дословно, кроме SENTINEL/prize_value/
    ATTACK_DAMAGE, которые здесь — модульные константы этого файла, а не импорт/движок)."""
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

    feats["prize_diff"] = own["n_prize_left"] - opp_b["n_prize_left"]
    feats["hp_diff"] = own["active_hp"] - opp_b["active_hp"]
    feats["bench_diff"] = own["bench_n"] - opp_b["bench_n"]
    feats["energy_diff"] = ((own["active_n_energies"] + own["bench_energy_sum"])
                             - (opp_b["active_n_energies"] + opp_b["bench_energy_sum"]))
    turn = cur.get("turn")
    feats["turn_parity"] = float(int(turn) % 2) if isinstance(turn, int) else SENTINEL

    # (ж) БЛОК 1 (энергия) — своя часть. Порт featurize_corpus_2b.py::decision_base_features
    # (ATTACK_READY_COST здесь — модульная константа, там был параметр функции).
    own_active_mon = _mon(mine.get("active"))
    if own_active_mon is not None:
        _cost = ATTACK_READY_COST.get(int(own_active_mon.get("id") or 0))
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

    # (л) БЛОК 4 (H1) — угроза оппонента + производные. Порт
    # featurize_corpus_2b.py::decision_base_features (тот же блок там, см. её докстринг).
    own_active_mon_full = _mon(mine.get("active"))
    opp_active_mon_full = _mon(theirs.get("active"))
    own_energies = list(own_active_mon_full.get("energies") or []) if own_active_mon_full else []
    opp_energies = list(opp_active_mon_full.get("energies") or []) if opp_active_mon_full else []

    _opp_min_cost, _opp_ready, _opp_max_dmg_to_own = mon_threat(opp_b["active_id"], opp_energies)
    feats["opp_active_min_attack_cost"] = _opp_min_cost
    feats["opp_active_attack_ready"] = _opp_ready
    feats["opp_active_max_damage_to_own"] = _opp_max_dmg_to_own
    feats["opp_lethal_to_own_active"] = (
        1.0 if (own["active_id"] is not None and opp_b["active_id"] is not None
                and own["active_hp"] != SENTINEL and _opp_max_dmg_to_own > 0
                and _opp_max_dmg_to_own >= own["active_hp"]) else 0.0)

    _own_min_cost2, _own_ready2, _own_max_dmg_to_opp = mon_threat(own["active_id"], own_energies)
    feats["own_kills_opp_active_this_turn"] = (
        1.0 if (opp_b["active_id"] is not None and opp_b["active_hp"] != SENTINEL
                and _own_max_dmg_to_opp > 0 and _own_max_dmg_to_opp >= opp_b["active_hp"])
        else 0.0)

    feats["own_n_prize_left_after_exchange"] = max(0.0, own["n_prize_left"] - (
        float(prize_value(opp_b["active_id"]))
        if feats["own_kills_opp_active_this_turn"] == 1.0 else 0.0))
    feats["opp_n_prize_left_after_exchange"] = max(0.0, opp_b["n_prize_left"] - (
        float(prize_value(own["active_id"]))
        if feats["opp_lethal_to_own_active"] == 1.0 else 0.0))

    # (з) БЛОК 2 (история хода) — стейтовый харвест, см. _turn_history_step выше. Несёт также
    # own_munkidori_ability_used_this_turn (БЛОК 5/H3), см. _turn_history_step докстринг.
    feats.update(_turn_history_step(obs))

    # (и)+(к) БЛОК 3 (T1.5) — состав руки/бенча/сброса + видимый борд оппонента. Порт
    # featurize_corpus_2b.py::decision_base_features (тот же блок там).
    hbd_feats = hand_bench_discard_features(mine, own["active_id"])
    feats.update(hbd_feats)
    feats.update(opp_board_features(theirs))

    # (н) БЛОК 6 (v085, диагноз-Б §5) — доли исчерпания колоды/призов + сжатый turn. Порт
    # featurize_corpus_2b.py::decision_base_features (тот же блок там, дословно).
    _own_total = (own["deck_count"] + own["discard_n"] + own["hand_count"]
                  if SENTINEL not in (own["deck_count"], own["discard_n"], own["hand_count"])
                  else None)
    feats["own_deck_frac"] = (own["deck_count"] / _own_total
                               if _own_total not in (None, 0) else SENTINEL)
    _opp_total = (opp_b["deck_count"] + opp_b["discard_n"] + opp_b["hand_count"]
                  if SENTINEL not in (opp_b["deck_count"], opp_b["discard_n"], opp_b["hand_count"])
                  else None)
    feats["opp_deck_frac"] = (opp_b["deck_count"] / _opp_total
                               if _opp_total not in (None, 0) else SENTINEL)
    feats["prize_frac_left"] = own["n_prize_left"] / 6.0
    feats["opp_prize_frac_left"] = opp_b["n_prize_left"] / 6.0
    feats["turn_capped"] = float(min(int(turn), 20)) if isinstance(turn, int) else SENTINEL

    feats["_me"] = me
    feats["_players"] = players
    feats["_looking"] = cur.get("looking")
    feats["_sel_deck"] = sel.get("deck")
    feats["_opp_active_hp"] = opp_b["active_hp"]
    # (о) БЛОК 7 (v085, диагноз-Б §1) — attack_id последней СВОЕЙ атаки этим ходом, из
    # obs['logs']. Порт featurize_corpus_2b.py::decision_base_features (тот же блок там).
    _last_atk_id = None
    for _t_ev, _cid_ev, _aid_ev in reversed(_extract_turn_log_events(obs)):
        if _t_ev == OPT_ATTACK:
            _last_atk_id = _aid_ev
            break
    feats["_last_own_attack_id"] = _last_atk_id
    return feats


def option_features(base: dict, opt: dict, opt_idx: int, n_options: int) -> dict:
    """Фичи одной опции. Порт featurize_corpus_2b.py::option_features (attack_dmg там был
    параметром функции — здесь модульная константа ATTACK_DAMAGE, тот же контракт)."""
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
        dmg = ATTACK_DAMAGE.get(int(attack_id), SENTINEL)
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

    # (ж) БЛОК 1 (энергия) — по опции: цель ATTACH. Порт featurize_corpus_2b.py::option_features.
    is_attach = (t == OPT_ATTACH)
    target_mon = mon_at(players, me, ipa, ipi) if is_attach else None
    if target_mon is not None:
        _ten = float(len(target_mon.get("energies") or []))
        _tcost = ATTACK_READY_COST.get(int(target_mon.get("id") or 0))
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

    # (к) БЛОК 3 (T1.5) — "опция в контексте руки". Порт featurize_corpus_2b.py::option_features
    # (тот же блок там, тот же decrement-only-for-hand-resolved-types расчёт).
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

    # (о) БЛОК 7 (v085, диагноз-Б §3/§5) — состояние ЦЕЛИ card-type опций. Порт
    # featurize_corpus_2b.py::option_features (тот же блок там, дословно; ATTACK_DAMAGE здесь
    # модульная константа, там был параметр attack_dmg).
    _target_pidx_resolved = pidx if isinstance(pidx, int) else me
    feats["opt_target_is_self"] = 1.0 if _target_pidx_resolved == me else 0.0
    if is_attach:
        card_target_mon = target_mon
    elif t == OPT_CARD:
        card_target_mon = mon_at(players, _target_pidx_resolved, area, idx)
    else:
        card_target_mon = None
    if card_target_mon is not None:
        _thp_raw = card_target_mon.get("hp")
        _tmx_raw = card_target_mon.get("maxHp")
        _thp = float(_thp_raw) if isinstance(_thp_raw, (int, float)) else None
        _tmx = float(_tmx_raw) if isinstance(_tmx_raw, (int, float)) else None
        feats["opt_target_hp"] = _thp if _thp is not None else SENTINEL
        feats["opt_target_hp_frac"] = (
            _thp / _tmx if (_thp is not None and _tmx not in (None, 0)) else SENTINEL)
        feats["opt_target_energy_n"] = float(len(card_target_mon.get("energies") or []))
        feats["opt_target_tools_n"] = float(len(card_target_mon.get("tools") or []))
        _last_atk_id = base.get("_last_own_attack_id")
        _last_atk_dmg = (ATTACK_DAMAGE.get(int(_last_atk_id), SENTINEL)
                          if _last_atk_id is not None else SENTINEL)
        if _last_atk_dmg != SENTINEL and _thp is not None:
            feats["opt_target_dmg_kills"] = 1.0 if _last_atk_dmg >= _thp else 0.0
            feats["opt_target_dmg_minus_hp"] = _last_atk_dmg - _thp
        else:
            feats["opt_target_dmg_kills"] = 0.0
            feats["opt_target_dmg_minus_hp"] = SENTINEL
    else:
        feats["opt_target_hp"] = SENTINEL
        feats["opt_target_hp_frac"] = SENTINEL
        feats["opt_target_energy_n"] = SENTINEL
        feats["opt_target_tools_n"] = SENTINEL
        feats["opt_target_dmg_kills"] = 0.0
        feats["opt_target_dmg_minus_hp"] = SENTINEL
    return feats


def build_features(obs_dict: dict, option: dict, option_idx: int, n_options: int) -> list:
    """КОНТРАКТ (см. докстринг модуля): list[float] длины N_FEATURES, порядок =
    FEATURE_ORDER. Пересчитывает decision_base_features на КАЖДЫЙ вызов (без кэша по
    id(obs_dict) — см. докстринг модуля, риск GC-коллизии id()). ★ ИСКЛЮЧЕНИЕ БЛОКА 2: сам
    буфер истории хода (_turn_hist_*, модуль-левел) НАМЕРЕННО переживает между вызовами —
    это единственное состояние во всём файле, что накапливается между решениями; см.
    _turn_history_step про сигнатуру-дедуп, которая делает повторные вызовы decision_base_
    features() на опциях ОДНОГО решения безопасными (харвест происходит один раз)."""
    base = decision_base_features(obs_dict)
    opt_feats = option_features(base, option, option_idx, n_options)
    row = {}
    for k in FEATURE_ORDER:
        if k in base:
            row[k] = base[k]
        elif k in opt_feats:
            row[k] = opt_feats[k]
        else:
            row[k] = SENTINEL
    return [row[k] for k in FEATURE_ORDER]
'''


# ═══════════════════════════ 3. Загрузка входов ═══════════════════════════

def load_feature_spec(features_dir: Path) -> dict:
    return json.loads((features_dir / "feature_spec.json").read_text())


def load_boosters(models_dir: Path) -> dict:
    return {g: lgb.Booster(model_file=str(models_dir / f"{g}.txt")) for g in GROUPS}


def load_generated_module(path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════ 4. Сэмплинг решений для теста ═══════════════════════════

def decode_decision_uid(uid: str) -> tuple:
    ep_id, agent_idx, obs_ref = uid.split(":")
    return int(ep_id), int(agent_idx), int(obs_ref)


def sample_decisions(features_dir: Path, n_test_rows: int, seed: int) -> dict:
    """decision_uid -> group, стратифицировано пропорционально числу test-строк каждой
    группы (пока суммарное число опций не достигнет n_test_rows; маленькие группы —
    используем ВСЕ их test-решения, если их меньше выделенной квоты)."""
    rng = np.random.default_rng(seed)
    group_test_rows = {}
    group_uid_sizes = {}
    for g in GROUPS:
        df = pd.read_parquet(features_dir / f"{g}.parquet", columns=["decision_uid", "split"])
        test_df = df[df["split"] == "test"]
        sizes = test_df.groupby("decision_uid").size()
        group_uid_sizes[g] = sizes
        group_test_rows[g] = int(sizes.sum())
    total_rows = sum(group_test_rows.values())

    chosen = {}
    chosen_counts = {}
    for g in GROUPS:
        sizes = group_uid_sizes[g]
        target = round(n_test_rows * group_test_rows[g] / total_rows) if total_rows else 0
        uids = sizes.index.to_numpy()
        rng.shuffle(uids)
        acc = 0
        n_uids = 0
        for u in uids:
            if acc >= target:
                break
            acc += int(sizes[u])
            chosen[u] = g
            n_uids += 1
        chosen_counts[g] = {"n_decisions": n_uids, "n_rows": acc, "target_rows": target,
                              "available_rows": int(group_test_rows[g])}
    return chosen, chosen_counts


def load_obs_for_decisions(obs_dir: Path, decision_uids: list) -> dict:
    """{decision_uid: obs} — группирует по episode_id, открывает каждый файл ровно один раз."""
    by_ep = defaultdict(list)
    for uid in decision_uids:
        ep_id, _agent_idx, obs_ref = decode_decision_uid(uid)
        by_ep[ep_id].append((uid, obs_ref))
    out = {}
    for ep_id, items in by_ep.items():
        shard_path = obs_dir / f"{ep_id}.jsonl"
        needed = {ref for _uid, ref in items}
        obs_by_ref = {}
        with open(shard_path, "r") as f:
            for i, line in enumerate(f):
                if i in needed:
                    obs_by_ref[i] = json.loads(line)
                    if len(obs_by_ref) == len(needed):
                        break
        for uid, ref in items:
            if ref in obs_by_ref:
                out[uid] = obs_by_ref[ref]
    return out


# ═══════════════════════════ 5. Тест эквивалентности + замер времени ═══════════════════════════

def topk_indices(scores: list, k: int) -> set:
    """top-k индексов по убыванию score, тай-брейк — меньший index раньше (та же конвенция,
    что train_bc_2b.py::decision_level_eval)."""
    if k <= 0:
        return set()
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return set(order[:k])


def _priming_events_from_row(row, sentinel: float) -> list:
    """★ БЛОК 2 test-priming (T1.4). Решения сэмплируются ВРАЗНОБОЙ из случайных эпизодов
    (sample_decisions) — изолированный вызов build_features() иначе видел бы ПУСТОЙ/чужой
    модуль-левел буфер истории (_turn_hist_buf), а не тот, что накопил бы живой агент, реально
    доигравший до этого хода. Чтобы check (a) тестировал именно ПЛАМБИНГ (buffer -> FEATURE_
    ORDER-колонки, SENTINEL-обработка) — то самое место, где реально мог бы случиться разъезд
    "разложили буфер не в те колонки" — а не падал по методологической причине, буфер
    ПРЕДЗАПОЛНЯЕТСЯ готовыми prev1-3 X-колонками ЭТОЙ ЖЕ строки (посчитаны featurize_corpus_2b.
    py настоящим обходом decisions.jsonl -- независимый эталон). Возвращает буфер в
    хронологическом порядке (buf[-1]=prev1), как ждёт _turn_history_step."""
    events = []
    for k in (3, 2, 1):  # от старого к новому -> buf[-1] окажется prev1
        typ = row.get(f"prev{k}_type")
        if typ is None or typ == sentinel:
            continue
        cid = row.get(f"prev{k}_card_id")
        aid = row.get(f"prev{k}_attack_id")
        events.append((
            int(typ),
            None if (cid is None or cid == sentinel) else int(cid),
            None if (aid is None or aid == sentinel) else int(aid),
        ))
    return events


def _hist_signature(obs: dict) -> tuple:
    """★ БЛОК 2 test-priming. Та же сигнатура-дедуп, что bc_features_2b.py::_turn_history_step
    вычислит САМА для этого obs -- предзаписываем её в _turn_hist_last_sig, чтобы
    build_features() НЕ харвестил obs['logs'] этого решения ПОВЕРХ уже готового (из
    _priming_events_from_row) буфера, задваивая prev1 и всё, что после."""
    cur = obs.get("current") or {}
    sel = obs.get("select") or {}
    return (cur.get("turn"), cur.get("turnActionCount"), len(obs.get("logs") or []),
            sel.get("context"), sel.get("type"), sel.get("minCount"), sel.get("maxCount"),
            len(sel.get("option") or []))


def _prime_turn_history(feat_mod, obs: dict, row) -> None:
    """Ставит модуль-левел стейт БЛОКА 2 (_turn_hist_turn/_turn_hist_buf/_turn_hist_last_sig/
    _turn_hist_ability_used сгенерированного bc_features_2b.py) в состояние "как будто агент
    честно доиграл до этого решения" — см. _priming_events_from_row/_hist_signature. Вызывается
    ПЕРЕД build_features() на КАЖДОМ сэмплированном решении (перезаписывает предыдущее —
    естественный сброс между решениями теста, никакой контаминации между сэмплами).
    ★ БЛОК 5 (H3): _turn_hist_ability_used праймится НАПРЯМУЮ из эталонного значения
    own_munkidori_ability_used_this_turn ЭТОЙ строки (посчитано featurize_corpus_2b.py
    независимым обходом decisions.jsonl) — та же логика, что и у _turn_hist_buf."""
    feat_mod._turn_hist_turn = row.get("turn")
    feat_mod._turn_hist_buf = _priming_events_from_row(row, feat_mod.SENTINEL)
    feat_mod._turn_hist_last_sig = _hist_signature(obs)
    feat_mod._turn_hist_ability_used = bool(
        row.get("own_munkidori_ability_used_this_turn") == 1.0)


def run_equivalence_test(features_dir: Path, obs_dir: Path, boosters: dict,
                          feat_mod, model_mod, n_test_rows: int, seed: int) -> dict:
    chosen, chosen_counts = sample_decisions(features_dir, n_test_rows, seed)
    print(f"[export_bc_2b] сэмпл решений по группам: {json.dumps(chosen_counts, ensure_ascii=False)}",
          file=sys.stderr)

    # X-строки parquet для выбранных decision_uid (нужны и как эталон feat-вектора, и как
    # прямой вход в booster.predict для изоляции "дерево верно сериализовано" отдельно от
    # "фичи верно построены").
    parquet_rows = {}
    for g in GROUPS:
        uids_g = [u for u, gg in chosen.items() if gg == g]
        if not uids_g:
            continue
        df = pd.read_parquet(features_dir / f"{g}.parquet")
        sub = df[df["decision_uid"].isin(uids_g)].sort_values(["decision_uid", "option_idx"])
        for uid, rows in sub.groupby("decision_uid"):
            parquet_rows[uid] = rows

    obs_by_uid = load_obs_for_decisions(obs_dir, list(chosen.keys()))
    feature_order = feat_mod.FEATURE_ORDER

    max_feat_diff = 0.0
    max_score_diff = 0.0
    n_rows_checked = 0
    n_decisions_checked = 0
    n_decisions_skipped = 0
    n_topk_match = 0
    n_topk_total = 0
    per_group_topk = defaultdict(lambda: [0, 0])
    decision_wall_times = []  # build_features+predict_group на ВСЕ опции одного решения

    for uid, group in chosen.items():
        if uid not in obs_by_uid or uid not in parquet_rows:
            n_decisions_skipped += 1
            continue
        obs = obs_by_uid[uid]
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        n_options = len(opts)
        prow = parquet_rows[uid]
        if len(prow) != n_options:
            n_decisions_skipped += 1
            continue
        # роутинг через СГЕНЕРИРОВАННЫЙ route_group, должен совпасть с group из parquet
        obs_group = feat_mod.route_group(sel.get("context"))
        if obs_group != group:
            n_decisions_skipped += 1
            continue

        booster = boosters[group]
        k = int(sel.get("minCount") or 0)
        k = max(0, min(k, n_options))

        # ★ БЛОК 2 test-priming (T1.4) — см. _prime_turn_history: решения тут сэмплированы
        # вразнобой (не последовательный replay игры), так что модуль-левел буфер истории
        # нужно предзаполнить эталонными prev1-3 ЭТОЙ строки ДО вызова build_features(),
        # иначе он читал бы пустой/чужой буфер с предыдущей итерации цикла.
        _prime_turn_history(feat_mod, obs, prow.iloc[0])

        t0 = time.perf_counter()
        feats_pp = [feat_mod.build_features(obs, opt, i, n_options) for i, opt in enumerate(opts)]
        scores_pp = [model_mod.predict_group(group, f) for f in feats_pp]
        decision_wall_times.append(time.perf_counter() - t0)

        # (a) feat-vector поэлементно против parquet X
        x_ref = prow[feature_order].to_numpy(dtype=np.float64)
        feats_arr = np.asarray(feats_pp, dtype=np.float64)
        fd = float(np.max(np.abs(feats_arr - x_ref))) if feats_arr.size else 0.0
        max_feat_diff = max(max_feat_diff, fd)

        # (b) predict_group против booster.predict, НА ТЕХ ЖЕ фичах, что построил build_features
        scores_lgb = booster.predict(feats_arr)
        sd = float(np.max(np.abs(np.asarray(scores_pp) - scores_lgb))) if len(scores_pp) else 0.0
        max_score_diff = max(max_score_diff, sd)

        # (c) top-k (k=minCount) — pure-python против lightgbm, ОБЯЗАНО совпасть 100%
        pred_pp = topk_indices(scores_pp, k)
        pred_lgb = topk_indices(list(scores_lgb), k)
        match = pred_pp == pred_lgb
        n_topk_match += int(match)
        n_topk_total += 1
        per_group_topk[group][0] += int(match)
        per_group_topk[group][1] += 1

        n_rows_checked += n_options
        n_decisions_checked += 1

    decision_wall_times.sort()
    n_dt = len(decision_wall_times)
    mean_wall = sum(decision_wall_times) / n_dt if n_dt else None
    p99_wall = decision_wall_times[min(n_dt - 1, int(math.ceil(0.99 * n_dt)) - 1)] if n_dt else None

    return {
        "n_decisions_checked": n_decisions_checked,
        "n_decisions_skipped": n_decisions_skipped,
        "n_rows_checked": n_rows_checked,
        "max_abs_feat_diff": max_feat_diff,
        "max_abs_score_diff": max_score_diff,
        "topk_match_rate": (n_topk_match / n_topk_total) if n_topk_total else None,
        "topk_match_n": n_topk_match, "topk_total_n": n_topk_total,
        "topk_match_by_group": {g: {"match": v[0], "total": v[1],
                                     "rate": round(v[0] / v[1], 4) if v[1] else None}
                                 for g, v in per_group_topk.items()},
        "decision_wall_sec_mean": mean_wall, "decision_wall_sec_p99": p99_wall,
        "decision_wall_sec_max": decision_wall_times[-1] if decision_wall_times else None,
        "sample_meta": chosen_counts,
    }


# ═══════════════════════════ 6. Прунинг (если файл > --max-bytes) ═══════════════════════════

def contiguous_bounds(uid: np.ndarray) -> tuple:
    n = len(uid)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    changes = np.where(uid[1:] != uid[:-1])[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [n]))
    return starts, ends


def predict_with_trees(trees: list, feat) -> float:
    """Та же логика, что bc_model_2b.py::predict_group, но берёт список деревьев напрямую
    (а не через MODELS[group]) — нужна, чтобы сравнить top-1 ДО/ПОСЛЕ прунинга без записи
    промежуточных .py-файлов на диск на каждой итерации подбора --max-bytes."""
    total = 0.0
    for nodes in trees:
        node = nodes[0]
        f = node[0]
        while f != -1:
            v = feat[f]
            go_left = (v <= node[1]) if v == v else bool(node[4])
            node = nodes[node[2] if go_left else node[3]]
            f = node[0]
        total += node[1]
    return total


def eval_top1_pure_python(features_dir: Path, group: str, trees: list, feature_order: list) -> dict:
    """top-1 fidelity на test той же группы, что train_bc_2b.py::decision_level_eval
    (упрощённая копия без баланса/категорий), скорится predict_with_trees (чистый Python,
    те же деревья, что пойдут в артефакт) — используется и для 'до', и 'после' прунинга,
    чтобы разница была ЧЕСТНО приписана числу деревьев, а не разнице lightgbm/pure-python."""
    df = pd.read_parquet(features_dir / f"{group}.parquet")
    df = df[df["split"] == "test"].sort_values(["decision_uid", "option_idx"])
    X = df[feature_order].to_numpy(dtype=np.float64)
    scores = [predict_with_trees(trees, row) for row in X]
    uid = df["decision_uid"].to_numpy()
    opt_idx = df["option_idx"].to_numpy()
    y = df["y"].to_numpy()
    min_count = df["min_count"].to_numpy()
    starts, ends = contiguous_bounds(uid)
    n_match = 0
    for s, e in zip(starts, ends):
        k = int(min_count[s])
        k = max(0, min(k, e - s))
        true_idx = set(opt_idx[s:e][y[s:e] == 1].tolist())
        pred_idx = topk_indices(scores[s:e], k)
        pred_idx = {int(opt_idx[s:e][i]) for i in pred_idx}
        n_match += int(pred_idx == true_idx)
    n_dec = len(starts)
    return {"n_decisions": n_dec, "top1": round(n_match / n_dec, 4) if n_dec else None}


# ═══════════════════════════ main ═══════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=str(WORKTREE / "out" / "corpus_2b" / "models"))
    ap.add_argument("--features-dir", default=str(WORKTREE / "out" / "corpus_2b" / "features"))
    ap.add_argument("--obs-dir", default=str(WORKTREE / "out" / "corpus_2b" / "obs_shards"))
    ap.add_argument("--out-dir", default=str(WORKTREE / "out" / "corpus_2b" / "export"))
    ap.add_argument("--n-test", type=int, default=2000, help="целевое суммарное число строк (опций) в тесте эквивалентности")
    ap.add_argument("--max-bytes", type=int, default=5 * 1024 * 1024, help="порог размера bc_model_2b.py, выше -> прунинг")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t_start = time.perf_counter()
    models_dir = Path(args.models_dir)
    features_dir = Path(args.features_dir)
    obs_dir = Path(args.obs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_spec = load_feature_spec(features_dir)
    feature_order = feature_spec["feature_order"]
    print(f"[export_bc_2b] feature_spec: {feature_spec['n_features']} фич", file=sys.stderr)

    # ── таблицы для bc_features_2b.py ──
    prize_values_int = {int(k): int(v) for k, v in _PRIZE_VALUES_STR.items()}
    attack_damage = {int(a.attackId): int(a.damage) for a in all_attack()}
    card_db = {int(c.cardId): c for c in all_card_data()}
    attack_db = {int(a.attackId): a for a in all_attack()}
    attack_ready_cost = compute_attack_ready_cost(card_db, attack_db)
    card_attack_costs = compute_card_attack_costs(card_db, attack_db)
    print(f"[export_bc_2b] PRIZE_VALUES={len(prize_values_int)} записей, "
          f"ATTACK_DAMAGE={len(attack_damage)} записей, "
          f"ATTACK_READY_COST={len(attack_ready_cost)} записей ({attack_ready_cost}), "
          f"CARD_ATTACK_COSTS={len(card_attack_costs)} записей (БЛОК 4/H1)",
          file=sys.stderr)

    # ── деревья ──
    boosters = load_boosters(models_dir)
    group_trees = {g: export_group_trees(b) for g, b in boosters.items()}
    n_trees_orig = {g: len(t) for g, t in group_trees.items()}
    n_nodes_orig = {g: sum(len(tree) for tree in t) for g, t in group_trees.items()}
    print(f"[export_bc_2b] деревья по группам: {n_trees_orig}, узлов: {n_nodes_orig}", file=sys.stderr)

    # ── рендер + запись ──
    features_path = out_dir / "bc_features_2b.py"
    model_path = out_dir / "bc_model_2b.py"
    features_src = render_features_module(feature_spec, prize_values_int, attack_damage,
                                           attack_ready_cost, card_attack_costs)
    model_src = render_models_module(group_trees, feature_spec)
    features_path.write_text(features_src)
    model_path.write_text(model_src)
    compile(features_src, str(features_path), "exec")  # синтаксис-чек до импорта
    compile(model_src, str(model_path), "exec")
    model_bytes = model_path.stat().st_size
    features_bytes = features_path.stat().st_size
    print(f"[export_bc_2b] записано: {model_path} ({model_bytes} bytes), "
          f"{features_path} ({features_bytes} bytes)", file=sys.stderr)

    pruning_report = None
    if model_bytes > args.max_bytes:
        print(f"[export_bc_2b] bc_model_2b.py {model_bytes} > --max-bytes {args.max_bytes} -> прунинг",
              file=sys.stderr)
        # прунинг пропорционально: обрезаем каждую группу до доли деревьев, дающей нужный размер
        # (грубая оценка по узлам, затем итеративно ужимаем при недостаче)
        keep_frac = args.max_bytes / model_bytes
        keep_by_group = {g: max(1, int(round(n * keep_frac))) for g, n in n_trees_orig.items()}
        pruned_trees = prune_group_trees(group_trees, keep_by_group)
        pruned_src = render_models_module(pruned_trees, feature_spec)
        while len(pruned_src.encode()) > args.max_bytes and any(v > 1 for v in keep_by_group.values()):
            keep_by_group = {g: max(1, int(v * 0.9)) for g, v in keep_by_group.items()}
            pruned_trees = prune_group_trees(group_trees, keep_by_group)
            pruned_src = render_models_module(pruned_trees, feature_spec)
        model_path.write_text(pruned_src)
        compile(pruned_src, str(model_path), "exec")
        model_bytes = model_path.stat().st_size
        print(f"[export_bc_2b] после прунинга: {model_bytes} bytes, деревьев={keep_by_group}",
              file=sys.stderr)

        before_after = {}
        for g in GROUPS:
            full = eval_top1_pure_python(features_dir, g, group_trees[g], feature_order)
            pruned = eval_top1_pure_python(features_dir, g, pruned_trees[g], feature_order)
            before_after[g] = {
                "n_trees_full": n_trees_orig[g], "top1_full": full["top1"],
                "n_trees_kept": keep_by_group[g], "top1_pruned": pruned["top1"],
                "top1_lost": (round(full["top1"] - pruned["top1"], 4)
                              if full["top1"] is not None and pruned["top1"] is not None else None),
            }
        pruning_report = {"keep_by_group": keep_by_group, "top1_before_after": before_after}
    else:
        print(f"[export_bc_2b] {model_bytes} bytes <= --max-bytes {args.max_bytes} — прунинг не требуется",
              file=sys.stderr)

    # ── импорт сгенерированных модулей и тест эквивалентности ──
    feat_mod = load_generated_module(features_path, "bc_features_2b")
    model_mod = load_generated_module(model_path, "bc_model_2b")

    assert feat_mod.FEATURE_ORDER == feature_order, "FEATURE_ORDER разъехался с feature_spec.json"
    for g in GROUPS:
        assert g in model_mod.MODELS, f"группа {g} отсутствует в MODELS"

    equiv = run_equivalence_test(features_dir, obs_dir, boosters, feat_mod, model_mod,
                                  args.n_test, args.seed)

    ok = (equiv["max_abs_score_diff"] < 1e-4 and equiv["topk_match_rate"] == 1.0
          and equiv["n_decisions_checked"] > 0)

    report = {
        "n_features": feature_spec["n_features"],
        "trees_per_group": n_trees_orig, "nodes_per_group": n_nodes_orig,
        "model_bytes": model_bytes, "features_bytes": features_bytes,
        "total_bytes": model_bytes + features_bytes,
        "pruning": pruning_report,
        "equivalence": equiv,
        "budget_check": {
            "live_budget_sec_per_episode": 600,
            "typical_decisions_per_episode_low": 150, "typical_decisions_per_episode_high": 200,
            "implied_budget_sec_per_decision_high_load": round(600 / 200, 3),
            "measured_mean_sec_per_decision": equiv["decision_wall_sec_mean"],
            "measured_p99_sec_per_decision": equiv["decision_wall_sec_p99"],
            "comfortably_within_budget": (
                equiv["decision_wall_sec_p99"] is not None
                and equiv["decision_wall_sec_p99"] < (600 / 200) * 0.1
            ),
        },
        "pass": ok,
        "wall_sec": round(time.perf_counter() - t_start, 1),
    }
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[export_bc_2b] {'PASS' if ok else 'FAIL'} — done in {report['wall_sec']}s -> {out_dir}",
          file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
