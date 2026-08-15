"""bc_model_2b.py — LIVE-SAFE чистый-Python инференс BC-гибрида (T1.3a, PLAN_WR60 §T1.3).
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

293 фич на вектор, порядок = out/corpus_2b/features/feature_spec.json
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


# ── v087: данные деревьев вынесены в part-модули (пик компиляции CPython
# ~155MB/MB исходника платится ЗА МОДУЛЬ; монолит 10.7MB давал пик 1.66GB,
# сборка из частей возвращает пик в проверенную Kaggle-зону <=714MB) ──
import bc_model_p1
import bc_model_p2
import bc_model_p3
import bc_model_p4
import bc_model_p5
import bc_model_p6
import bc_model_p7
import bc_model_p8
import bc_model_p9

MODELS = {
  "main": bc_model_p1.TREES_main_0 + bc_model_p2.TREES_main_1 + bc_model_p3.TREES_main_2 + bc_model_p4.TREES_main_3 + bc_model_p5.TREES_main_4 + bc_model_p6.TREES_main_5,
  "c7": bc_model_p7.TREES_c7_0 + bc_model_p8.TREES_c7_1,
  "low": bc_model_p8.TREES_low_0,
  "mid": bc_model_p8.TREES_mid_0,
  "easy": bc_model_p9.TREES_easy_0,
  "rest": bc_model_p9.TREES_rest_0,
}



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
