#!/usr/bin/env python3
"""train_bc_2b.py — обучает ПО МОДЕЛИ LightGBM-РАНКЕРА НА ГРУППУ РОУТИНГА (main/c7/low/mid/
easy/rest) поверх таблиц out/corpus_2b/features/<group>.parquet (см. featurize_corpus_2b.py)
и меряет top-1 fidelity на held-out тесте против якоря submissions/v076_final (offline top-1
0.5906 overall / 0.605 attack, out/corpus_2b/fidelity/v076_final_test.json).

СХЕМА (порт tetsutani-ранкера, см. featurize_corpus_2b.py docstring): модель скорит КАЖДУЮ
опцию решения отдельным числом; предсказание решения = top-k опций по скору, k=minCount
(см. "ГЛАВНАЯ МЕТРИКА" ниже). LightGBM учится либо через objective='binary' (каждая строка-
опция — независимый пример y=0/1, дисбаланс класса компенсирует is_unbalance), либо через
objective='lambdarank' (LambdaMART, группа = все опции одного решения, релевантность 0/1,
метрика NDCG@1). Обе схемы дают admissible скор для top-k — какая лучше, ЭМПИРИЧЕСКИ решает
--objective auto (по умолчанию): обе обучаются на группе 'main' (крупнейшая, 31756 решений)
с одинаковыми train/val-эпизодами и гиперпараметрами, сравниваются по top-1 НА VAL (не на
test — val для этого и существует, чтобы не подглядывать), победитель применяется КО ВСЕМ
группам. См. распечатанный objective_selection в metrics.json.

ГЛАВНАЯ МЕТРИКА (задание): top-1 EXACT MATCH на уровне РЕШЕНИЯ (не строки-опции).
  k = minCount решения (не maxCount — так и v076_final устроен: пилот всегда возвращает
  ровно minCount опций через свой scorer/top-k, кроме явных guard-веток; см. PLAN_WR60).
  predicted = top-k опций этого решения по скору модели (ties → меньший option_idx раньше,
  тот же tie-break что у "always index0"-бейзлайна — иначе разница будет артефактом
  тай-брейка, а не модели).
  match = set(predicted_option_idx) == set(expert_option_idx).
Из-за фиксированного k=minCount решения, где эксперт реально выбрал БОЛЬШЕ minCount опций
(minCount != maxCount, есть в c7/low/mid/easy — см. проверку ниже) СТРУКТУРНО не могут дать
match=1 при этой инструментальной конвенции (predicted и true сеты разного размера) — это
не баг метрики, а её заявленное определение из задания, обе схемы (модель/бейзлайны) страдают
одинаково, сравнение всё равно честное.

ОБЯЗАТЕЛЬНЫЕ КОНТРОЛИ (на TEST, не на val):
  chance_floor      — среднее по решениям P(match | случайный легальный набор размера k):
                       1/C(n_options, k), ЕСЛИ реальный ответ эксперта тоже размера k (иначе
                       0.0 — набор размера k структурно не может совпасть с ответом другого
                       размера, см. decision_level_eval). n_options разный по решениям —
                       считается ПОРЕШЕНИЙно и усредняется, не по формуле от средних.
  baseline_first_k  — детерминированный бейзлайн "всегда первые k опций по порядку в
                       obs.select.option" (обобщение "всегда option_idx=0" на k>1).
Модель обязана бить ОБА контроля — если нет, это провал, а не успех "выше нуля".

РЕГУЛЯРИЗАЦИЯ (590 игр — риск переобучения на конкретных партиях):
  - num_leaves/min_data_in_leaf подобраны ПО РАЗМЕРУ inner-train группы (маленькие группы —
    маленькие деревья, см. pick_hyperparams); без этого 'rest' (270 строк) переобучился бы
    моментально при дефолтных num_leaves=31.
  - early stopping ПО ЭПИЗОДАМ: val-эпизоды — отдельные ЦЕЛЫЕ игры (split_val_episodes),
    никогда не по решениям (иначе решения одной игры утекли бы между train/val — ровно та
    же ловушка, которую split.json уже закрыл для train/test).
  - bagging/feature_fraction + lambda_l2 — стандартная защита GBDT от переобучения на шуме
    отдельных строк-опций.
  - val ИЗ train, TEST не трогается до financial-числа в конце (используется РОВНО один раз
    на группу — финальная оценка сохранённой модели).

ЛОВУШКА "решение внутри группы контекста": строки одного decision_uid в parquet идут
ПОДРЯД (featurize_corpus_2b.py пишет их так; булева фильтрация по train/test/val-эпизодам
удаляет решения ЦЕЛИКОМ, не переставляя оставшиеся строки) — decision_level_eval() полагается
на это (contiguous_bounds по смене decision_uid), а НЕ на groupby (в ~30x быстрее на 393k
строк). Если когда-нибудь featurize начнёт шаффлить строки — эта функция молча сломается,
нужен .sort_values(['decision_uid','option_idx']) перед вызовом.

ЛОВУШКА lambdarank + group: LightGBM требует Dataset.set_group (сумма == число строк) —
group_sizes считается ИЗ ТЕХ ЖЕ contiguous_bounds на уже отфильтрованном (inner-train/val)
подмножестве, а не из исходных n_options колонки (которая описывает ПОЛНОЕ число опций
решения и совпадает с фактическим числом строк только если решение не потеряло ни одной
строки при фильтрации — фильтрация всегда удаляет решения ЦЕЛИКОМ, так что это тождество
держится, но если это когда-то перестанет быть так, group_sizes и Dataset разъедутся молча).

Запуск:
    .venv/bin/python scripts/train_bc_2b.py --smoke                 # быстрый смоук (маленькие бюджеты)
    .venv/bin/python scripts/train_bc_2b.py                         # полный прогон, все 6 групп
    .venv/bin/python scripts/train_bc_2b.py --groups main c7        # только эти группы
    .venv/bin/python scripts/train_bc_2b.py --objective lambdarank  # без сравнения, форсировать objective
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

WORKTREE = Path(__file__).resolve().parent.parent

# OptionType-константы (см. submissions/v076_final/main.py:365-423 / featurize_corpus_2b.py)
OPT_PLAY, OPT_ATTACH, OPT_ABILITY = 7, 8, 10
OPT_RETREAT, OPT_ATTACK, OPT_END = 12, 13, 14
CATEGORY_MAP = {
    OPT_ATTACK: "attack", OPT_ATTACH: "energy-attach", OPT_RETREAT: "retreat",
    OPT_PLAY: "play", OPT_ABILITY: "ability", OPT_END: "end",
}
GROUPS = ["main", "c7", "low", "mid", "easy", "rest"]

# Якорь для сравнения — out/corpus_2b/fidelity/v076_final_test.json (offline top-1 v076_final
# на том же held-out; headline_categories["attack"/"energy"/"retreat/positioning"] и
# overall_match_rate). attack сравним 1:1 (attack-тип опции существует ТОЛЬКО в MAIN-контексте
# у якоря тоже); energy/retreat у якоря блендят MAIN с другими select-контекстами (см.
# HEADLINE-маппинг в fidelity_expert_topk.py) — наши "energy-attach"/"retreat" в группе main
# УЖЕ, это подмножество, сравнение информативное, не строго 1:1 (см. отчёт).
ANCHOR = {"overall": 0.5906, "attack": 0.605, "energy_headline": 0.565,
          "retreat_positioning_headline": 0.729, "adrena_target_headline": 0.628,
          "other_headline": 0.565}


def category_of_type(t) -> str:
    try:
        ti = int(t)
    except (TypeError, ValueError):
        return "other"
    return CATEGORY_MAP.get(ti, "other")


def contiguous_bounds(decision_uid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Индексы (start,end) для каждого блока одинаковых decision_uid, ПРЕДПОЛАГАЯ, что
    строки одного решения идут подряд (см. докстринг модуля)."""
    n = len(decision_uid)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    changes = np.where(decision_uid[1:] != decision_uid[:-1])[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [n]))
    return starts, ends


def decision_level_eval(df: pd.DataFrame, scores: np.ndarray, with_category: bool) -> pd.DataFrame:
    """Одна строка на РЕШЕНИЕ: model_match / baseline_match(=всегда первые k опций) /
    chance_floor(=1/C(n,k)) / n_options / k [/ category, если with_category]."""
    uid = df["decision_uid"].to_numpy()
    opt_idx = df["option_idx"].to_numpy()
    y = df["y"].to_numpy()
    min_count = df["min_count"].to_numpy()
    max_count = df["max_count"].to_numpy() if "max_count" in df.columns else None
    opt_type = df["opt_type"].to_numpy() if with_category else None
    starts, ends = contiguous_bounds(uid)
    n_dec = len(starts)
    model_match = np.zeros(n_dec, dtype=bool)
    baseline_match = np.zeros(n_dec, dtype=bool)
    floor = np.full(n_dec, np.nan, dtype=float)
    n_opt_arr = np.zeros(n_dec, dtype=int)
    k_arr = np.zeros(n_dec, dtype=int)
    cats = [None] * n_dec if with_category else None
    uid_out = [None] * n_dec

    for j in range(n_dec):
        s, e = starts[j], ends[j]
        n_opt = e - s
        # ★ ПРАВКА 08-09 (оркестратор): k = minCount ошибочно при minCount==0.
        # Замер на train (50 506 решений с min>0 и 7 376 с min==0): при min>0 эксперт
        # берёт ровно min в 99.98% случаев, а при min==0 берёт МАКСИМУМ, min(max,n),
        # в 98.2% (c7: 0/1→1; low: 0/2→2; mid: 0/5→5). Со старым правилом k=min=0 наш
        # ответ — пустое множество, которое НЕ МОЖЕТ совпасть с непустым выбором
        # эксперта: 1810 из 14 434 test-решений (12.5%, а в группе c7 — 60.9%) были
        # гарантированными промахами метрики, а не ошибками модели.
        k = int(min_count[s])
        if k == 0 and max_count is not None:
            mx = int(max_count[s]) if max_count[s] == max_count[s] else 1
            k = mx if mx > 0 else 1
        k = max(0, min(k, n_opt))
        seg_opt = opt_idx[s:e]
        seg_y = y[s:e]
        true_idx = set(seg_opt[seg_y == 1].tolist())

        order_by_idx = np.argsort(seg_opt, kind="stable")
        base_idx = set(seg_opt[order_by_idx[:k]].tolist()) if k > 0 else set()

        if k > 0:
            seg_scores = scores[s:e]
            order_by_score = np.argsort(-seg_scores, kind="stable")  # ties -> исходный (opt_idx-возр.) порядок
            pred_idx = set(seg_opt[order_by_score[:k]].tolist())
        else:
            pred_idx = set()

        model_match[j] = (pred_idx == true_idx)
        baseline_match[j] = (base_idx == true_idx)
        # ЛОВУШКА (найдена на group=c7, где minCount!=maxCount в 60.7% решений): floor —
        # это P(match | случайный ЛЕГАЛЬНЫЙ набор размера k). Если истинный ответ эксперта
        # имеет ДРУГОЙ размер (len(true_idx) != k — бывает, когда minCount=0, а эксперт
        # реально выбрал >=1 опцию, что admissible при minCount!=maxCount), то ЛЮБОЙ набор
        # размера k, включая случайный, НЕ МОЖЕТ совпасть с true_idx по построению (разные
        # мощности множеств) — правильный floor тут 0.0, а не 1/C(n,k). Наивная формула
        # 1/C(n,0)=1.0 в этом случае наврала бы "случайный ответ угадывает в 100% случаев",
        # хотя структурно совпадение невозможно (проверено эмпирически: 60.9% решений c7 —
        # minCount=0, но y_sum>=1 всегда, ни одного y_sum==0 среди них).
        if len(true_idx) == k:
            try:
                floor[j] = 1.0 / math.comb(n_opt, k) if 0 <= k <= n_opt else 0.0
            except ValueError:
                floor[j] = 0.0
        else:
            floor[j] = 0.0
        n_opt_arr[j] = n_opt
        k_arr[j] = k
        uid_out[j] = uid[s]
        if with_category:
            true_rows = np.where(seg_y == 1)[0]
            cats[j] = category_of_type(opt_type[s + true_rows[0]]) if len(true_rows) else "other"

    out = pd.DataFrame({
        "decision_uid": uid_out, "n_options": n_opt_arr, "k": k_arr,
        "model_match": model_match, "baseline_match": baseline_match, "chance_floor": floor,
    })
    if with_category:
        out["category"] = cats
    return out


def summarize(dec: pd.DataFrame) -> dict:
    n = len(dec)
    if n == 0:
        return {"n_decisions": 0, "model_top1": None, "baseline_first_k_top1": None,
                "chance_floor_mean": None, "lift_vs_floor": None}
    model_top1 = float(dec["model_match"].mean())
    floor_mean = float(dec["chance_floor"].mean())
    return {
        "n_decisions": int(n),
        "model_top1": round(model_top1, 4),
        "baseline_first_k_top1": round(float(dec["baseline_match"].mean()), 4),
        "chance_floor_mean": round(floor_mean, 4),
        "lift_vs_floor": round(model_top1 - floor_mean, 4),
    }


def split_val_episodes(episode_ids: np.ndarray, val_frac: float, seed: int) -> set:
    """Val = целые эпизоды (не решения), сэмплированные ИЗ train-эпизодов ЭТОЙ группы.
    <=3 уникальных эпизодов -> вырожденный случай, val пуст (ранняя остановка отключается
    вызывающей стороной, см. train_lgb_model)."""
    uniq = np.unique(episode_ids)
    if len(uniq) <= 3:
        return set()
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    n_val = min(n_val, len(uniq) - 1)  # хотя бы 1 эпизод должен остаться в inner-train
    val_ids = rng.choice(uniq, size=n_val, replace=False)
    return set(val_ids.tolist())


# ── T1.4 п.1: HP-поиск по val (scripts/hp_search_2b.py, 2^4-факторный дизайн вокруг
# продакшн-точки, 17 обучений/группу) — см. out/corpus_2b/models/hp_search_report_{main,mid,c7}.json.
# main: val_top1 0.6947->0.7089 (+1.42пп), большие деревья (num_leaves x2, min_data x2) стабильно
#   лучше на ВСЕХ lr/ff соседях с nl=126 (0.6996-0.7089), а не только в одной угловой точке —
#   не шум одной комбинации.
# mid:  val_top1 0.7968->0.8092 (+1.24пп), тот же паттерн: nl=62 (x2 от базы) лучше nl=15 почти
#   во всех соседних lr/ff (0.8064-0.8092 vs 0.7924-0.8020).
# c7:   продакшн-точка УЖЕ лучшая во всей сетке (val_top1=0.6254) — 16/16 угловых комбинаций
#       проиграли базе, override не применяется (честный отрицательный результат поиска).
HYPERPARAM_OVERRIDES = {
    "main": dict(learning_rate=0.03, num_leaves=126, min_data_in_leaf=400, feature_fraction=0.7),
    "mid": dict(learning_rate=0.08, num_leaves=62, min_data_in_leaf=50, feature_fraction=0.7),
}


def pick_hyperparams(n_train_rows: int, group: str | None = None) -> dict:
    """num_leaves/min_data_in_leaf по размеру inner-train — маленькие группы получают
    маленькие деревья (защита от переобучения на 590 играх, см. докстринг модуля).
    `group` — если задан и есть в HYPERPARAM_OVERRIDES, num_leaves/min_data_in_leaf/
    learning_rate/feature_fraction берутся из HP-поиска по val (см. блок выше) вместо
    формулы по размеру; num_boost_round/early_stopping_rounds — формула не меняется (эти два
    поля не входили в запрошенную сетку HP-поиска)."""
    if n_train_rows >= 100_000:
        base = dict(num_leaves=63, min_data_in_leaf=200, num_boost_round=3000, early_stopping_rounds=80)
    elif n_train_rows >= 20_000:
        base = dict(num_leaves=31, min_data_in_leaf=100, num_boost_round=3000, early_stopping_rounds=80)
    elif n_train_rows >= 3_000:
        base = dict(num_leaves=15, min_data_in_leaf=30, num_boost_round=2000, early_stopping_rounds=60)
    else:
        base = dict(num_leaves=7, min_data_in_leaf=10, num_boost_round=800, early_stopping_rounds=30)
    if group is not None and group in HYPERPARAM_OVERRIDES:
        ov = HYPERPARAM_OVERRIDES[group]
        base = dict(base, num_leaves=ov["num_leaves"], min_data_in_leaf=ov["min_data_in_leaf"])
        base["learning_rate"] = ov["learning_rate"]
        base["feature_fraction"] = ov["feature_fraction"]
    return base


def build_common_params(hp: dict, n_jobs: int, seed: int) -> dict:
    return dict(num_leaves=hp["num_leaves"], min_data_in_leaf=hp["min_data_in_leaf"],
                learning_rate=hp.get("learning_rate", 0.05), feature_fraction=hp.get("feature_fraction", 0.8),
                bagging_fraction=0.8, bagging_freq=5,
                lambda_l2=1.0, num_threads=n_jobs, verbose=-1, seed=seed, deterministic=True,
                force_row_wise=True)


def train_lgb_model(objective: str, X_tr: np.ndarray, y_tr: np.ndarray,
                     X_val: np.ndarray | None, y_val: np.ndarray | None,
                     group_tr: np.ndarray | None, group_val: np.ndarray | None,
                     hp: dict, n_jobs: int, seed: int,
                     weight_tr: np.ndarray | None = None,
                     weight_val: np.ndarray | None = None) -> tuple[lgb.Booster, dict]:
    """weight_tr/weight_val — опциональные per-row веса сэмплов ([T2-B, вариант (б)]
    zеркальная специализация со взвешиванием в пользу сильных зеркальных экспертов Peel/
    Taichicchi — см. sample_weights_for). None -> все веса 1.0 (поведение не меняется для
    обычного не-специализированного обучения)."""
    params = build_common_params(hp, n_jobs, seed)
    has_val = X_val is not None and len(X_val) > 0

    if objective == "binary":
        params.update(objective="binary", metric="auc", is_unbalance=True)
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=weight_tr, free_raw_data=False)
    elif objective == "lambdarank":
        params.update(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1], label_gain=[0, 1])
        dtrain = lgb.Dataset(X_tr, label=y_tr, group=group_tr, weight=weight_tr, free_raw_data=False)
    else:
        raise ValueError(f"unknown objective: {objective}")

    callbacks = [lgb.log_evaluation(period=0)]
    valid_sets, valid_names = None, None
    if has_val:
        if objective == "binary":
            dval = lgb.Dataset(X_val, label=y_val, weight=weight_val, reference=dtrain, free_raw_data=False)
        else:
            dval = lgb.Dataset(X_val, label=y_val, group=group_val, weight=weight_val,
                                reference=dtrain, free_raw_data=False)
        valid_sets, valid_names = [dval], ["val"]
        callbacks.append(lgb.early_stopping(hp["early_stopping_rounds"], verbose=False))
        num_round = hp["num_boost_round"]
    else:
        # вырожденный val (см. split_val_episodes) -> без ранней остановки, урезанный бюджет
        # раундов вместо неё (best-effort защита от переобучения на крошечной группе)
        num_round = max(50, hp["num_boost_round"] // 4)

    booster = lgb.train(params, dtrain, num_boost_round=num_round,
                         valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks)
    info = {
        "objective": objective, "num_boost_round_cap": num_round,
        "best_iteration": int(booster.best_iteration) if has_val and booster.best_iteration else int(booster.current_iteration()),
        "used_early_stopping": bool(has_val),
        "num_trees": int(booster.num_trees()),
    }
    return booster, info


def load_group_df(features_dir: Path, group: str, feature_order: list) -> pd.DataFrame:
    path = features_dir / f"{group}.parquet"
    df = pd.read_parquet(path)
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise RuntimeError(f"{group}: отсутствуют X-колонки из feature_spec.json: {missing[:5]}...")
    return df


# ── [T2-B] Зеркальная специализация ──────────────────────────────────────────
# Задача: submissions/v080_bc2b/main.py::_bc_decide сейчас 0.160 top-1 в зеркале
# (mirror_grimmsnarl) против 0.328 у старого live-прода (v076_final) — см. отчёт приёмки.
# opp_archetype/expert/game_won — метаколонки, пишутся featurize_corpus_2b.py 1:1 из
# decisions.jsonl (истина леджера), присутствуют во ВСЕХ 6 groups.parquet.
MIRROR_ARCHETYPE = "mirror_grimmsnarl"
HIGH_WR_MIRROR_EXPERTS = ("Dominic Peel", "__Taichicchi__")  # зеркальный WR 0.708/0.672 против
                                                              # 0.533 (Luca) / 0.460 (snieg) —
                                                              # см. project docstring/задание


def apply_subcorpus_filter(df: pd.DataFrame, subcorpus: str, variant: str) -> pd.DataFrame:
    """subcorpus='mirror' -> оставить ТОЛЬКО решения из игр против mirror_grimmsnarl (истина
    opp_archetype, не наш live-детект). Фильтрация выполняется ДО make_masks/split_val_episodes,
    так что train/val/test-границы по эпизодам (split.json) не меняются — просто внутри каждой
    границы остаются только строки нужного подкорпуса (нет утечки эпизодов между train/test,
    тот же контракт, что и у общего пайплайна).

    ВАЖНО: `variant='wins_only'` (вариант (в) задания) здесь НЕ применяется -- он фильтрует
    только TRAIN/VAL (см. apply_train_variant_filter ниже), чтобы TEST оставался ОДНИМ И ТЕМ ЖЕ
    зеркальным held-out для всех трёх вариантов (а/б/в) и сравнение top-1 между ними было
    честным (иначе wins_only тестировался бы на другом, более лёгком подмножестве, и разница
    в top-1 была бы частично артефактом другого теста, а не другой модели)."""
    if subcorpus == "none":
        return df
    if subcorpus != "mirror":
        raise ValueError(f"unknown subcorpus: {subcorpus}")
    return df[df["opp_archetype"] == MIRROR_ARCHETYPE]


def apply_train_variant_filter(is_tr: np.ndarray, is_val: np.ndarray, df: pd.DataFrame,
                                variant: str) -> tuple[np.ndarray, np.ndarray]:
    """variant='wins_only' -> дополнительно ограничивает TRAIN/VAL масками game_won==True
    (вариант (в) задания). TEST не трогается (см. apply_subcorpus_filter докстринг)."""
    if variant != "wins_only":
        return is_tr, is_val
    won = (df["game_won"] == True).to_numpy()  # noqa: E712
    return is_tr & won, is_val & won


def sample_weights_for(df: pd.DataFrame, mask: np.ndarray, variant: str, high_weight: float) -> np.ndarray | None:
    """variant='expert_weight' -> вариант (б) задания: строки экспертов Peel/Taichicchi
    (зеркальный WR 0.708/0.672) получают вес `high_weight`, Luca/snieg (0.533/0.460) — вес
    1.0. Иначе (variant 'all'/'wins_only'/не-mirror запуск) -> None (равные веса, поведение
    train_lgb_model не меняется)."""
    if variant != "expert_weight":
        return None
    expert = df.loc[mask, "expert"].to_numpy()
    return np.where(np.isin(expert, HIGH_WR_MIRROR_EXPERTS), high_weight, 1.0).astype(np.float64)


def retreat_weight_for(df: pd.DataFrame, mask: np.ndarray, weight: float) -> np.ndarray | None:
    """★ Плюс-задание 08-09 (minority-class collapse у retreat, см. докстринг модуля §H1
    гипотеза-предшественник): вес `weight` для ВСЕХ строк (и y=0, и y=1) решений, чья ИСТИННАЯ
    категория — retreat (эксперт выбрал OPT_RETREAT), остальные решения — вес 1.0.
    weight==1.0 -> None (без изменений, тот же контракт, что sample_weights_for — no-op не
    подаёт lgb.Dataset(weight=...) вовсе). Взвешивание ВСЕГО решения (обеих меток), а не
    только положительной строки — стандартный приём для lambdarank: увеличивает вклад ЦЕЛОГО
    ранжирующего списка в градиент (NDCG считается по списку), не одной опции внутри него.
    Работает вне зависимости от группы (retreat встречается только в group=='main' — на
    остальных группах retreat_uids пуст, функция безвредно возвращает все веса 1.0)."""
    if weight == 1.0:
        return None
    n = int(mask.sum())
    if n == 0:
        return None
    sub = df.loc[mask, ["decision_uid", "y", "opt_type"]]
    true_rows = sub[sub["y"] == 1]
    true_type = true_rows.groupby("decision_uid")["opt_type"].first()
    retreat_uids = set(true_type.index[true_type == OPT_RETREAT])
    return np.where(sub["decision_uid"].isin(retreat_uids).to_numpy(), weight, 1.0).astype(np.float64)


def _combine_weights(a: np.ndarray | None, b: np.ndarray | None, n: int) -> np.ndarray | None:
    """a*b поэлементно; None трактуется как все-единицы; оба None -> None (сохраняет контракт
    "None = равные веса, train_lgb_model не передаёт weight в lgb.Dataset")."""
    if a is None and b is None:
        return None
    a = a if a is not None else np.ones(n, dtype=np.float64)
    b = b if b is not None else np.ones(n, dtype=np.float64)
    return a * b


def load_general_booster(models_dir: Path, group: str) -> lgb.Booster | None:
    """Уже обученная НЕ-специализированная модель группы (полный корпус, out/corpus_2b/
    models/<group>.txt по умолчанию) — нужна для сравнения 'выигрывает ли специализация'
    на ТОМ ЖЕ held-out подмножестве (задание §1: 'обязательно приведи и результат общей
    модели на том же подмножестве, иначе непонятно, есть ли вообще выигрыш')."""
    path = models_dir / f"{group}.txt"
    if not path.exists():
        return None
    return lgb.Booster(model_file=str(path))


def stable_str_hash(s: str) -> int:
    """★ БАГ-ФИКС 08-09 (T1.4 HP-поиск): builtin hash() на строках РАНДОМИЗИРОВАН по
    процессам в CPython (PYTHONHASHSEED не зафиксирован в окружении проекта, проверено
    эмпирически: hash('rest')%10000 давал 4529/5697/5334 в трёх отдельных запусках)
    — значит make_masks(..., seed=42, ...) НЕ был воспроизводим между запусками, вопреки
    докстрингу модуля ("seed=42, deterministic=True"): val_ep_set менялся от запуска к
    запуску -> разный inner-train/val -> разная точка ранней остановки -> разная финальная
    модель на ТЕХ ЖЕ гиперпараметрах и test (test сам по себе не зависит от этого бага —
    test-эпизоды фиксированы в split.json на этапе featurize, а не здесь). zlib.crc32
    детерминирован для одной и той же строки на любой платформе/процессе/версии Python."""
    return zlib.crc32(s.encode("utf-8"))


def make_masks(df: pd.DataFrame, val_frac: float, seed: int, group_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    split = df["split"].to_numpy()
    ep = df["episode_id"].to_numpy()
    is_test = split == "test"
    is_train_all = split == "train"
    train_ep = ep[is_train_all]
    val_ep_set = split_val_episodes(train_ep, val_frac, seed=seed + stable_str_hash(group_name) % 10_000)
    if val_ep_set:
        is_val = is_train_all & np.isin(ep, list(val_ep_set))
    else:
        is_val = np.zeros(len(df), dtype=bool)
    is_inner_train = is_train_all & ~is_val
    meta = {
        "n_train_episodes_total": int(len(np.unique(train_ep))),
        "n_val_episodes": int(len(val_ep_set)),
        "degenerate_val": bool(not val_ep_set),
    }
    return is_inner_train, is_val, is_test, meta


def xy(df: pd.DataFrame, mask: np.ndarray, feature_order: list) -> tuple[np.ndarray, np.ndarray]:
    sub = df.loc[mask]
    X = sub[feature_order].to_numpy(dtype=np.float32)
    y = sub["y"].to_numpy(dtype=np.int32)
    return X, y


def group_sizes_for(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    uid = df.loc[mask, "decision_uid"].to_numpy()
    starts, ends = contiguous_bounds(uid)
    return (ends - starts).astype(np.int32)


def compare_objectives(df_main: pd.DataFrame, feature_order: list, val_frac: float,
                        seed: int, n_jobs: int) -> dict:
    is_tr, is_val, _is_test, meta = make_masks(df_main, val_frac, seed, "main")
    hp = pick_hyperparams(int(is_tr.sum()), group="main")
    X_tr, y_tr = xy(df_main, is_tr, feature_order)
    X_val, y_val = xy(df_main, is_val, feature_order) if is_val.any() else (None, None)
    g_tr = group_sizes_for(df_main, is_tr)
    g_val = group_sizes_for(df_main, is_val) if is_val.any() else None

    dec_val_meta = df_main.loc[is_val, ["decision_uid", "option_idx", "y", "min_count", "max_count", "opt_type"]] if is_val.any() else None

    results = {}
    for obj in ("binary", "lambdarank"):
        t0 = time.perf_counter()
        booster, info = train_lgb_model(obj, X_tr, y_tr, X_val, y_val, g_tr, g_val, hp, n_jobs, seed)
        wall = round(time.perf_counter() - t0, 1)
        if X_val is not None and len(X_val):
            scores_val = booster.predict(X_val, num_iteration=booster.best_iteration or None)
            dec_val = decision_level_eval(dec_val_meta, scores_val, with_category=False)
            val_summary = summarize(dec_val)
        else:
            val_summary = None
        results[obj] = {**info, "wall_sec": wall, "val_top1": val_summary}
        print(f"[compare] main/{obj}: best_iter={info['best_iteration']} "
              f"val_top1={val_summary['model_top1'] if val_summary else 'N/A'} "
              f"(floor={val_summary['chance_floor_mean'] if val_summary else 'N/A'}) "
              f"wall={wall}s", file=sys.stderr)

    b_top1 = (results["binary"]["val_top1"] or {}).get("model_top1")
    l_top1 = (results["lambdarank"]["val_top1"] or {}).get("model_top1")
    if b_top1 is None and l_top1 is None:
        chosen = "lambdarank"  # без val-сигнала — по умолчанию ранкер (нативно top-k-ориентирован)
        rationale = "val пуст на group=main (не должно случаться, main имеет 590 train эпизодов) — дефолт lambdarank"
    elif l_top1 is None:
        chosen = "binary"
        rationale = "lambdarank не дал val-метрику"
    elif b_top1 is None:
        chosen = "lambdarank"
        rationale = "binary не дал val-метрику"
    else:
        chosen = "lambdarank" if l_top1 >= b_top1 else "binary"
        rationale = f"val top-1 на group=main: binary={b_top1:.4f} vs lambdarank={l_top1:.4f}"

    return {"candidates": results, "chosen": chosen, "rationale": rationale,
            "val_meta": meta, "hyperparams_used": hp}


def train_and_eval_group(group: str, features_dir: Path, feature_order: list, objective: str,
                          val_frac: float, seed: int, n_jobs: int, out_dir: Path,
                          subcorpus: str = "none", variant: str = "all", high_weight: float = 3.0,
                          out_suffix: str = "", general_models_dir: Path | None = None,
                          retreat_weight: float = 1.0) -> dict:
    df_full = load_group_df(features_dir, group, feature_order)
    df = apply_subcorpus_filter(df_full, subcorpus, variant)
    is_tr, is_val, is_test, meta = make_masks(df, val_frac, seed, group)
    # [T2-B, вариант (в)] wins_only фильтрует ТОЛЬКО train/val, is_test не трогается -- один и
    # тот же held-out для а/б/в, см. apply_train_variant_filter докстринг.
    n_tr_before_variant = int(is_tr.sum())
    is_tr, is_val = apply_train_variant_filter(is_tr, is_val, df, variant)
    hp = pick_hyperparams(int(is_tr.sum()), group=group)

    X_tr, y_tr = xy(df, is_tr, feature_order)
    X_val, y_val = xy(df, is_val, feature_order) if is_val.any() else (None, None)
    g_tr = group_sizes_for(df, is_tr) if objective == "lambdarank" else None
    g_val = group_sizes_for(df, is_val) if (objective == "lambdarank" and is_val.any()) else None
    w_tr = sample_weights_for(df, is_tr, variant, high_weight)
    w_val = sample_weights_for(df, is_val, variant, high_weight) if is_val.any() else None
    # ★ Плюс-задание 08-09 (retreat minority-class collapse) — см. retreat_weight_for
    # докстринг; комбинируется с sample_weights_for (T2-B) поэлементным умножением, если оба
    # активны (в текущих прогонах T2-B не используется одновременно, но контракт общий).
    rw_tr = retreat_weight_for(df, is_tr, retreat_weight)
    rw_val = retreat_weight_for(df, is_val, retreat_weight) if is_val.any() else None
    w_tr = _combine_weights(w_tr, rw_tr, int(is_tr.sum()))
    w_val = _combine_weights(w_val, rw_val, int(is_val.sum())) if is_val.any() else None

    t0 = time.perf_counter()
    booster, info = train_lgb_model(objective, X_tr, y_tr, X_val, y_val, g_tr, g_val, hp, n_jobs, seed,
                                     weight_tr=w_tr, weight_val=w_val)
    wall = round(time.perf_counter() - t0, 1)

    model_path = out_dir / f"{group}{out_suffix}.txt"
    booster.save_model(str(model_path))
    model_bytes = model_path.stat().st_size

    X_test, _y_test = xy(df, is_test, feature_order)
    test_meta = df.loc[is_test, ["decision_uid", "option_idx", "y", "min_count", "max_count", "opt_type"]]
    scores_test = booster.predict(X_test, num_iteration=booster.best_iteration or None)
    dec_test = decision_level_eval(test_meta, scores_test, with_category=(group == "main"))
    test_summary = summarize(dec_test)

    by_category = None
    if group == "main":
        by_category = {}
        for cat, sub in dec_test.groupby("category"):
            by_category[cat] = summarize(sub)

    result = {
        "objective": objective, "hyperparams": hp, "wall_sec": wall, "model_bytes": model_bytes,
        "model_path": str(model_path),
        "subcorpus": subcorpus, "variant": variant,
        "n_inner_train_rows_before_variant_filter": n_tr_before_variant,
        "n_inner_train_rows": int(is_tr.sum()), "n_val_rows": int(is_val.sum()),
        "n_test_rows": int(is_test.sum()),
        "n_inner_train_decisions": int(df.loc[is_tr, "decision_uid"].nunique()),
        "n_val_decisions": int(df.loc[is_val, "decision_uid"].nunique()) if is_val.any() else 0,
        "n_test_decisions": int(df.loc[is_test, "decision_uid"].nunique()),
        "val_meta": meta, **info,
        "test": test_summary,
    }
    if by_category is not None:
        result["test_by_category"] = by_category

    # [T2-B задание §1] "обязательно приведи и результат общей (не специализированной) модели
    # на том же подмножестве" — общая модель = уже обученная НЕ-специализированная (полный
    # корпус) booster группы, оценённая на ТОМ ЖЕ X_test/test_meta (тот же held-out подкорпус,
    # та же decision_level_eval), иначе нет базы для сравнения "выигрывает ли специализация".
    if subcorpus != "none" and general_models_dir is not None:
        gen_booster = load_general_booster(general_models_dir, group)
        if gen_booster is not None:
            gen_scores = gen_booster.predict(X_test, num_iteration=gen_booster.best_iteration or None)
            gen_dec = decision_level_eval(test_meta, gen_scores, with_category=False)
            result["general_model_on_subcorpus_test"] = {
                **summarize(gen_dec), "model_path": str(general_models_dir / f"{group}.txt"),
            }
        else:
            result["general_model_on_subcorpus_test"] = None
            print(f"[train] {group}: general booster не найден в {general_models_dir} — сравнение пропущено",
                  file=sys.stderr)

    gen_line = ""
    if "general_model_on_subcorpus_test" in result and result["general_model_on_subcorpus_test"]:
        g = result["general_model_on_subcorpus_test"]
        gen_line = f" general_on_same_subset={g['model_top1']}"
    print(f"[train] {group}{out_suffix}: obj={objective} best_iter={info['best_iteration']} "
          f"test_top1={test_summary['model_top1']} baseline={test_summary['baseline_first_k_top1']} "
          f"floor={test_summary['chance_floor_mean']} n_test_dec={test_summary['n_decisions']}"
          f"{gen_line} wall={wall}s -> {model_path.name} ({model_bytes} bytes)", file=sys.stderr)
    return result, dec_test[["model_match", "baseline_match", "chance_floor"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", default=str(WORKTREE / "out" / "corpus_2b" / "features"))
    ap.add_argument("--out-dir", default=str(WORKTREE / "out" / "corpus_2b" / "models"))
    ap.add_argument("--groups", nargs="+", default=GROUPS, choices=GROUPS)
    ap.add_argument("--objective", default="auto", choices=["auto", "binary", "lambdarank"],
                     help="'auto' = сравнить оба на group=main (val), победитель -> все группы")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="урезанные бюджеты раундов для быстрой проверки кода")
    # [T2-B] Зеркальная специализация (см. apply_subcorpus_filter/sample_weights_for докстринги).
    ap.add_argument("--subcorpus", default="none", choices=["none", "mirror"],
                     help="'mirror' -> обучать/мерить ТОЛЬКО на решениях из игр против "
                          "opp_archetype==mirror_grimmsnarl (истина леджера)")
    ap.add_argument("--variant", default="all", choices=["all", "expert_weight", "wins_only"],
                     help="применимо только при --subcorpus mirror: 'all'=(а) все зеркальные "
                          "решения равным весом; 'expert_weight'=(б) Peel/Taichicchi весом "
                          "--high-weight; 'wins_only'=(в) только решения из ВЫИГРАННЫХ "
                          "зеркальных партий (game_won==True)")
    ap.add_argument("--high-weight", type=float, default=3.0,
                     help="вес строк экспертов Peel/Taichicchi при --variant expert_weight")
    ap.add_argument("--out-suffix", default="",
                     help="суффикс имени файла модели: <group><suffix>.txt (напр. _mirror_b)")
    ap.add_argument("--general-models-dir", default=str(WORKTREE / "out" / "corpus_2b" / "models"),
                     help="где искать уже обученные НЕ-специализированные модели для сравнения "
                          "на том же --subcorpus held-out (см. train_and_eval_group)")
    ap.add_argument("--retreat-weight", type=float, default=1.0,
                     help="★ Плюс-задание 08-09: вес строк решений с истинной категорией "
                          "retreat (minority-class collapse, см. retreat_weight_for). "
                          "1.0 = без изменений (по умолчанию).")
    args = ap.parse_args()

    if args.n_jobs > 6:
        print(f"[train_bc_2b] WARNING: n_jobs={args.n_jobs} > 6, ограничение проекта — форсирую 6", file=sys.stderr)
        args.n_jobs = 6

    t_start = time.perf_counter()
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = json.loads((features_dir / "feature_spec.json").read_text())
    feature_order = spec["feature_order"]
    print(f"[train_bc_2b] {len(feature_order)} X-фич, группы={args.groups}, objective={args.objective}",
          file=sys.stderr)

    global pick_hyperparams
    if args.smoke:
        _orig = pick_hyperparams

        def pick_hyperparams(n, group=None):  # noqa: F811
            hp = _orig(n, group=group)
            hp["num_boost_round"] = min(hp["num_boost_round"], 100)
            hp["early_stopping_rounds"] = min(hp["early_stopping_rounds"], 15)
            return hp

    objective_selection = None
    if args.objective == "auto":
        print("[train_bc_2b] --objective auto: сравниваю binary vs lambdarank на group=main (val)...",
              file=sys.stderr)
        df_main = apply_subcorpus_filter(load_group_df(features_dir, "main", feature_order),
                                          args.subcorpus, args.variant)
        objective_selection = compare_objectives(df_main, feature_order, args.val_frac, args.seed, args.n_jobs)
        chosen_objective = objective_selection["chosen"]
        print(f"[train_bc_2b] выбран objective={chosen_objective} ({objective_selection['rationale']})",
              file=sys.stderr)
        del df_main
    else:
        chosen_objective = args.objective

    if args.subcorpus != "none":
        print(f"[train_bc_2b] [T2-B] subcorpus={args.subcorpus} variant={args.variant} "
              f"high_weight={args.high_weight} out_suffix='{args.out_suffix}' "
              f"general_models_dir={args.general_models_dir}", file=sys.stderr)

    group_results = {}
    pooled_match = pooled_base_match = pooled_floor_sum = pooled_n = 0
    pooled_gen_match = 0
    for group in args.groups:
        res, dec_flags = train_and_eval_group(
            group, features_dir, feature_order, chosen_objective,
            args.val_frac, args.seed, args.n_jobs, out_dir,
            subcorpus=args.subcorpus, variant=args.variant, high_weight=args.high_weight,
            out_suffix=args.out_suffix, general_models_dir=Path(args.general_models_dir),
            retreat_weight=args.retreat_weight,
        )
        group_results[group] = res
        pooled_match += int(dec_flags["model_match"].sum())
        pooled_base_match += int(dec_flags["baseline_match"].sum())
        pooled_floor_sum += float(dec_flags["chance_floor"].sum())
        pooled_n += len(dec_flags)

    overall_test = {
        "n_decisions": pooled_n,
        "model_top1": round(pooled_match / pooled_n, 4) if pooled_n else None,
        "baseline_first_k_top1": round(pooled_base_match / pooled_n, 4) if pooled_n else None,
        "chance_floor_mean": round(pooled_floor_sum / pooled_n, 4) if pooled_n else None,
    }
    if overall_test["model_top1"] is not None and overall_test["chance_floor_mean"] is not None:
        overall_test["lift_vs_floor"] = round(overall_test["model_top1"] - overall_test["chance_floor_mean"], 4)

    # [T2-B] Пул "общей (не-специализированной) модели на том же подмножестве" по всем группам
    # -- прямое сравнение с overall_test (специализированная модель) на идентичном held-out.
    overall_general_on_subcorpus = None
    if args.subcorpus != "none":
        g_match_sum = g_n_sum = 0
        for group, res in group_results.items():
            gcell = res.get("general_model_on_subcorpus_test")
            if gcell is None or gcell.get("model_top1") is None:
                continue
            n = gcell["n_decisions"]
            g_match_sum += gcell["model_top1"] * n
            g_n_sum += n
        if g_n_sum:
            overall_general_on_subcorpus = {
                "n_decisions": g_n_sum,
                "model_top1": round(g_match_sum / g_n_sum, 4),
            }

    attack_ours = None
    if "main" in group_results and "test_by_category" in group_results["main"]:
        attack_cell = group_results["main"]["test_by_category"].get("attack")
        attack_ours = attack_cell["model_top1"] if attack_cell else None

    anchor_cmp = {
        "overall_anchor": ANCHOR["overall"], "overall_ours": overall_test["model_top1"],
        "overall_delta": (round(overall_test["model_top1"] - ANCHOR["overall"], 4)
                           if overall_test["model_top1"] is not None else None),
        "overall_beats_anchor": (overall_test["model_top1"] > ANCHOR["overall"]
                                  if overall_test["model_top1"] is not None else None),
        "attack_anchor": ANCHOR["attack"], "attack_ours": attack_ours,
        "attack_delta": (round(attack_ours - ANCHOR["attack"], 4) if attack_ours is not None else None),
        "attack_beats_anchor": (attack_ours > ANCHOR["attack"] if attack_ours is not None else None),
        "note": ("overall/attack сравнимы 1:1 с якорём (attack-опция существует только в MAIN-"
                 "контексте у обеих сторон); energy-attach/retreat/ability в test_by_category "
                 "группы main — ПОДМНОЖЕСТВО якорных headline-категорий 'energy'/'retreat/"
                 "positioning'/'adrena-target' (те блендят несколько select-контекстов, не только "
                 "MAIN, см. HEADLINE в fidelity_expert_topk.py) — сравнение информативное, не 1:1."),
    }

    metrics = {
        "feature_spec_n_features": len(feature_order),
        "objective_selection": objective_selection,
        "chosen_objective": chosen_objective,
        "groups": group_results,
        "overall_test": overall_test,
        "overall_general_on_subcorpus_test": overall_general_on_subcorpus,
        "anchor": ANCHOR,
        "anchor_comparison": anchor_cmp,
        "run_meta": {
            "seed": args.seed, "val_frac": args.val_frac, "n_jobs": args.n_jobs,
            "smoke": args.smoke, "groups_run": args.groups,
            "subcorpus": args.subcorpus, "variant": args.variant, "high_weight": args.high_weight,
            "retreat_weight": args.retreat_weight,
            "out_suffix": args.out_suffix, "general_models_dir": args.general_models_dir,
            "lightgbm_version": lgb.__version__,
            "wall_sec": round(time.perf_counter() - t_start, 1),
        },
    }
    metrics_name = f"metrics{args.out_suffix}.json" if args.out_suffix else "metrics.json"
    (out_dir / metrics_name).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({"overall_test": overall_test,
                       "overall_general_on_subcorpus_test": overall_general_on_subcorpus,
                       "anchor_comparison": anchor_cmp}, indent=2, ensure_ascii=False))
    print(f"\n[train_bc_2b] done in {metrics['run_meta']['wall_sec']}s -> {out_dir / metrics_name}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
