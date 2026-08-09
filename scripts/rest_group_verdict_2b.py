#!/usr/bin/env python3
"""rest_group_verdict_2b.py — T1.4 п.2: стоит ли держать модель для группы 'rest' (контексты
вне схемы роутинга) или в live честнее воздерживаться (текущее поведение _bc_decide).

НЕ трогает test (уже использован один раз в train_bc_2b.py финальном прогоне) — весь замер
идёт episode-level K-fold CV на train+val группы 'rest' (77 эпизодов, 124 решения — см.
train_bc_2b.py metrics.json n_inner_train_decisions=107 + n_val_decisions=17). Единственный
способ получить статистически осмысленный сигнал на такой крошечной группе: усреднить
model_top1 vs baseline_first_k_top1 по НЕСКОЛЬКИМ фолдам вместо одного маленького held-out
(test даёт всего 33 решения — высокая дисперсия одной оценки, отсюда и вопрос задания).

Схема фолда идентична train_bc_2b.py: разбиение ПО ЭПИЗОДАМ (не по решениям), lambdarank,
гиперпараметры = pick_hyperparams(n_train_rows) той же формулы (сейчас это самый маленький
бакет: num_leaves=7, min_data_in_leaf=10 при <3000 строк).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "scripts"))
import train_bc_2b as tb  # noqa: E402


def kfold_episode_splits(episode_ids: np.ndarray, k: int, seed: int) -> list[tuple[set, set]]:
    """k непересекающихся фолдов по уникальным эпизодам -> список (train_eps, val_eps)."""
    uniq = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    folds = np.array_split(perm, k)
    out = []
    for i in range(k):
        val_eps = set(folds[i].tolist())
        train_eps = set(uniq.tolist()) - val_eps
        out.append((train_eps, val_eps))
    return out


def main() -> int:
    features_dir = WORKTREE / "out" / "corpus_2b" / "features"
    spec = json.loads((features_dir / "feature_spec.json").read_text())
    feature_order = spec["feature_order"]

    df_full = tb.load_group_df(features_dir, "rest", feature_order)
    # train+val ТОЛЬКО (test исключается полностью -- задание явно просит train+val)
    df = df_full[df_full["split"] == "train"].reset_index(drop=True)
    ep = df["episode_id"].to_numpy()
    n_eps = len(np.unique(ep))
    n_dec_total = df["decision_uid"].nunique()
    print(f"[rest_verdict] group=rest train-split: {len(df)} rows, {n_dec_total} decisions, {n_eps} episodes",
          file=sys.stderr)

    K = 5
    seed = 42
    folds = kfold_episode_splits(ep, K, seed)

    fold_results = []
    for i, (train_eps, val_eps) in enumerate(folds):
        is_tr = np.isin(ep, list(train_eps))
        is_val = np.isin(ep, list(val_eps))
        n_tr_rows = int(is_tr.sum())
        if n_tr_rows == 0 or is_val.sum() == 0:
            continue
        hp = tb.pick_hyperparams(n_tr_rows)
        X_tr, y_tr = tb.xy(df, is_tr, feature_order)
        X_val, y_val = tb.xy(df, is_val, feature_order)
        g_tr = tb.group_sizes_for(df, is_tr)
        g_val = tb.group_sizes_for(df, is_val)
        dec_val_meta = df.loc[is_val, ["decision_uid", "option_idx", "y", "min_count", "max_count", "opt_type"]]

        t0 = time.perf_counter()
        # без early stopping (val фолда тут же служит и eval-сетом -- держим бюджет урезанным,
        # как train_bc_2b делает при вырожденном val: num_boost_round//4)
        booster, info = tb.train_lgb_model("lambdarank", X_tr, y_tr, None, None, g_tr, None,
                                            hp, n_jobs=4, seed=seed)
        wall = round(time.perf_counter() - t0, 1)
        scores_val = booster.predict(X_val)
        dec_val = tb.decision_level_eval(dec_val_meta, scores_val, with_category=False)
        summary = tb.summarize(dec_val)
        fold_results.append({
            "fold": i, "n_train_rows": n_tr_rows, "n_val_rows": int(is_val.sum()),
            "n_val_decisions": summary["n_decisions"], "model_top1": summary["model_top1"],
            "baseline_first_k_top1": summary["baseline_first_k_top1"],
            "chance_floor_mean": summary["chance_floor_mean"], "wall_sec": wall,
        })
        print(f"  fold {i}: n_val_dec={summary['n_decisions']} model_top1={summary['model_top1']} "
              f"baseline={summary['baseline_first_k_top1']} floor={summary['chance_floor_mean']} "
              f"wall={wall}s", file=sys.stderr)

    n_dec_sum = sum(r["n_val_decisions"] for r in fold_results)
    model_match_sum = sum(r["model_top1"] * r["n_val_decisions"] for r in fold_results)
    base_match_sum = sum(r["baseline_first_k_top1"] * r["n_val_decisions"] for r in fold_results)
    floor_sum = sum(r["chance_floor_mean"] * r["n_val_decisions"] for r in fold_results)

    pooled_model_top1 = model_match_sum / n_dec_sum if n_dec_sum else None
    pooled_baseline_top1 = base_match_sum / n_dec_sum if n_dec_sum else None
    pooled_floor = floor_sum / n_dec_sum if n_dec_sum else None

    # доля фолдов, где модель ПРОИГРЫВАЕТ тривиальному бейзлайну -- прямой сигнал для решения
    n_folds_model_worse = sum(1 for r in fold_results if r["model_top1"] < r["baseline_first_k_top1"])

    report = {
        "group": "rest", "k_folds": K, "seed": seed,
        "n_episodes_train_split": n_eps, "n_decisions_train_split": n_dec_total,
        "fold_results": fold_results,
        "pooled_model_top1": round(pooled_model_top1, 4) if pooled_model_top1 is not None else None,
        "pooled_baseline_first_k_top1": round(pooled_baseline_top1, 4) if pooled_baseline_top1 is not None else None,
        "pooled_chance_floor_mean": round(pooled_floor, 4) if pooled_floor is not None else None,
        "n_folds_model_worse_than_baseline": n_folds_model_worse,
        "n_folds_total": len(fold_results),
        # для контекста -- уже известный (использованный ОДИН раз) test-результат из
        # models/metrics.json, НЕ участвует в решении, только для полноты картины в отчёте
        "reference_test_result_already_spent": {"model_top1": 0.5758, "baseline_first_k_top1": 0.7273, "n_test_decisions": 33},
    }
    verdict = "ABSTAIN" if (pooled_model_top1 is None or pooled_baseline_top1 is None
                             or pooled_model_top1 <= pooled_baseline_top1 or n_folds_model_worse >= K // 2 + 1) else "USE_MODEL"
    report["verdict"] = verdict

    out_path = WORKTREE / "out" / "corpus_2b" / "models" / "rest_group_verdict.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[rest_verdict] VERDICT={verdict} pooled_model_top1={pooled_model_top1} "
          f"pooled_baseline={pooled_baseline_top1} worse_in={n_folds_model_worse}/{len(fold_results)} folds "
          f"-> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
