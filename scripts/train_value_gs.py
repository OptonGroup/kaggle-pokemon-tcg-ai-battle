#!/usr/bin/env python3
"""train_value_gs.py -- обучает LightGBM binary value-модель на out/value_gs/dataset.npz
(PLAN_v084_SEARCH.md §3ter, пункт 2 "Лист = обученная value-модель").

Вход: out/value_gs/dataset.npz -- X (N,95) float32, y (N,) float32 {0,1}, episode_id,
agent_index, turn_idx, deck_key, opp_archetype, is_test (bool, held-out ПО ЭПИЗОДАМ, см.
предыдущий отчёт: zlib.crc32(episode_id) % 5 == 0 -> test).

Test (is_test=True) остаётся ПОЛНОСТЬЮ held-out до финальной оценки -- для early stopping
нужен отдельный val-сплит, вырезанный ИЗ train (тоже по эпизодам, не по строкам, иначе решения
одной игры утекут между train/val).

Выход:
  out/value_gs/model.txt        -- LightGBM booster (text format, booster.save_model)
  out/value_gs/train_report.json -- метрики (AUC/logloss/calibration by decile) + гиперпараметры
"""
from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

WORKTREE = Path(__file__).resolve().parent.parent
DATASET = WORKTREE / "out" / "value_gs" / "dataset.npz"
OUT_DIR = WORKTREE / "out" / "value_gs"


def episode_val_split(episode_id: np.ndarray, is_test: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    """True для строк выбранных в val (подмножество train, is_test уже False). Сплит по
    уникальным episode_id (не по строкам) -- та же дисциплина, что train/test сплит дальше
    вверх по пайплайну (build_value_dataset_gs.py использует crc32 % 5 для test; здесь для
    val используем отдельный hash-соль, чтобы val и test были независимы)."""
    train_mask = ~is_test
    train_eps = np.unique(episode_id[train_mask])
    rng = np.random.default_rng(seed)
    rng.shuffle(train_eps)
    n_val_eps = max(1, int(round(len(train_eps) * val_frac)))
    val_eps = set(train_eps[:n_val_eps].tolist())
    is_val = train_mask & np.isin(episode_id, list(val_eps))
    return is_val


def calibration_table(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> list:
    order = np.argsort(y_pred)
    y_true_s = y_true[order]
    y_pred_s = y_pred[order]
    n = len(y_pred_s)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    table = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        seg_true = y_true_s[lo:hi]
        seg_pred = y_pred_s[lo:hi]
        table.append({
            "decile": i + 1,
            "n": int(hi - lo),
            "pred_mean": round(float(seg_pred.mean()), 4),
            "actual_winrate": round(float(seg_true.mean()), 4),
            "pred_range": [round(float(seg_pred.min()), 4), round(float(seg_pred.max()), 4)],
        })
    return table


def main() -> int:
    t0 = time.perf_counter()
    d = np.load(DATASET, allow_pickle=True)
    X = d["X"].astype(np.float64)
    y = d["y"].astype(np.float64)
    episode_id = d["episode_id"]
    is_test = d["is_test"]

    n_total = len(y)
    n_test = int(is_test.sum())
    print(f"[train_value_gs] датасет: {n_total} строк, {n_test} test ({n_test/n_total:.1%}), "
          f"{len(np.unique(episode_id))} эпизодов", file=sys.stderr)

    SEED = 42
    is_val = episode_val_split(episode_id, is_test, val_frac=0.15, seed=SEED)
    is_train = (~is_test) & (~is_val)

    n_train, n_val = int(is_train.sum()), int(is_val.sum())
    print(f"[train_value_gs] сплит: train={n_train} ({len(np.unique(episode_id[is_train]))} эп), "
          f"val={n_val} ({len(np.unique(episode_id[is_val]))} эп), "
          f"test={n_test} ({len(np.unique(episode_id[is_test]))} эп)", file=sys.stderr)
    # эпизоды train/val/test не пересекаются (по конструкции: val <- train_eps, test отдельно
    # через is_test) -- явная проверка, не полагаемся молча на конструкцию
    eps_train = set(episode_id[is_train].tolist())
    eps_val = set(episode_id[is_val].tolist())
    eps_test = set(episode_id[is_test].tolist())
    assert not (eps_train & eps_val), "train/val episode leakage"
    assert not (eps_train & eps_test), "train/test episode leakage"
    assert not (eps_val & eps_test), "val/test episode leakage"

    X_train, y_train = X[is_train], y[is_train]
    X_val, y_val = X[is_val], y[is_val]
    X_test, y_test = X[is_test], y[is_test]

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": SEED,
    }
    N_ROUNDS = 400
    EARLY_STOP = 40

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    evals_result = {}
    booster = lgb.train(
        params, train_set,
        num_boost_round=N_ROUNDS,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOP, verbose=True),
            lgb.log_evaluation(period=25),
            lgb.record_evaluation(evals_result),
        ],
    )

    best_iter = booster.best_iteration
    print(f"[train_value_gs] лучшая итерация: {best_iter} / {N_ROUNDS}", file=sys.stderr)

    # ── метрики на held-out test (по эпизодам, не тронуто ни разу до сих пор) ──
    pred_test = booster.predict(X_test, num_iteration=best_iter)
    pred_val = booster.predict(X_val, num_iteration=best_iter)
    pred_train = booster.predict(X_train, num_iteration=best_iter)

    auc_test = roc_auc_score(y_test, pred_test)
    logloss_test = log_loss(y_test, pred_test, labels=[0, 1])
    auc_val = roc_auc_score(y_val, pred_val)
    logloss_val = log_loss(y_val, pred_val, labels=[0, 1])
    auc_train = roc_auc_score(y_train, pred_train)
    logloss_train = log_loss(y_train, pred_train, labels=[0, 1])

    calib_test = calibration_table(y_test, pred_test, n_bins=10)

    print(f"[train_value_gs] TEST  AUC={auc_test:.4f} logloss={logloss_test:.4f} (n={n_test})",
          file=sys.stderr)
    print(f"[train_value_gs] VAL   AUC={auc_val:.4f} logloss={logloss_val:.4f} (n={n_val})",
          file=sys.stderr)
    print(f"[train_value_gs] TRAIN AUC={auc_train:.4f} logloss={logloss_train:.4f} (n={n_train})",
          file=sys.stderr)
    print("[train_value_gs] calibration (test, by decile):", file=sys.stderr)
    for row in calib_test:
        print(f"  decile={row['decile']:2d} n={row['n']:5d} pred_mean={row['pred_mean']:.4f} "
              f"actual_wr={row['actual_winrate']:.4f} range={row['pred_range']}", file=sys.stderr)

    dump = booster.dump_model()
    n_trees = len(dump["tree_info"])
    print(f"[train_value_gs] деревьев в финальной модели: {n_trees}, objective={dump.get('objective')}, "
          f"average_output={dump.get('average_output')}", file=sys.stderr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "model.txt"
    booster.save_model(str(model_path), num_iteration=best_iter)
    print(f"[train_value_gs] booster сохранён: {model_path} ({model_path.stat().st_size} bytes)",
          file=sys.stderr)

    report = {
        "n_total": n_total, "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "n_episodes": {"train": len(eps_train), "val": len(eps_val), "test": len(eps_test)},
        "params": params, "num_boost_round": N_ROUNDS, "early_stopping_rounds": EARLY_STOP,
        "best_iteration": best_iter, "n_trees_final": n_trees,
        "objective": dump.get("objective"), "average_output": dump.get("average_output"),
        "metrics": {
            "train": {"auc": auc_train, "logloss": logloss_train},
            "val": {"auc": auc_val, "logloss": logloss_val},
            "test": {"auc": auc_test, "logloss": logloss_test},
        },
        "calibration_test_by_decile": calib_test,
        "acceptance": {"auc_target": 0.68, "auc_test": auc_test, "pass": bool(auc_test >= 0.68)},
        "wall_sec": round(time.perf_counter() - t0, 1),
    }
    (OUT_DIR / "train_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"metrics": report["metrics"], "acceptance": report["acceptance"],
                       "best_iteration": best_iter, "n_trees_final": n_trees}, indent=2))
    return 0 if auc_test >= 0.68 else 1


if __name__ == "__main__":
    raise SystemExit(main())
