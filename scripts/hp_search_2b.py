#!/usr/bin/env python3
"""hp_search_2b.py — аккуратный поиск гиперпараметров LightGBM-ранкера ПО VAL (не по test!)
для трёх групп роутинга (main/mid/c7, см. train_bc_2b.py::GROUPS), задание T1.4 п.1.

СХЕМА: val — это тот же самый по-episode-сплит, что train_bc_2b.py делает по умолчанию
(seed=42, val_frac=0.15, split_val_episodes) — переиспользуем функции train_bc_2b как модуль,
не дублируем логику сплита/eval. Test НЕ трогается (test-парquet вообще не загружается сюда,
кроме служебных META_COLUMNS через load_group_df, но decision_level_eval вызывается только
на срезе is_val).

СЕТКА: 2^4 факторный дизайн (low/high по каждому из 4 измерений: learning_rate, num_leaves,
min_data_in_leaf, feature_fraction) вокруг текущей продакшн-точки (pick_hyperparams) + сама
продакшн-точка как 17-я точка = 17 обучений на группу. Не полный 3^4=81 грид (задание просит
"небольшая и осмысленная, не гигантская") — угловые точки гиперкуба ловят взаимодействия
между параметрами, чего не даёт покоординатный спуск, но втрое дешевле полного 3^4.

num_boost_round у ВСЕХ комбо = 5000 (общий щедрый потолок) с early stopping на val по
NDCG@1 (as-is у train_lgb_model) — early_stopping_rounds берём из продакшн pick_hyperparams
данной группы (не варьируем этот параметр отдельно, он не входит в запрошенную сетку).

Запуск:
    .venv/bin/python scripts/hp_search_2b.py                      # все 3 группы
    .venv/bin/python scripts/hp_search_2b.py --groups main        # одна группа
    .venv/bin/python scripts/hp_search_2b.py --n-jobs 4
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "scripts"))
import train_bc_2b as tb  # noqa: E402

GROUPS = ["main", "mid", "c7"]


def build_grid(base_hp: dict) -> list[dict]:
    """base_hp — продакшн pick_hyperparams(n_train_rows) для этой группы. Возвращает список
    из 17 словарей {learning_rate, num_leaves, min_data_in_leaf, feature_fraction, label}:
    точка 'base' (продакшн-значения, feature_fraction=0.8 — текущий хардкод в
    build_common_params) + 16 угловых точек 2^4-дизайна low/high вокруг base."""
    base_nl = base_hp["num_leaves"]
    base_md = base_hp["min_data_in_leaf"]
    lr_lo, lr_hi = 0.03, 0.08
    nl_lo, nl_hi = max(7, base_nl // 2), base_nl * 2
    md_lo, md_hi = max(10, base_md // 2), base_md * 2
    ff_lo, ff_hi = 0.7, 0.9

    grid = [{"learning_rate": 0.05, "num_leaves": base_nl, "min_data_in_leaf": base_md,
              "feature_fraction": 0.8, "label": "base"}]
    for lr, nl, md, ff in itertools.product((lr_lo, lr_hi), (nl_lo, nl_hi),
                                             (md_lo, md_hi), (ff_lo, ff_hi)):
        label = f"lr{lr}_nl{nl}_md{md}_ff{ff}"
        grid.append({"learning_rate": lr, "num_leaves": nl, "min_data_in_leaf": md,
                      "feature_fraction": ff, "label": label})
    return grid


def eval_combo(combo: dict, base_hp: dict, X_tr, y_tr, g_tr, X_val, y_val, g_val,
                dec_val_meta, n_jobs: int, seed: int) -> dict:
    hp_for_rounds = dict(base_hp)  # num_boost_round/early_stopping_rounds не варьируем
    params = tb.build_common_params(hp_for_rounds, n_jobs, seed)
    params.update(learning_rate=combo["learning_rate"], num_leaves=combo["num_leaves"],
                  min_data_in_leaf=combo["min_data_in_leaf"], feature_fraction=combo["feature_fraction"])
    params.update(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1], label_gain=[0, 1])

    import lightgbm as lgb
    dtrain = lgb.Dataset(X_tr, label=y_tr, group=g_tr, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, group=g_val, reference=dtrain, free_raw_data=False)
    t0 = time.perf_counter()
    booster = lgb.train(
        params, dtrain, num_boost_round=5000, valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(base_hp["early_stopping_rounds"], verbose=False)],
    )
    wall = round(time.perf_counter() - t0, 1)
    scores_val = booster.predict(X_val, num_iteration=booster.best_iteration or None)
    dec_val = tb.decision_level_eval(dec_val_meta, scores_val, with_category=False)
    summary = tb.summarize(dec_val)
    return {**combo, "best_iteration": int(booster.best_iteration or booster.current_iteration()),
            "wall_sec": wall, "val_top1": summary["model_top1"],
            "val_baseline_first_k": summary["baseline_first_k_top1"],
            "val_chance_floor": summary["chance_floor_mean"],
            "val_n_decisions": summary["n_decisions"]}


def search_group(group: str, features_dir: Path, feature_order: list, val_frac: float,
                  seed: int, n_jobs: int) -> dict:
    df = tb.load_group_df(features_dir, group, feature_order)
    is_tr, is_val, _is_test, meta = tb.make_masks(df, val_frac, seed, group)
    base_hp = tb.pick_hyperparams(int(is_tr.sum()))
    X_tr, y_tr = tb.xy(df, is_tr, feature_order)
    X_val, y_val = tb.xy(df, is_val, feature_order)
    g_tr = tb.group_sizes_for(df, is_tr)
    g_val = tb.group_sizes_for(df, is_val)
    dec_val_meta = df.loc[is_val, ["decision_uid", "option_idx", "y", "min_count", "max_count", "opt_type"]]

    grid = build_grid(base_hp)
    print(f"[hp_search] group={group}: base_hp={base_hp} n_tr={len(X_tr)} n_val={len(X_val)} "
          f"n_val_decisions={dec_val_meta['decision_uid'].nunique()} grid_size={len(grid)}", file=sys.stderr)

    results = []
    for combo in grid:
        r = eval_combo(combo, base_hp, X_tr, y_tr, g_tr, X_val, y_val, g_val, dec_val_meta, n_jobs, seed)
        results.append(r)
        print(f"  [{group}] {combo['label']:32s} val_top1={r['val_top1']:.4f} "
              f"(floor={r['val_chance_floor']:.4f} base_first_k={r['val_baseline_first_k']:.4f}) "
              f"best_iter={r['best_iteration']} wall={r['wall_sec']}s", file=sys.stderr)

    # best: наивысший val_top1; тай-брейк — меньше num_leaves*min_data screening не нужен,
    # берём первый найденный максимум (grid порядок детерминирован -> воспроизводимо), кроме
    # случая, когда 'base' даёт тот же top1 -- тогда предпочитаем base (не менять прод без
    # выигрыша, парсимония).
    best_top1 = max(r["val_top1"] for r in results)
    tied = [r for r in results if r["val_top1"] == best_top1]
    base_result = next(r for r in results if r["label"] == "base")
    if base_result["val_top1"] == best_top1:
        best = base_result
    else:
        best = tied[0]

    return {
        "group": group, "base_hp": base_hp, "grid_results": results,
        "best": best, "base_result": base_result,
        "improved_over_base": round(best["val_top1"] - base_result["val_top1"], 4),
        "val_meta": meta,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", default=str(WORKTREE / "out" / "corpus_2b" / "features"))
    ap.add_argument("--out", default=str(WORKTREE / "out" / "corpus_2b" / "models" / "hp_search_report.json"))
    ap.add_argument("--groups", nargs="+", default=GROUPS, choices=GROUPS)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    if args.n_jobs > 4:
        print(f"[hp_search_2b] WARNING: n_jobs={args.n_jobs} > 4, ограничение нагрузки — форсирую 4", file=sys.stderr)
        args.n_jobs = 4

    features_dir = Path(args.features_dir)
    spec = json.loads((features_dir / "feature_spec.json").read_text())
    feature_order = spec["feature_order"]

    t0 = time.perf_counter()
    out = {}
    for group in args.groups:
        out[group] = search_group(group, features_dir, feature_order, args.val_frac, args.seed, args.n_jobs)

    report = {
        "groups": out,
        "run_meta": {"seed": args.seed, "val_frac": args.val_frac, "n_jobs": args.n_jobs,
                     "wall_sec": round(time.perf_counter() - t0, 1)},
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[hp_search_2b] done in {report['run_meta']['wall_sec']}s -> {args.out}", file=sys.stderr)

    print("\n=== SUMMARY ===")
    for group, res in out.items():
        b = res["best"]
        base = res["base_result"]
        print(f"{group}: base val_top1={base['val_top1']:.4f} ({base['label']}) -> "
              f"best val_top1={b['val_top1']:.4f} ({b['label']}), delta={res['improved_over_base']:+.4f}")
        print(f"   best hp: lr={b['learning_rate']} num_leaves={b['num_leaves']} "
              f"min_data_in_leaf={b['min_data_in_leaf']} feature_fraction={b['feature_fraction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
