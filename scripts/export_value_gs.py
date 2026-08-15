#!/usr/bin/env python3
"""export_value_gs.py -- экспортирует out/value_gs/model.txt (LightGBM binary, train_value_gs.py)
в LIVE-SAFE чистый-Python артефакт submissions/v084_bc2b_search/value_model_gs.py:

  value_predict(feat: list[float]) -> float   # p(win), sigmoid(сумма leaf_value по деревьям)

Порт паттерна scripts/export_bc_2b.py (flatten_tree/render_node/render_tree,
equivalence-тест, size guard) -- ключевое ОТЛИЧИЕ: bc_model_2b.py экспортировал
objective="lambdarank" (predict() == сумма leaf_value БЕЗ sigmoid); здесь
objective="binary" -- LightGBM's booster.predict() (raw_score=False, по умолчанию)
ПРИМЕНЯЕТ sigmoid к сумме leaf_value. Экспортёр поэтому даёт ДВЕ функции:
  raw_score(feat) -> float   -- сумма leaf_value (сравнивается с booster.predict(X, raw_score=True))
  value_predict(feat) -> float -- sigmoid(raw_score(feat)) (сравнивается с booster.predict(X))
Только math.exp используется (stdlib, не numpy/pandas/lightgbm/pickle -- см. CLAUDE.md
"submission — ТОЛЬКО чистый Python"; math уже используется в этом сабмите, см.
submissions/v084_bc2b_search/search_gs.py:95).

ПРУНИНГ ЗАПРЕЩЁН как способ уложиться в --max-bytes (урок "итерации 5" из project memory:
прунинг ПОСЛЕ обучения ломает эквивалентность python_pred == booster.predict). Если файл
превышает лимит -- сигнал переобучить с меньшим num_boost_round/num_leaves
(scripts/train_value_gs.py), а не обрезать готовую модель.

Запуск:
    .venv/bin/python scripts/export_value_gs.py                  # полный экспорт + тест (2000 строк)
    .venv/bin/python scripts/export_value_gs.py --n-test 5000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

WORKTREE = Path(__file__).resolve().parent.parent
DATASET = WORKTREE / "out" / "value_gs" / "dataset.npz"
MODEL_TXT = WORKTREE / "out" / "value_gs" / "model.txt"
OUT_PATH = WORKTREE / "submissions" / "v084_bc2b_search" / "value_model_gs.py"
MAX_BYTES = 3 * 1024 * 1024


# ═══════════════════════════ 1. Экспорт дерева (порт export_bc_2b.py::flatten_tree) ═══════════════════════════

def flatten_tree(node: dict) -> list:
    """Идентично export_bc_2b.py::flatten_tree -- dump_model()['tree_structure'] (вложенный
    dict) -> плоский pre-order список узлов (feature_idx, value, left, right, default_left).
    feature_idx=-1 -> лист (value=leaf_value)."""
    nodes: list = []

    def rec(n: dict) -> int:
        if "split_index" not in n:
            nodes.append((-1, float(n["leaf_value"]), -1, -1, 0))
            return len(nodes) - 1
        my_idx = len(nodes)
        nodes.append(None)
        left_idx = rec(n["left_child"])
        right_idx = rec(n["right_child"])
        nodes[my_idx] = (
            int(n["split_feature"]), float(n["threshold"]),
            left_idx, right_idx, 1 if n.get("default_left") else 0,
        )
        return my_idx

    rec(node)
    return nodes


def export_trees(booster: lgb.Booster) -> tuple:
    dump = booster.dump_model()
    if dump.get("objective") != "binary sigmoid:1":
        raise RuntimeError(
            f"неожиданный objective={dump.get('objective')!r} -- value_predict() ниже "
            "предполагает objective='binary' (sigmoid к сумме leaf_value); нужна ревизия формулы."
        )
    if dump.get("average_output"):
        raise RuntimeError("average_output=True неожидан для binary -- ревизия формулы value_predict()")
    trees = tuple(flatten_tree(t["tree_structure"]) for t in dump["tree_info"])
    return trees, dump


def render_node(node: tuple) -> str:
    f, v, l, r, dl = node
    return f"({f},{v!r},{l},{r},{dl})"


def render_tree(nodes: list) -> str:
    return "(" + ",".join(render_node(n) for n in nodes) + ",)"


_MODULE_HEADER = '''"""value_model_gs.py -- LIVE-SAFE чистый-Python инференс value-модели (PLAN_v084_SEARCH.md
§3ter п.2). СГЕНЕРИРОВАНО scripts/export_value_gs.py из out/value_gs/model.txt
(scripts/train_value_gs.py) -- НЕ РЕДАКТИРОВАТЬ РУКАМИ, перезапустить экспорт при переобучении.

Ноль внешних зависимостей (не numpy/lightgbm/pandas/eval/exec/pickle), только stdlib `math`
(sigmoid) -- та же дисциплина, что submissions/v084_bc2b_search/bc_model_2b.py (см. его
докстринг "три меньше точек отказа live") и search_gs.py (уже импортирует math).

ФОРМАТ ДЕРЕВА: идентичен bc_model_2b.py -- tuple деревьев; дерево = tuple узлов в pre-order;
узел = (feature_idx, value, left, right, default_left). feature_idx==-1 -> ЛИСТ
(value=leaf_value). feature_idx>=0 -> ВНУТРЕННИЙ узел, "feat[feature_idx] <= value" ->
left, иначе right; default_left -- куда идти при NaN/отсутствующей фиче.

@@N_FEATURES@@ фич на вектор, порядок = submissions/v084_bc2b_search/value_features_gs.py
::FEATURE_ORDER -- контракт, индексы должны совпадать байт-в-байт с тем, что видела модель
при обучении (out/value_gs/dataset.npz, построен той же value_features_gs.value_features()).

СЕМАНТИКА СКОРА: objective="binary" (LightGBM). В отличие от bc_model_2b.py (lambdarank,
predict() БЕЗ sigmoid), booster.predict() ПО УМОЛЧАНИЮ применяет sigmoid к сумме leaf_value
(raw_score=False -- это вероятность). raw_score() ниже -- голая сумма leaf_value (сравнивается
с booster.predict(X, raw_score=True) в тесте эквивалентности, допуск <1e-9); value_predict() --
sigmoid(raw_score()) == p(win) (сравнивается с booster.predict(X) по умолчанию).

@@N_TREES@@ деревьев, обучены train_value_gs.py (LightGBM num_leaves=@@NUM_LEAVES@@,
max_depth=@@MAX_DEPTH@@, learning_rate=@@LR@@, best_iteration=@@BEST_ITER@@ early-stop по val
AUC/logloss). Метрики held-out (по эпизодам): см. out/value_gs/train_report.json.
"""
from __future__ import annotations

import math

N_FEATURES = @@N_FEATURES@@

TREES = (
'''

_MODULE_FOOTER = '''
)


def raw_score(feat) -> float:
    """Сумма leaf_value по всем деревьям (list/tuple длины N_FEATURES, порядок =
    value_features_gs.FEATURE_ORDER). ТО ЖЕ САМОЕ число, что
    booster.predict(X, raw_score=True) у LightGBM (см. докстринг модуля и отчёт
    эквивалентности export_value_gs.py: max|diff|<1e-9)."""
    total = 0.0
    for nodes in TREES:
        node = nodes[0]
        f = node[0]
        while f != -1:
            v = feat[f]
            go_left = (v <= node[1]) if v == v else bool(node[4])
            node = nodes[node[2] if go_left else node[3]]
            f = node[0]
        total += node[1]
    return total


def value_predict(feat) -> float:
    """p(win) в [0,1] -- sigmoid(raw_score(feat)). ТО ЖЕ САМОЕ число, что
    booster.predict(X) (raw_score=False, поведение LightGBM по умолчанию для
    objective='binary') -- см. отчёт эквивалентности export_value_gs.py."""
    x = raw_score(feat)
    # sigmoid численно устойчивый (без overflow на больших |x|) -- та же формула,
    # что LightGBM's internal Sigmoid::transform (1 / (1 + exp(-x))), переписанная
    # через exp(-|x|) чтобы избежать OverflowError на редких экстремальных raw-score.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
'''


def render_module(trees: tuple, n_features: int, meta: dict) -> str:
    header = (_MODULE_HEADER
              .replace("@@N_FEATURES@@", str(n_features))
              .replace("@@N_TREES@@", str(len(trees)))
              .replace("@@NUM_LEAVES@@", str(meta.get("num_leaves")))
              .replace("@@MAX_DEPTH@@", str(meta.get("max_depth")))
              .replace("@@LR@@", str(meta.get("learning_rate")))
              .replace("@@BEST_ITER@@", str(meta.get("best_iteration"))))
    lines = [header]
    for nodes in trees:
        lines.append(f"  {render_tree(nodes)},")
    body = "\n".join(lines)
    return body + _MODULE_FOOTER


# ═══════════════════════════ 2. Тест эквивалентности ═══════════════════════════

def run_equivalence_test(booster: lgb.Booster, mod, X: np.ndarray, n_test: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(n_test, n), replace=False)
    X_sub = X[idx].astype(np.float64)

    t0 = time.perf_counter()
    raw_pp = [mod.raw_score(row) for row in X_sub]
    pred_pp = [mod.value_predict(row) for row in X_sub]
    wall = time.perf_counter() - t0

    raw_lgb = booster.predict(X_sub, raw_score=True)
    pred_lgb = booster.predict(X_sub, raw_score=False)

    max_raw_diff = float(np.max(np.abs(np.asarray(raw_pp) - raw_lgb)))
    max_pred_diff = float(np.max(np.abs(np.asarray(pred_pp) - pred_lgb)))

    return {
        "n_rows": int(len(idx)),
        "max_abs_raw_score_diff": max_raw_diff,
        "max_abs_pred_diff": max_pred_diff,
        "sec_per_row_mean": wall / max(1, len(idx)),
    }


# ═══════════════════════════ main ═══════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-txt", default=str(MODEL_TXT))
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    t_start = time.perf_counter()
    booster = lgb.Booster(model_file=str(args.model_txt))
    trees, dump = export_trees(booster)
    n_features = booster.num_feature()

    n_nodes = sum(len(t) for t in trees)
    print(f"[export_value_gs] деревьев={len(trees)}, узлов={n_nodes}, n_features={n_features}, "
          f"objective={dump.get('objective')}", file=sys.stderr)

    train_report_path = args.model_txt.replace("model.txt", "train_report.json") \
        if "model.txt" in args.model_txt else None
    meta = {"num_leaves": None, "max_depth": None, "learning_rate": None, "best_iteration": len(trees)}
    if train_report_path and Path(train_report_path).exists():
        tr = json.loads(Path(train_report_path).read_text())
        meta["num_leaves"] = tr["params"].get("num_leaves")
        meta["max_depth"] = tr["params"].get("max_depth")
        meta["learning_rate"] = tr["params"].get("learning_rate")
        meta["best_iteration"] = tr.get("best_iteration", len(trees))

    src = render_module(trees, n_features, meta)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(src)
    compile(src, str(out_path), "exec")
    out_bytes = out_path.stat().st_size
    print(f"[export_value_gs] записано: {out_path} ({out_bytes} bytes)", file=sys.stderr)

    if out_bytes > args.max_bytes:
        print(f"[export_value_gs] РАЗМЕР {out_bytes} > --max-bytes {args.max_bytes} -- "
              "прунинг после обучения ЗАПРЕЩЁН (ломает эквивалентность, см. докстринг модуля). "
              "Переобучить scripts/train_value_gs.py с меньшим num_boost_round/num_leaves.",
              file=sys.stderr)
        return 1

    # ── импорт сгенерированного модуля + тест эквивалентности ──
    import importlib.util
    spec = importlib.util.spec_from_file_location("value_model_gs_generated", out_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.N_FEATURES == n_features

    d = np.load(args.dataset, allow_pickle=True)
    X = d["X"]
    equiv = run_equivalence_test(booster, mod, X, args.n_test, args.seed)
    print(f"[export_value_gs] эквивалентность: n={equiv['n_rows']} "
          f"max|raw_diff|={equiv['max_abs_raw_score_diff']:.3e} "
          f"max|pred_diff|={equiv['max_abs_pred_diff']:.3e} "
          f"({equiv['sec_per_row_mean']*1000:.4f} ms/строка)", file=sys.stderr)

    ok = (equiv["max_abs_raw_score_diff"] < 1e-9 and equiv["n_rows"] >= 2000
          and out_bytes <= args.max_bytes)

    report = {
        "n_trees": len(trees), "n_nodes": n_nodes, "n_features": n_features,
        "objective": dump.get("objective"), "meta": meta,
        "out_bytes": out_bytes, "max_bytes": args.max_bytes,
        "equivalence": equiv,
        "pass": ok,
        "wall_sec": round(time.perf_counter() - t_start, 1),
    }
    report_path = WORKTREE / "out" / "value_gs" / "export_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[export_value_gs] {'PASS' if ok else 'FAIL'} -- done in {report['wall_sec']}s -> {out_path}",
          file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
