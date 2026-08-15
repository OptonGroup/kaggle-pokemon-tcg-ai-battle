#!/usr/bin/env python3
"""eval_holdout_v088.py — острый офлайн-прибор для v088: сравнение N наборов моделей на
СВЕЖЕМ холдауте поля (эпизоды, которых не видел ни один кандидат).

Зачем: замороженный test корпуса даёт всего ~108 retreat-решений (se≈4.8пп) — целевой
эффект K1 (+5пп) там СТРУКТУРНО неотличим от шума. Холдаут из
select_extension_v088.py (4000 свободных эпизодов пула) даёт ~3-4k retreat-решений,
se≈0.8пп, плюс режется по архетипу соперника (в т.ч. dragapult — цель K2).

Сравнение ПАРНОЕ: все модели скорят ОДНИ И ТЕ ЖЕ решения, поэтому дельта считается не
через независимые доверительные интервалы, а по МакНемару на дискордантных парах (это и
есть правильный тест для «та же выборка, два классификатора»; MEASUREMENT_POLICY:
по-парно, без пулинга разнородных страт).

Срезы: overall, по группе роутинга, по категории решения (main: attack/retreat/energy-
attach/...), по opp_archetype, и отдельно КРИТИЧЕСКОЕ состояние retreat (тот же предикат,
что у train_bc_2b_v088.retreat_state_weight_for) — прямая мишень K1.

Пример:
  .venv/bin/python scripts/eval_holdout_v088.py \
    --features-dir out/corpus_2b/versions/v088_dragapult/holdout_features \
    --model base=out/corpus_2b/versions/v086_targeted/models_v087_bestiter \
    --model k1=out/corpus_2b/versions/v088_run/models_w15_c30 \
    --baseline base --out out/corpus_2b/versions/v088_run/holdout_report.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
GROUPS = ["main", "c7", "low", "mid", "easy", "rest"]


def load_trainer():
    """train_bc_2b_v088.py — источник истины для decision_level_eval/summarize/OPT_*:
    прибор ОБЯЗАН считать top-1 ровно тем же кодом, что и тренер, иначе цифры
    несопоставимы с metrics.json."""
    path = REPO / "scripts" / "train_bc_2b_v088.py"
    spec = importlib.util.spec_from_file_location("trainer_v088", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trainer_v088"] = mod
    spec.loader.exec_module(mod)
    return mod


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Парный тест «модель a лучше b на одних и тех же решениях». b01 = b попал, a промахнулся;
    b10 = a попал, b промахнулся. z — нормальное приближение биномиального знакового теста
    по дискордантным парам (n_disc>=25 — приближение адекватно; иначе честно помечаем)."""
    a = a.astype(bool); b = b.astype(bool)
    b10 = int((a & ~b).sum())
    b01 = int((~a & b).sum())
    n_disc = b10 + b01
    if n_disc == 0:
        return {"delta_pp": 0.0, "n_discordant": 0, "z": None, "p_two_sided": None,
                "note": "нет дискордантных пар"}
    z = (b10 - b01) / math.sqrt(n_disc)
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"delta_pp": round(100.0 * (b10 - b01) / len(a), 3),
            "n_discordant": n_disc, "wins_a": b10, "wins_b": b01,
            "z": round(z, 3), "p_two_sided": round(p, 6),
            "note": None if n_disc >= 25 else "n_discordant<25 — z-приближение шаткое"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=DIR",
                    help="каталог с <group>.txt; можно повторять")
    ap.add_argument("--baseline", default=None, help="LABEL модели-базы для парных дельт")
    ap.add_argument("--groups", nargs="+", default=GROUPS, choices=GROUPS)
    ap.add_argument("--crit-hp-frac", type=float, default=0.6)
    ap.add_argument("--crit-bench-energy", type=float, default=2.0)
    ap.add_argument("--crit-turn", type=float, default=10.0)
    ap.add_argument("--min-slice-n", type=int, default=200,
                    help="страты меньше этого размера не печатаются (шум)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    T = load_trainer()
    features_dir = Path(args.features_dir)
    spec = json.loads((features_dir / "feature_spec.json").read_text())
    feature_order = spec["feature_order"]

    models: dict[str, Path] = {}
    for m in args.model:
        if "=" not in m:
            sys.exit(f"--model должен быть LABEL=DIR, получено: {m}")
        label, d = m.split("=", 1)
        models[label] = Path(d)
    labels = list(models)
    base_label = args.baseline
    if base_label is not None and base_label not in models:
        sys.exit(f"--baseline {base_label} не найден среди --model {labels}")

    frames = []
    for group in args.groups:
        pq = features_dir / f"{group}.parquet"
        if not pq.exists():
            print(f"[eval] {group}: нет {pq} — пропуск", file=sys.stderr)
            continue
        df = T.load_group_df(features_dir, group, feature_order)
        if not len(df):
            continue
        X = df[feature_order].to_numpy(dtype=np.float32)
        meta = df[["decision_uid", "option_idx", "y", "min_count", "max_count",
                   "opt_type", "opt_card_id"]]
        dec_ref = None
        per_model_match = {}
        for label in labels:
            mp = models[label] / f"{group}.txt"
            if not mp.exists():
                sys.exit(f"нет модели {mp}")
            booster = lgb.Booster(model_file=str(mp))
            scores = booster.predict(X, num_iteration=booster.best_iteration or None)
            dec = T.decision_level_eval(meta, scores, with_category=(group == "main"),
                                        with_multiset=False)
            if dec_ref is None:
                dec_ref = dec
            elif not (dec["decision_uid"].to_numpy() == dec_ref["decision_uid"].to_numpy()).all():
                sys.exit(f"{group}: порядок решений разъехался между моделями")
            per_model_match[label] = dec["model_match"].to_numpy()

        # decision-уровневые метаданные + предикат критического состояния (тот же, что в тренере)
        dmeta_cols = ["decision_uid", "opp_archetype", "expert", "turn_idx",
                      "active_hp_frac", "bench_energy_sum"]
        dmeta = df[dmeta_cols].groupby("decision_uid", sort=False).first()
        out = dec_ref[["decision_uid", "baseline_match", "chance_floor"]].copy()
        if "category" in dec_ref.columns:
            out["category"] = dec_ref["category"].to_numpy()
        else:
            out["category"] = None
        out = out.join(dmeta, on="decision_uid", rsuffix="_m")
        out["group"] = group
        out["is_crit"] = ((out["active_hp_frac"] <= args.crit_hp_frac)
                          & (out["bench_energy_sum"] >= args.crit_bench_energy)
                          & (out["turn_idx"] >= args.crit_turn))
        for label in labels:
            out[f"m_{label}"] = per_model_match[label]
        frames.append(out)
        print(f"[eval] {group}: {len(out)} решений, "
              + ", ".join(f"{l}={per_model_match[l].mean():.4f}" for l in labels),
              file=sys.stderr)

    if not frames:
        sys.exit("нечего оценивать")
    dec = pd.concat(frames, ignore_index=True)

    def slice_report(sub: pd.DataFrame) -> dict:
        r = {"n_decisions": int(len(sub)),
             "always_first_k": round(float(sub["baseline_match"].mean()), 4)}
        for label in labels:
            r[label] = round(float(sub[f"m_{label}"].mean()), 4)
        if base_label:
            for label in labels:
                if label == base_label:
                    continue
                r[f"paired_{label}_vs_{base_label}"] = mcnemar(
                    sub[f"m_{label}"].to_numpy(), sub[f"m_{base_label}"].to_numpy())
        return r

    report: dict = {
        "features_dir": str(features_dir),
        "models": {l: str(p) for l, p in models.items()},
        "baseline": base_label,
        "crit_predicate": {"active_hp_frac<=": args.crit_hp_frac,
                           "bench_energy_sum>=": args.crit_bench_energy,
                           "turn_idx>=": args.crit_turn},
        "overall": slice_report(dec),
        "by_group": {}, "by_category": {}, "by_archetype": {},
        "retreat_focus": {},
    }
    for g, sub in dec.groupby("group"):
        report["by_group"][g] = slice_report(sub)
    main_dec = dec[dec["group"] == "main"]
    for c, sub in main_dec.groupby("category"):
        if len(sub) >= args.min_slice_n:
            report["by_category"][str(c)] = slice_report(sub)
    for a, sub in dec.groupby("opp_archetype"):
        if len(sub) >= args.min_slice_n:
            report["by_archetype"][str(a)] = slice_report(sub)

    # ГЛАВНАЯ мишень K1: retreat-решения, критические и нет
    ret = main_dec[main_dec["category"] == "retreat"]
    report["retreat_focus"] = {
        "all_retreat": slice_report(ret),
        "retreat_critical": slice_report(ret[ret["is_crit"]]),
        "retreat_noncritical": slice_report(ret[~ret["is_crit"]]),
        # контроль на ложные срабатывания: НЕ-retreat решения в критическом состоянии
        "nonretreat_in_critical_state": slice_report(
            main_dec[(main_dec["category"] != "retreat") & main_dec["is_crit"]]),
    }

    txt = json.dumps(report, indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt)
        print(f"[eval] отчёт -> {args.out}", file=sys.stderr)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
