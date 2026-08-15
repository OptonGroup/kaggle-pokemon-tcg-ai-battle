#!/usr/bin/env python3
"""eval_holdout_v089.py — прибор для v089 (БЛОК 8, "готовность замены").

ОТЛИЧИЕ ОТ eval_holdout_v088.py: сравниваемые модели живут на РАЗНЫХ наборах фич
(v087/v088 — 284 фичи, v089 — 293), поэтому одной матрицей X их не прогнать. Решение:
каждая модель скорится на СВОЁМ holdout_features (оба построены из ОДНОГО
decisions_extra.jsonl = одни и те же 4000 эпизодов / 441829 решений), а результаты
сшиваются по decision_uid. Инвариант проверяется явно (шаг 0): множества decision_uid
обязаны совпадать, иначе парный МакНемар невалиден и скрипт падает.

Валидации (все обязательны, СТОП при провале):
  0. decision_uid старого и нового холдаутов совпадают как множества.
  1. v087/v088 на старом холдауте воспроизводят holdout_final.json (retreat 0.4900 -> +9.35пп).
  2. own_bench_healthier_same_exists (новая фича) совпадает с предикатом подкласса,
     посчитанным независимо по СЫРЫМ obs — то есть БЛОК 8 действительно измеряет то,
     что заявлено (а не читает борт мимо).

Срезы: overall, retreat, подкласс "своп повреждённого активного 648", retreat вне подкласса,
attack (контроль просадки), плюс те же срезы внутри зеркала.

Пример:
  .venv/bin/python scripts/eval_holdout_v089.py \
    --old-features out/corpus_2b/versions/v088_dragapult/holdout_features \
    --new-features out/corpus_2b/versions/v089_block8/holdout_features \
    --model-base out/corpus_2b/versions/v086_targeted/models_v087_bestiter \
    --model-v088 out/corpus_2b/versions/v088_dragapult/final/B_k1k2_16 \
    --model-v089 out/corpus_2b/versions/v089_block8/models_w16 \
    --out out/corpus_2b/versions/v089_block8/holdout_report.json
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
GRIMMSNARL_EX = 648
DAMAGE_THRESHOLD = 60.0
PUBLISHED = {"n": 4182, "base": 0.4900, "v088": 0.5835, "delta_pp": 9.35}


def load_trainer():
    path = REPO / "scripts" / "train_bc_2b_v088.py"
    spec = importlib.util.spec_from_file_location("trainer_v088", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trainer_v088"] = mod
    spec.loader.exec_module(mod)
    return mod


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """a лучше b на одних и тех же решениях. b10 = a попал/b промахнулся."""
    a = a.astype(bool)
    b = b.astype(bool)
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


def score_model(T, features_dir: Path, model_dir: Path, group: str = "main"):
    """-> DataFrame[decision_uid, match, category, ...] + сырой df фич для метаданных."""
    spec = json.loads((features_dir / "feature_spec.json").read_text())
    feature_order = spec["feature_order"]
    df = T.load_group_df(features_dir, group, feature_order)
    X = df[feature_order].to_numpy(dtype=np.float32)
    meta = df[["decision_uid", "option_idx", "y", "min_count", "max_count",
               "opt_type", "opt_card_id"]]
    booster = lgb.Booster(model_file=str(model_dir / f"{group}.txt"))
    scores = booster.predict(X, num_iteration=booster.best_iteration or None)
    dec = T.decision_level_eval(meta, scores, with_category=True, with_multiset=False)
    return dec, df


def subclass_from_obs(cand_uids, obs_root: Path) -> tuple[set, int]:
    """Предикат подкласса НЕЗАВИСИМО от фичей — по сырым obs (тот же код, что в A2)."""
    by_ep: dict[int, list] = {}
    for uid in cand_uids:
        eid_s, ai_s, ref_s = uid.split(":")
        by_ep.setdefault(int(eid_s), []).append((uid, int(ai_s), int(ref_s)))
    found, missing = set(), 0
    for eid, items in by_ep.items():
        shard = obs_root / f"{eid}.jsonl"
        if not shard.exists():
            missing += len(items)
            continue
        lines = shard.read_text().splitlines()
        for uid, ai, ref in items:
            if ref >= len(lines):
                continue
            obs = json.loads(lines[ref])
            me = obs["current"]["players"][ai]
            active = me.get("active") or []
            if not active or active[0].get("id") != GRIMMSNARL_EX:
                continue
            act = active[0]
            act_dmg = float(act.get("maxHp", 0) - act.get("hp", 0))
            if any((b.get("id") == GRIMMSNARL_EX)
                   and (float(b.get("maxHp", 0) - b.get("hp", 0)) < act_dmg)
                   for b in (me.get("bench") or [])):
                found.add(uid)
    return found, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-features", required=True)
    ap.add_argument("--new-features", required=True)
    ap.add_argument("--obs-root",
                    default="out/corpus_2b/versions/v088_dragapult/holdout_field/obs_shards")
    ap.add_argument("--model-base", required=True, help="v087, retreat_weight=1.0")
    ap.add_argument("--model-v088", required=True, help="v088 shipped (вес 1.6 + K2)")
    ap.add_argument("--model-v089", required=True, help="v089 (вес 1.6 + K2 + БЛОК 8)")
    ap.add_argument("--skip-published-check", action="store_true",
                    help="пропустить валидацию #1 (только для отладки)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    T = load_trainer()
    old_dir, new_dir = Path(args.old_features), Path(args.new_features)
    obs_root = Path(args.obs_root)

    print("[v089] скорю base/v088 на СТАРОМ холдауте (284 фичи)...", file=sys.stderr)
    dec_base, df_old = score_model(T, old_dir, Path(args.model_base))
    dec_v088, _ = score_model(T, old_dir, Path(args.model_v088))
    print("[v089] скорю v089 на НОВОМ холдауте (293 фичи)...", file=sys.stderr)
    dec_v089, df_new = score_model(T, new_dir, Path(args.model_v089))

    # ── валидация 0: одни и те же решения ──
    u_old = dec_base["decision_uid"].to_numpy()
    u_v088 = dec_v088["decision_uid"].to_numpy()
    u_new = dec_v089["decision_uid"].to_numpy()
    assert (u_old == u_v088).all(), "порядок решений разъехался между base и v088"
    if not (len(u_old) == len(u_new) and set(u_old) == set(u_new)):
        print(f"[v089] ВАЛИДАЦИЯ 0 ПРОВАЛЕНА: старый холдаут {len(u_old)} решений, "
              f"новый {len(u_new)}; пересечение {len(set(u_old) & set(u_new))}", file=sys.stderr)
        return 1
    print(f"[v089] ВАЛИДАЦИЯ 0 OK — {len(u_old)} решений совпадают в обоих холдаутах",
          file=sys.stderr)

    base_df = dec_base[["decision_uid", "baseline_match", "category"]].copy()
    base_df["m_base"] = dec_base["model_match"].to_numpy()
    base_df["m_v088"] = dec_v088["model_match"].to_numpy()
    v089_map = dict(zip(u_new, dec_v089["model_match"].to_numpy()))
    base_df["m_v089"] = base_df["decision_uid"].map(v089_map).to_numpy()
    assert not base_df["m_v089"].isna().any(), "не все decision_uid нашлись у v089"

    # ── валидация 1: воспроизведение опубликованных чисел v088 ──
    ret_all = base_df[base_df["category"] == "retreat"]
    v_base, v_v088 = float(ret_all["m_base"].mean()), float(ret_all["m_v088"].mean())
    v_mc = mcnemar(ret_all["m_v088"].to_numpy(), ret_all["m_base"].to_numpy())
    print(f"[v089] ВАЛИДАЦИЯ 1 retreat: n={len(ret_all)} base={v_base:.4f} "
          f"v088={v_v088:.4f} delta={v_mc['delta_pp']}пп "
          f"(ожидается n=4182 / 0.4900 / 0.5835 / +9.35)", file=sys.stderr)
    ok1 = (len(ret_all) == PUBLISHED["n"] and abs(v_base - PUBLISHED["base"]) < 1e-3
           and abs(v_v088 - PUBLISHED["v088"]) < 1e-3
           and abs(v_mc["delta_pp"] - PUBLISHED["delta_pp"]) < 0.05)
    if not ok1 and not args.skip_published_check:
        print("[v089] ВАЛИДАЦИЯ 1 ПРОВАЛЕНА — СТОП", file=sys.stderr)
        return 1
    print("[v089] ВАЛИДАЦИЯ 1 OK", file=sys.stderr)

    # ── метаданные решения + подкласс ──
    dmeta = df_old[["decision_uid", "opp_archetype", "active_id", "active_damage"]] \
        .groupby("decision_uid", sort=False).first()
    ret = ret_all.join(dmeta, on="decision_uid")
    cand = ret[(ret["active_id"] == GRIMMSNARL_EX)
               & (ret["active_damage"] >= DAMAGE_THRESHOLD)]
    sub_uids, missing = subclass_from_obs(list(cand["decision_uid"]), obs_root)
    if missing:
        print(f"[v089] ВНИМАНИЕ: {missing} решений без obs-шарда", file=sys.stderr)
    print(f"[v089] подкласс подтверждён на {len(sub_uids)} из {len(cand)} кандидатов",
          file=sys.stderr)

    # ── валидация 2: новая фича == независимый предикат по сырым obs ──
    feat_col = "own_bench_healthier_same_exists"
    val2 = {"checked": False}
    if feat_col in df_new.columns:
        fmeta = df_new[["decision_uid", feat_col, "active_id", "active_damage"]] \
            .groupby("decision_uid", sort=False).first()
        chk = cand.join(fmeta[[feat_col]], on="decision_uid")
        pred_feat = (chk[feat_col] == 1.0).to_numpy()
        pred_obs = chk["decision_uid"].isin(sub_uids).to_numpy()
        agree = int((pred_feat == pred_obs).sum())
        val2 = {"checked": True, "n": int(len(chk)), "agree": agree,
                "agreement": round(agree / len(chk), 4) if len(chk) else None,
                "feat_pos": int(pred_feat.sum()), "obs_pos": int(pred_obs.sum())}
        print(f"[v089] ВАЛИДАЦИЯ 2 (БЛОК 8 vs сырые obs): совпало {agree}/{len(chk)} "
              f"= {val2['agreement']}", file=sys.stderr)
        if val2["agreement"] is not None and val2["agreement"] < 0.98:
            print("[v089] ВАЛИДАЦИЯ 2 ПРОВАЛЕНА — фича читает борд не так, как предикат; СТОП",
                  file=sys.stderr)
            return 1
        print("[v089] ВАЛИДАЦИЯ 2 OK", file=sys.stderr)

    # ── срезы ──
    mirror_uids = set(dmeta.index[dmeta["opp_archetype"] == "mirror_grimmsnarl"])
    ret_uids = set(ret_all["decision_uid"])
    comp_uids = ret_uids - sub_uids

    def report(uids: set | None, name: str, frame: pd.DataFrame = None) -> dict:
        sub = frame if frame is not None else base_df[base_df["decision_uid"].isin(uids)]
        n = len(sub)
        if not n:
            return {"slice": name, "n": 0}
        r = {"slice": name, "n": n,
             "always_first_k": round(float(sub["baseline_match"].mean()), 4),
             "base": round(float(sub["m_base"].mean()), 4),
             "v088": round(float(sub["m_v088"].mean()), 4),
             "v089": round(float(sub["m_v089"].mean()), 4),
             "se_pp_rough": round(100.0 * 0.5 / math.sqrt(n), 2)}
        m_base = sub["m_base"].to_numpy()
        m_88 = sub["m_v088"].to_numpy()
        m_89 = sub["m_v089"].to_numpy()
        r["v089_vs_v088"] = mcnemar(m_89, m_88)
        r["v089_vs_base"] = mcnemar(m_89, m_base)
        r["v088_vs_base"] = mcnemar(m_88, m_base)
        return r

    slices = [
        report(None, "overall (main)", base_df),
        report(ret_uids, "retreat (весь класс)"),
        report(sub_uids, "★ подкласс: своп повреждённого 648"),
        report(comp_uids, "retreat вне подкласса"),
        report(None, "attack (контроль просадки)",
               base_df[base_df["category"] == "attack"]),
        report(sub_uids & mirror_uids, "подкласс ∩ зеркало"),
        report(comp_uids & mirror_uids, "вне подкласса ∩ зеркало"),
    ]

    print("\n" + "=" * 108)
    print(f"{'срез':<42} {'n':>7} {'base':>8} {'v088':>8} {'v089':>8} "
          f"{'v089-v088':>11} {'z':>7}")
    print("=" * 108)
    for s in slices:
        if not s["n"]:
            continue
        d = s["v089_vs_v088"]
        print(f"{s['slice']:<42} {s['n']:>7} {s['base']:>8.4f} {s['v088']:>8.4f} "
              f"{s['v089']:>8.4f} {d['delta_pp']:>+10.2f}пп {str(d['z']):>7}")
    print("=" * 108)

    out = {"validations": {"same_decisions": True, "published_v088": ok1, "block8_vs_obs": val2},
           "slices": slices,
           "models": {"base": args.model_base, "v088": args.model_v088, "v089": args.model_v089}}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"[v089] отчёт -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
