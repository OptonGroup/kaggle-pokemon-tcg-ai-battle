#!/usr/bin/env python3
"""validate_block8_v089.py — ручная проверка БЛОКА 8 ("готовность замены", v089) на КОНКРЕТНЫХ
решениях, найденных замером A3 (out/mirror_levers_20260815/a3_corpus_mechanic/report.md).

Зачем отдельным скриптом: прогон фичеризации по 1.04M решений стоит ~час, и если блок читает
борд неправильно, это выяснится только после. Здесь тот же код (decision_base_features) гоняется
на 5 решениях с ИЗВЕСТНЫМ ответом (проверены бит-в-бит по сырым obs в A3):
  87558014 ref=65  — Luca СВОПАЕТ: active 648 hp=140/320, bench[0]=648 hp=320/320, 2 энергии
  87558014 ref=110 — тот же своп позже в игре
  87556853 ref=56  — Luca ОТКАЗЫВАЕТСЯ: active 648 hp=130/320, лучшая копия hp=310
  87556853 ref=68  — отказ, active hp=190/320, копия hp=310
Ожидание: healthier_exists=1 во всех четырёх (копия здоровее), charged_healthier — по факту
энергий на копии. Блок описывает ВОЗМОЖНОСТЬ, не решение эксперта: он обязан быть =1 и там,
где эксперт не свопнул (иначе фича утекала бы таргет).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CASES = [
    (87558014, 65, "СВОП (Luca): active hp140/320, bench 648 hp320 + 2 энергии"),
    (87558014, 110, "СВОП (Luca): active hp140/320"),
    (87556853, 56, "ОТКАЗ (Luca): active hp130/320, лучшая копия hp310"),
    (87556853, 68, "ОТКАЗ (Luca): active hp190/320, лучшая копия hp310"),
]

BLOCK8 = ["own_bench_same_id_n", "own_bench_best_same_hp", "own_bench_best_same_hp_frac",
          "own_bench_best_same_hp_minus_active", "own_bench_best_same_energy_n",
          "own_bench_best_same_attack_ready", "own_bench_best_same_max_damage",
          "own_bench_healthier_same_exists", "own_bench_charged_healthier_same_exists"]


def find_shard(ep: int) -> Path | None:
    for cand in (REPO / "out/corpus_2b/versions/v088_dragapult/merged/obs_shards" / f"{ep}.jsonl",
                 REPO / "out/corpus_2b/obs_shards" / f"{ep}.jsonl"):
        if cand.exists():
            return cand
    return None


def main() -> int:
    spec = importlib.util.spec_from_file_location("fz", REPO / "scripts/featurize_corpus_2b.py")
    F = importlib.util.module_from_spec(spec)
    sys.modules["fz"] = F
    spec.loader.exec_module(F)

    from cg.api import all_attack, all_card_data
    card_db = {int(c.cardId): c for c in all_card_data()}
    attack_db = {int(a.attackId): a for a in all_attack()}
    ready_cost = F.compute_attack_ready_cost(card_db, attack_db)
    atk_costs = F.compute_card_attack_costs(card_db, attack_db)

    ok = True
    for ep, ref, desc in CASES:
        shard = find_shard(ep)
        if shard is None:
            print(f"[!] нет шарда для {ep}")
            ok = False
            continue
        with shard.open() as fh:
            for i, line in enumerate(fh):
                if i == ref:
                    obs = json.loads(line)
                    break
            else:
                print(f"[!] {ep}: нет строки {ref}")
                ok = False
                continue

        feats = F.decision_base_features(obs, ready_cost, atk_costs)
        cur = obs.get("current") or {}
        me = int(cur.get("yourIndex") or 0)
        mine = (cur.get("players") or [])[me]
        act = F._mon(mine.get("active"))
        bench = [m for m in (mine.get("bench") or []) if isinstance(m, dict)]

        print(f"\n=== {ep} ref={ref} — {desc}")
        print(f"  СЫРОЙ борд: active id={act.get('id')} hp={act.get('hp')}/{act.get('maxHp')} "
              f"energies={len(act.get('energies') or [])}")
        for j, m in enumerate(bench):
            print(f"    bench[{j}] id={m.get('id')} hp={m.get('hp')}/{m.get('maxHp')} "
                  f"energies={len(m.get('energies') or [])}")
        print("  БЛОК 8:")
        for k in BLOCK8:
            print(f"    {k} = {feats[k]}")

        # инварианты, которые обязаны держаться на всех четырёх кейсах
        if feats["own_bench_healthier_same_exists"] != 1.0:
            print("  [!] ОЖИДАЛСЯ healthier_same_exists=1 (в A3 копия здоровее активной)")
            ok = False
        if feats["own_bench_charged_healthier_same_exists"] == 1.0 and \
           feats["own_bench_best_same_attack_ready"] != 1.0:
            print("  [!] противоречие: charged=1 при attack_ready!=1")
            ok = False
        exp_gain = feats["own_bench_best_same_hp"] - feats["active_hp"]
        if abs(feats["own_bench_best_same_hp_minus_active"] - exp_gain) > 1e-6:
            print("  [!] hp_minus_active не совпадает с hp(best)-hp(active)")
            ok = False

    # негативный контроль: решение БЕЗ подходящей копии на бенче -> SENTINEL/0, не мусор
    print("\n=== негативный контроль (первое решение эпизода, бенч обычно пуст)")
    shard = find_shard(87556853)
    with shard.open() as fh:
        obs0 = json.loads(fh.readline())
    f0 = F.decision_base_features(obs0, ready_cost, atk_costs)
    print(f"  same_id_n={f0['own_bench_same_id_n']} best_hp={f0['own_bench_best_same_hp']} "
          f"charged={f0['own_bench_charged_healthier_same_exists']}")
    if f0["own_bench_same_id_n"] == 0.0 and f0["own_bench_best_same_hp"] != F.SENTINEL:
        print("  [!] при отсутствии копии hp обязан быть SENTINEL")
        ok = False

    print("\nИТОГ:", "OK — блок читает борд верно" if ok else "ЕСТЬ ПРОБЛЕМЫ, чинить до фичеризации")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
