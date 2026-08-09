#!/usr/bin/env python3
"""scout_gate_paired.py — парное чтение гейта для линии scout.

Отвечает на два независимых вопроса одним прибором:
  (а) scout против нашей линии (v076)   -- кто сильнее и в каких матчапах;
  (б) samples=32 против samples=8       -- даёт ли форвард-поиску больше бюджета
                                           хоть что-то сверх шума.

Протокол — `project_gate_stratified_measurement`: пулить сырые счётчики по парам
нельзя (парадокс Симпсона), поэтому каждая пара дифференцируется отдельно, а
пулятся уже разности с обратно-дисперсионными весами. Стенд не принимает сид и
невоспроизводим, поэтому значимость сама по себе не вердикт: нужна ещё
репликация знака на независимых реранах.

Usage:
    .venv/bin/python scripts/scout_gate_paired.py \
        --cand out/scout_gate/fast_s32_r1_core.json out/scout_gate/fast_s32_r2_core.json \
        --base out/scout_gate/scout_s8_r1_core.json out/scout_gate/fast_s8_r2_core.json \
        --title "samples 32 vs 8"

Опционально (Фаза 3.2, выключено по умолчанию, не меняет старое поведение):
    --reweight-by-field-share  -- домножает дельту каждой пары на долю её архетипа в
        свежем поле (docs/gold/FIELD_RECON_2026-08-02.md) перед агрегацией по стратам/
        полю; статистическая механика (обратно-дисперсионные веса, SE, p-value) не
        трогается. Пары без сопоставимого архетипа (напр. scout_deeper_h2h) получают
        вес 1.0. См. _load_field_share_weights() ниже.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

STRATA = {
    "v009a": "crustle", "ext_tusk_crustle": "crustle",
    "v044": "archaludon", "v036_h2h": "archaludon",
    "ext_masamikobayashi_archaludon": "archaludon",
    "ext_maulikgajera_lucario": "lucario", "ext_nursrijan_lucario": "lucario",
    "v017": "alakazam", "v016": "alakazam",
    "ext_prvsiyan_alak_v8": "alakazam", "ext_prvsiyan_alak_v9": "alakazam",
    "v063_h2h": "mirror", "v065_h2h": "mirror", "v072_h2h": "mirror",
    "v073_h2h": "mirror", "v049": "other",
}
# Зеркала — ближайшее к контролю, что есть: обе стороны получают одну и ту же
# правку, поэтому реальный эффект обязан проявиться ВНЕ этой страты.
CONTROL = "mirror"


# ---------------------------------------------------------------------------
# Опциональное домножение дельты на долю архетипа в свежем поле
# (--reweight-by-field-share, выключено по умолчанию).
#
# Источник долей — gate_scout_guard.BAND_MIX/BAND_N: они уже переписаны Фазой 3
# (2026-08-02) под таблицу (а) docs/gold/FIELD_RECON_2026-08-02.md (см. докстринг
# gate_scout_guard.py). Берём готовые константы вместо повторного хардкода тех же
# чисел здесь — один источник правды. Агрегируем по ВСЕМ бэндам, взвешивая BAND_N,
# чтобы получить одну долю на архетип по всему изученному окну (не по бэндам — этот
# гейт сравнивает кандидата против оппонентов безотносительно рейтинг-бэнда).
#
# Пары этого файла (см. STRATA выше) исторически названы иначе, чем CORE_PAIRS в
# gate_scout_guard.py после её Фазы-3 rename ("v044" здесь == "v044_archaludon" там —
# тот же submissions/v044_archaludon_abomasnow, см. out/scout_gate/*_core.json).
# Алиас ниже чинит только это расхождение; остальные исторические варианты ключей
# (v017/v016/ext_prvsiyan_alak_v8/ext_nursrijan_lucario/ext_masamikobayashi_archaludon/
# v036_h2h) уже присутствуют в ARCHETYPE_MAP.values() как есть.
_PAIR_KEY_ALIASES = {"v044": "v044_archaludon"}


def _load_field_share_weights() -> dict[str, float]:
    """pair_key -> доля архетипа в свежем поле (0..1). Пары без сопоставимого
    архетипа (напр. scout_deeper_h2h, v049) отсутствуют в словаре — вызывающий код
    берёт для них дефолт 1.0 (не переразмеривать то, чему нет аналога в поле)."""
    import os
    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    try:
        import gate_scout_guard as gsg
    except Exception as e:
        print(f"[reweight] WARNING: не удалось импортировать gate_scout_guard ({e}); "
              f"доля поля не применяется, все веса=1.0", flush=True)
        return {}

    pair_to_archetype: dict[str, str] = {}
    for archetype, pair_keys in gsg.ARCHETYPE_MAP.items():
        for pk in pair_keys:
            pair_to_archetype[pk] = archetype
    for local_key, canon_key in _PAIR_KEY_ALIASES.items():
        if canon_key in pair_to_archetype and local_key not in pair_to_archetype:
            pair_to_archetype[local_key] = pair_to_archetype[canon_key]

    total_n = sum(gsg.BAND_N.get(bk, 0) for bk in gsg.BAND_MIX)
    archetype_share: dict[str, float] = {}
    if total_n > 0:
        for band_key, mix in gsg.BAND_MIX.items():
            n = gsg.BAND_N.get(band_key, 0)
            for archetype, share in mix.items():
                archetype_share[archetype] = archetype_share.get(archetype, 0.0) + n * share
        archetype_share = {k: v / total_n for k, v in archetype_share.items()}

    return {pk: archetype_share[arche] for pk, arche in pair_to_archetype.items()
            if arche in archetype_share}


def load(paths: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in paths:
        data = json.loads(Path(p).read_text())
        for k, v in (data.get("opponents") or {}).items():
            if v.get("n_games"):
                out.setdefault(k, []).append(v)
    return out


def merge(runs: list[dict]) -> tuple[float, int]:
    """Схлопывает реран-повторы одной пары в WR и суммарное N."""
    n = sum(r["n_games"] for r in runs)
    wr = sum(r["win_rate"] * r["n_games"] for r in runs) / n
    return wr, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--title", default="cand vs base")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reweight-by-field-share", action="store_true",
                     help="Домножить дельту каждой пары на её долю в свежем поле "
                          "(docs/gold/FIELD_RECON_2026-08-02.md, через "
                          "gate_scout_guard.BAND_MIX/BAND_N) перед агрегацией по "
                          "стратам/полю. Веса/SE/p-value не трогаются. Выключено по "
                          "умолчанию — не меняет старое поведение.")
    args = ap.parse_args()

    field_weights: dict[str, float] = {}
    if args.reweight_by_field_share:
        field_weights = _load_field_share_weights()

    c, b = load(args.cand), load(args.base)
    keys = sorted(set(c) & set(b))
    if not keys:
        print("нет общих пар")
        return 1

    print(f"=== {args.title} ===")
    header = f"{'пара':32s} {'страта':11s} {'cand':>16s} {'base':>16s} {'дельта':>10s} {'репл':>6s}"
    if args.reweight_by_field_share:
        header += f" {'вес_поля':>10s}"
    print(header)
    rows = []
    for k in keys:
        pc, nc = merge(c[k])
        pb, nb = merge(b[k])
        var = pc * (1 - pc) / nc + pb * (1 - pb) / nb
        if var <= 0:
            continue
        # Репликация: совпадает ли знак дельты на каждом отдельном реране.
        # Осмысленна ТОЛЬКО при равном числе прогонов с обеих сторон — иначе zip
        # молча обрежет по короткой стороне и выдаст «репликацию» по одному
        # прогону, что читается как подтверждение, которого нет.
        if len(c[k]) != len(b[k]):
            repl = "n/a"
        else:
            signs = set()
            for rc, rb in zip(c[k], b[k]):
                d = rc["win_rate"] - rb["win_rate"]
                signs.add(1 if d > 0 else (-1 if d < 0 else 0))
            repl = "да" if len(signs) == 1 and 0 not in signs else "нет"
        weight = field_weights.get(k, 1.0) if args.reweight_by_field_share else 1.0
        rows.append((k, STRATA.get(k, "?"), pc - pb, var, repl, weight))
        line = (f"{k:32s} {STRATA.get(k,'?'):11s} {pc*100:8.1f}% n={nc:<4d} "
                f"{pb*100:8.1f}% n={nb:<4d} {(pc-pb)*100:+9.1f}пп {repl:>6s}")
        if args.reweight_by_field_share:
            line += f" {weight:10.5f}"
        print(line)

    def pool(rs, reweight: bool = False):
        w = sum(1 / r[3] for r in rs)
        if reweight:
            d = sum((r[2] * r[5]) / r[3] for r in rs) / w
        else:
            d = sum(r[2] / r[3] for r in rs) / w
        se = math.sqrt(1 / w)
        z = d / se
        return d, z, math.erfc(abs(z) / math.sqrt(2))

    print()
    for strat in sorted({r[1] for r in rows}):
        rs = [r for r in rows if r[1] == strat]
        d, z, p = pool(rs, reweight=args.reweight_by_field_share)
        tag = "  <-- КОНТРОЛЬ" if strat == CONTROL else ""
        mark = "ЗНАЧИМО" if p < 0.01 else ""
        print(f"{strat:11s} дельта={d*100:+7.2f}пп  z={z:+6.2f}  p={p:.5f}  {mark}{tag}")

    field = [r for r in rows if r[1] != CONTROL]
    if field:
        d, z, p = pool(field, reweight=args.reweight_by_field_share)
        print(f"\nПОЛЕ (без контроля): {d*100:+.2f}пп  z={z:+.2f}  p={p:.6f}")
        n_repl = sum(1 for r in field if r[4] == "да")
        print(f"репликация знака: {n_repl}/{len(field)} пар")
        if args.reweight_by_field_share:
            print("(дельта каждой пары домножена на долю её архетипа в свежем поле "
                  "перед пулингом — см. docs/gold/FIELD_RECON_2026-08-02.md; веса/SE/"
                  "p-value как статистический механизм не менялись, см. --help)")
        # MEASUREMENT_POLICY: ниже шумового пола грузить нельзя.
        verdict = ("ПРОХОДИТ порог +5пп" if d * 100 >= 5 and p < 0.01
                   else "НЕ проходит порог +5пп (см. docs/gold/MEASUREMENT_POLICY.md)")
        print(f"вердикт: {verdict}")

    if args.out:
        pair_dicts = []
        for r in rows:
            pd = {"key": r[0], "stratum": r[1], "delta": r[2], "repl": r[4]}
            if args.reweight_by_field_share:
                pd["field_weight"] = r[5]
            pair_dicts.append(pd)
        Path(args.out).write_text(json.dumps(
            {"title": args.title, "pairs": pair_dicts},
            ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
