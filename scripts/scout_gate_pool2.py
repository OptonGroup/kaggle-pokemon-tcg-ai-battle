#!/usr/bin/env python3
"""scout_gate_pool2.py — «гейт-пул 2.0» (R3, docs/gold/PLAN_TOP30_20260808.md §1/§3).

НЕ трогает submissions/scout_deeper_p0p1/**, scripts/scout_gate_paired.py,
scripts/gate_scout_guard.py — это отдельный новый инструмент. Физическое исполнение
партий переиспользуется из scripts/gate_scout_guard.py КАК БИБЛИОТЕКИ (импорт по
пути, без правок файла): run_pair()/_load_submission_module()/_resolve_deck()/
_pair_dir_valid()/wilson_ci()/two_prop_ztest()/battle_start/to_observation_class —
тот же движок cg_native (или cg на Linux), тот же spawn ProcessPoolExecutor, тот же
формат per-pair JSON (см. `opponent`/`wins`/`losses`/`draws`/`win_rate`/`n_games`/…).

=== Пул оппонентов ===
НОВЫЕ (подключены в этой сессии из data/external_agents/recon_20260808/b2_competitors/,
код конкурентов НЕ переписан — только скопирован/извлечён как есть в новые каталоги
submissions/external_<name>/, см. ATTRIBUTION в самих каталогах где есть):
  ext2_tetsutani_grimmsnarl  submissions/external_tetsutani_grimmsnarl
                             архетип mirror_grimmsnarl. 800.8 на ладдере (снимок 08-08),
                             полный ML-каскад (strategic_policy + coalition_expert +
                             matchup_router + guards), 188 файлов, БЕЗ numpy/torch —
                             только stdlib + cg.api. ПЕРВЫЙ реальный внешний
                             mirror-спарринг в пуле (раньше mirror = только наши же
                             старые версии v063/v070/v072/v073).
  ext2_jazivxt_codex         submissions/external_jazivxt_codex
                             архетип alakazam. 994.2 на ладдере — верхний живой
                             alakazam в пуле. Единственный из трёх новых с реальным
                             PIMC-поиском (search_begin/search_step, TIME_BUDGET_S=
                             0.80с на решениях с turn>=2 и 3..24 опций) — заметно
                             дороже по wall-clock, см. смоук ниже. Извлечён regex'ом
                             из авторского build-скрипта (MAIN_SOURCE/DECK_SOURCE),
                             байты не изменены.
  ext2_prvsiyan_router       submissions/external_prvsiyan_router_visuals
                             архетип "crustle_mill" — Great Tusk LO + Crustle wall +
                             терминальный милл. 844.4 на ладдере. ВАЖНО: содержит
                             adaptive veto-логику по видимым ID карт, которая
                             распознаёт именно нашу scout-архетипу (Applin/Grookey/
                             Thwackey/Dipplin/Celebi, id 42/89/90/93, у автора помечено
                             как "Festival Lead") и включает отдельный анти-scout
                             механизм — т.е. это НЕ нейтральный спарринг, а адаптивный
                             оппонент именно против нас. "crustle_mill" НЕ входит ни в
                             один вес band_700_799/band_800_plus ниже (см. секцию
                             BAND_WEIGHTS) — попадает в bucket "прочее".

СУЩЕСТВУЮЩИЕ (проверено `_pair_dir_valid` перед каждым прогоном, не хардкожено как
"живые" безусловно):
  ext_maulikgajera_lucario   submissions/external_maulikgajera_lucario  -> lucario
  v044_archaludon            submissions/v044_archaludon_abomasnow      -> archaludon
  ext_masamikobayashi_archaludon submissions/external_masamikobayashi_archaludon
                                                                          -> archaludon

=== Веса бэндов ===
ДОСЛОВНО из ТЗ R3 (которое само привязано к §1 docs/gold/PLAN_TOP30_20260808.md —
строка "Весовать дельту по составу целевого бэнда (§1): для подъёма 600→800 — веса
бэнда 700-799 (lucario .26, alakazam .19, archaludon .17...), для 800+ — mirror .35+."
— полная разбивка получена от оркестратора вместе с ТЗ, см. BAND_WEIGHTS ниже):

  band_700_799: lucario=.26 alakazam=.19 archaludon=.17 dragapult=.08 mirror=.09
                учтено = .79, "прочее" (неучтённое) = .21
  band_800_plus: mirror=.35 alakazam=.21 archaludon=.11
                учтено = .67, "прочее" = .33

В ЭТОМ ПУЛЕ нет оппонента для "dragapult" (в PLAN_TOP30 dragapult числится только
как PROXY greedy_meta_agent-дека в gate_scout_guard.py, не как полноценный агент;
подключать его сюда не входило в ТЗ R3) и "crustle_mill"/ext2_prvsiyan_router не
входит ни в одну из двух табличных раскладок вовсе (ни band_700_799, ни
band_800_plus её не перечисляют) — оба явно уходят в bucket "прочее" и НЕ участвуют
в WR-сумме как компонент. Никакого скрытого дефолта 0.50 (как в gate_scout_guard.py
ARCHETYPE_MAP degraded-default) здесь нет: непокрытая доля бэнда печатается и
пишется в JSON отдельным полем `uncovered_share`, а не подмешивается в WR по 0.50.

=== Метрики ===
На пару: win_rate + Wilson95 CI (g.wilson_ci), как в родительском гейте.
На архетип (если у архетипа >1 оппонента, напр. archaludon): пуленный WR = сумма
win-equivalent / сумма n_games по всем оппонентам архетипа (не невзвешенное среднее
WR, как в старом ARCHETYPE_MAP — тут откровенно пишем "pooled", не "mean").
На бэнд: weighted_wr = Σ share_i × archetype_wr_i по ПОКРЫТЫМ архетипам; variance =
Σ share_i² × var(archetype_wr_i) (независимые компоненты); дополнительно печатается
renormalized-версия (weighted_wr / covered_share) для варианта «а если бы прочее было
средним по покрытым» — оба числа в JSON, помечены явно.
Парная дельта vs --baseline (если передан): per-opponent two_prop_ztest (g.two_prop_
ztest), затем тот же pooled-по-архетипу и weighted-по-бэнду механизм для дельты.

Запуск:
    .venv/bin/python scripts/scout_gate_pool2.py --smoke --workers 4
    .venv/bin/python scripts/scout_gate_pool2.py --n 300 --workers 8
    .venv/bin/python scripts/scout_gate_pool2.py --n 300 --workers 8 \
        --baseline submissions/scout_deeper_p0p1_prev_copy --baseline-label prev
    .venv/bin/python scripts/scout_gate_pool2.py --only ext2_tetsutani_grimmsnarl --n 8 --workers 4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_gate_scout_guard():
    """Импортирует scripts/gate_scout_guard.py как библиотеку под стабильным
    именем модуля "gate_scout_guard" (регистрация в sys.modules ДО exec_module —
    иначе spawn-воркеры не находят модуль при unpickle функций из него и падают
    в 'crashes' без видимой причины на этой машине, проверено эмпирически)."""
    mod_name = "gate_scout_guard"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = os.path.join(ROOT, "scripts", "gate_scout_guard.py")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


g = _load_gate_scout_guard()

GATES_DIR = os.path.join(ROOT, "out", "gate_pool2")

# ---------------------------------------------------------------------------
# Пул оппонентов
# ---------------------------------------------------------------------------

_RAW_POOL = [
    dict(key="ext2_tetsutani_grimmsnarl",
         dir=g.resolve_dir("submissions/external_tetsutani_grimmsnarl"),
         archetype="mirror", note="800.8, новый, полный ML-каскад, PIMC нет"),
    dict(key="ext2_jazivxt_codex",
         dir=g.resolve_dir("submissions/external_jazivxt_codex"),
         archetype="alakazam", note="994.2, новый, PIMC search TIME_BUDGET_S=0.80 -> дороже"),
    dict(key="ext2_prvsiyan_router",
         dir=g.resolve_dir("submissions/external_prvsiyan_router_visuals"),
         archetype="crustle_mill", note="844.4, новый, adaptive anti-scout veto, "
                                        "архетип не входит в BAND_WEIGHTS -> 'прочее'"),
    dict(key="ext_maulikgajera_lucario",
         dir=g.resolve_dir("submissions/external_maulikgajera_lucario"),
         archetype="lucario", note="существующий"),
    dict(key="v044_archaludon",
         dir=g.resolve_dir("submissions/v044_archaludon_abomasnow"),
         archetype="archaludon", note="существующий, наш агент"),
    dict(key="ext_masamikobayashi_archaludon",
         dir=g.resolve_dir("submissions/external_masamikobayashi_archaludon"),
         archetype="archaludon", note="существующий, независимый архаладон"),
]

POOL: list[dict] = []
EXCLUDED: list[dict] = []
for _cfg in _RAW_POOL:
    _ok, _reason = g._pair_dir_valid(_cfg["dir"])
    if _ok:
        POOL.append(_cfg)
    else:
        EXCLUDED.append(dict(key=_cfg["key"], dir=_cfg["dir"], reason=_reason))

ARCHETYPE_TO_KEYS: dict[str, list[str]] = {}
for _cfg in POOL:
    ARCHETYPE_TO_KEYS.setdefault(_cfg["archetype"], []).append(_cfg["key"])

# ---------------------------------------------------------------------------
# Веса бэндов — см. докстринг выше, числа даны оркестратором дословно из §1 PLAN.
# ---------------------------------------------------------------------------

BAND_WEIGHTS: dict[str, dict[str, float]] = {
    "band_700_799": dict(lucario=.26, alakazam=.19, archaludon=.17, dragapult=.08, mirror=.09),
    "band_800_plus": dict(mirror=.35, alakazam=.21, archaludon=.11),
}


# ---------------------------------------------------------------------------
# Прогон пула для одного кандидата
# ---------------------------------------------------------------------------

def run_pool(candidate_dir: str, label: str, n_games: int, seed: int, workers: int,
             only: set[str] | None, out_path: str) -> dict:
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    opponents = existing.get("opponents", {})

    for cfg in POOL:
        key = cfg["key"]
        if only and key not in only:
            continue
        have = opponents.get(key, {})
        if have.get("n_games", 0) >= n_games:
            print(f"[{label}] {key}: already have n_games={have['n_games']} >= {n_games}, skip",
                  flush=True)
            continue
        print(f"[{label}] {key} ({cfg['archetype']}, {cfg['note']}): running n={n_games} "
              f"vs {cfg['dir']} ...", flush=True)
        t0 = time.time()
        stats = g.run_pair(candidate_dir, "sub:" + cfg["dir"], key, n_games, seed, workers)
        dt = time.time() - t0
        s_per_game = dt / max(1, stats["n_games"])
        print(f"[{label}] {key}: done in {dt:.1f}s ({s_per_game:.2f}s/game)  "
              f"WR={stats['win_rate']:.4f} [{stats['wilson95_lo']:.3f},{stats['wilson95_hi']:.3f}]  "
              f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']}  "
              f"avg_len={stats['avg_game_length']:.1f}  "
              f"exc={stats['exc_agent']}/{stats['exc_opp']} illegal={stats['illegal_agent']}/"
              f"{stats['illegal_opp']} timeouts={stats['timeouts']} crashes={stats['crashes']}",
              flush=True)
        opponents[key] = stats
        existing = dict(
            tag=f"pool2_{label}",
            agent=dict(agent_dir=os.path.basename(candidate_dir), label=label,
                       deck=os.path.join(candidate_dir, "deck.csv")),
            excluded=EXCLUDED,
            opponents=opponents,
            generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(existing, f, indent=2)

    return existing


# ---------------------------------------------------------------------------
# Агрегация: архетип -> pooled WR/var, бэнд -> weighted WR/var, дельта vs baseline
# ---------------------------------------------------------------------------

def _pooled_archetype(archetype: str, opponents: dict) -> dict | None:
    keys = ARCHETYPE_TO_KEYS.get(archetype, [])
    we_total = 0.0
    n_total = 0
    per_opp = {}
    for k in keys:
        o = opponents.get(k)
        if not o or not o.get("n_games"):
            continue
        we = o["wins"] + 0.5 * o["draws"]
        we_total += we
        n_total += o["n_games"]
        per_opp[k] = dict(win_rate=o["win_rate"], n_games=o["n_games"])
    if n_total == 0:
        return None
    wr = we_total / n_total
    var = wr * (1 - wr) / n_total
    return dict(win_rate=wr, n_games=n_total, variance=var, per_opponent=per_opp)


def compute_archetype_table(opponents: dict) -> dict:
    return {arche: _pooled_archetype(arche, opponents) for arche in ARCHETYPE_TO_KEYS}


def compute_band(band_key: str, archetype_table: dict) -> dict:
    mix = BAND_WEIGHTS[band_key]
    covered_share = 0.0
    weighted_wr = 0.0
    weighted_var = 0.0
    components = {}
    uncovered = {}
    for arche, share in mix.items():
        a = archetype_table.get(arche)
        if a is None:
            uncovered[arche] = share
            continue
        covered_share += share
        weighted_wr += share * a["win_rate"]
        weighted_var += (share ** 2) * a["variance"]
        components[arche] = dict(share=share, win_rate=a["win_rate"], n_games=a["n_games"])
    uncovered_share = 1.0 - covered_share  # включает и "прочее" вне таблицы, и дыры внутри неё
    renorm_wr = (weighted_wr / covered_share) if covered_share > 0 else None
    renorm_var = (weighted_var / (covered_share ** 2)) if covered_share > 0 else None
    return dict(
        band=band_key,
        components=components,
        uncovered_in_table=uncovered,       # архетипы таблицы без живого оппонента в пуле
        covered_share=covered_share,        # доля бэнда, реально взвешенная числами
        uncovered_share=uncovered_share,    # "прочее" + дыры — НЕ участвует в WR
        weighted_wr_raw=weighted_wr,        # сумма без перенормировки (нижняя граница)
        weighted_wr_renormalized=renorm_wr, # то же, поделенное на covered_share
        weighted_var_raw=weighted_var,
        weighted_var_renormalized=renorm_var,
    )


def compute_band_delta(band_key: str, cand_arch: dict, base_arch: dict) -> dict:
    mix = BAND_WEIGHTS[band_key]
    covered_share = 0.0
    delta = 0.0
    var = 0.0
    per_arch = {}
    for arche, share in mix.items():
        ca = cand_arch.get(arche)
        ba = base_arch.get(arche)
        if ca is None or ba is None:
            continue
        covered_share += share
        d = ca["win_rate"] - ba["win_rate"]
        v = ca["variance"] + ba["variance"]
        delta += share * d
        var += (share ** 2) * v
        z, p = g.two_prop_ztest(ca["win_rate"] * ca["n_games"], ca["n_games"],
                                 ba["win_rate"] * ba["n_games"], ba["n_games"])
        per_arch[arche] = dict(share=share, cand_wr=ca["win_rate"], base_wr=ba["win_rate"],
                                delta=d, z=z, p_value=p)
    se = math.sqrt(var) if var > 0 else None
    z_band = delta / se if se else None
    p_band = math.erfc(abs(z_band) / math.sqrt(2)) if z_band is not None else None
    return dict(band=band_key, per_archetype=per_arch, covered_share=covered_share,
                delta_raw=delta, se_raw=se, z_raw=z_band, p_value_raw=p_band)


def print_report(cand_label: str, cand_opponents: dict, archetype_table: dict,
                  base_label: str | None, base_opponents: dict | None,
                  base_archetype_table: dict | None) -> dict:
    print(f"\n=== PER-OPPONENT ({cand_label}) ===")
    for cfg in POOL:
        o = cand_opponents.get(cfg["key"])
        if not o:
            print(f"  {cfg['key']:32s} [{cfg['archetype']:12s}] no data yet")
            continue
        print(f"  {cfg['key']:32s} [{cfg['archetype']:12s}] WR={o['win_rate']:.4f} "
              f"[{o['wilson95_lo']:.3f},{o['wilson95_hi']:.3f}] n={o['n_games']} "
              f"W/L/D={o['wins']}/{o['losses']}/{o['draws']} "
              f"avg_len={o['avg_game_length']:.1f} "
              f"crashes={o['crashes']} illegal={o['illegal_agent']}/{o['illegal_opp']}")

    print(f"\n=== ARCHETYPE POOLED WR ({cand_label}) ===")
    for arche, a in sorted(archetype_table.items()):
        if a is None:
            print(f"  {arche:12s} no data")
            continue
        lo, hi = g.wilson_ci(a["win_rate"] * a["n_games"], a["n_games"])
        print(f"  {arche:12s} WR={a['win_rate']:.4f} [{lo:.3f},{hi:.3f}] n={a['n_games']} "
              f"(from {list(a['per_opponent'].keys())})")

    band_results = {}
    print(f"\n=== BAND-WEIGHTED WR ({cand_label}) ===")
    for band_key in BAND_WEIGHTS:
        b = compute_band(band_key, archetype_table)
        band_results[band_key] = b
        se_raw = math.sqrt(b["weighted_var_raw"]) if b["weighted_var_raw"] > 0 else 0.0
        print(f"  {band_key:16s} covered_share={b['covered_share']*100:5.1f}%  "
              f"weighted_wr_raw={b['weighted_wr_raw']*100:5.1f}% (+/-{1.96*se_raw*100:4.1f}pp 95%CI, "
              f"lower-bound, uncovered treated as 0-contribution)")
        renorm = b["weighted_wr_renormalized"]
        if renorm is not None:
            se_r = math.sqrt(b["weighted_var_renormalized"])
            print(f"  {'':16s} weighted_wr_renormalized={renorm*100:5.1f}% "
                  f"(+/-{1.96*se_r*100:4.1f}pp, среди покрытых {b['covered_share']*100:.1f}% бэнда)")
        if b["uncovered_in_table"]:
            print(f"  {'':16s} НЕ покрыто (в самой таблице бэнда, нет живого оппонента): "
                  f"{b['uncovered_in_table']}")
        print(f"  {'':16s} 'прочее'/вне таблицы: {b['uncovered_share']*100:.1f}% "
              f"(archetypes вне BAND_WEIGHTS[{band_key}], напр. crustle_mill)")

    out = dict(candidate_label=cand_label, archetype_table=archetype_table, bands=band_results)

    if base_label and base_opponents is not None and base_archetype_table is not None:
        print(f"\n=== DELTA vs BASELINE ({cand_label} - {base_label}) ===")
        for arche in sorted(set(archetype_table) | set(base_archetype_table)):
            ca = archetype_table.get(arche)
            ba = base_archetype_table.get(arche)
            if not ca or not ba:
                continue
            d = ca["win_rate"] - ba["win_rate"]
            z, p = g.two_prop_ztest(ca["win_rate"] * ca["n_games"], ca["n_games"],
                                     ba["win_rate"] * ba["n_games"], ba["n_games"])
            flag = "SIGNIFICANT" if p < 0.05 else ""
            print(f"  {arche:12s} cand={ca['win_rate']*100:5.1f}%(n={ca['n_games']}) "
                  f"base={ba['win_rate']*100:5.1f}%(n={ba['n_games']}) "
                  f"delta={d*100:+5.1f}pp z={z:+.2f} p={p:.4f} {flag}")

        band_deltas = {}
        print(f"\n=== BAND-WEIGHTED DELTA vs BASELINE ===")
        for band_key in BAND_WEIGHTS:
            bd = compute_band_delta(band_key, archetype_table, base_archetype_table)
            band_deltas[band_key] = bd
            sig = "p<0.01 SIGNIFICANT" if (bd["p_value_raw"] is not None and bd["p_value_raw"] < 0.01) else ""
            print(f"  {band_key:16s} covered_share={bd['covered_share']*100:5.1f}%  "
                  f"delta={bd['delta_raw']*100:+5.1f}pp  "
                  f"z={bd['z_raw']:+.2f}  p={bd['p_value_raw']:.4f}  {sig}"
                  if bd["z_raw"] is not None else
                  f"  {band_key:16s} нет перекрытия покрытых архетипов между cand/baseline")
        out["baseline_label"] = base_label
        out["baseline_archetype_table"] = base_archetype_table
        out["band_deltas"] = band_deltas

    if EXCLUDED:
        print("\n=== EXCLUDED FROM POOL (invalid dir) ===")
        for e in EXCLUDED:
            print(f"  {e['key']}: {e['reason']}")

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="submissions/scout_deeper_p0p1")
    ap.add_argument("--candidate-label", default="candidate")
    ap.add_argument("--baseline", default=None, help="Каталог baseline-сабмита (опционально)")
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--n", type=int, default=300, help="Игр на оппонента")
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="Только эти ключи пула (через запятую)")
    ap.add_argument("--smoke", action="store_true", help="n=8, sanity-check")
    ap.add_argument("--out-dir", default=None,
                     help="По умолчанию out/gate_pool2/<timestamp>/")
    args = ap.parse_args()

    n_games = 8 if args.smoke else args.n
    only = {x.strip() for x in args.only.split(",")} if args.only else None

    out_dir = args.out_dir or os.path.join(GATES_DIR, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[init] pool={[c['key'] for c in POOL]}", flush=True)
    if EXCLUDED:
        print(f"[init] excluded={EXCLUDED}", flush=True)
    print(f"[init] out_dir={out_dir}  n={n_games}  workers={args.workers}  seed={args.seed}",
          flush=True)

    cand_dir = g.resolve_dir(args.candidate)
    if not os.path.isdir(cand_dir):
        print(f"[init] FATAL: candidate dir not found: {cand_dir}")
        return 2
    cand_out = os.path.join(out_dir, f"{args.candidate_label}_core.json")
    cand_data = run_pool(cand_dir, args.candidate_label, n_games, args.seed, args.workers,
                          only, cand_out)
    cand_opponents = cand_data.get("opponents", {})
    cand_archetypes = compute_archetype_table(cand_opponents)

    base_opponents = None
    base_archetypes = None
    base_label = None
    if args.baseline:
        base_dir = g.resolve_dir(args.baseline)
        if not os.path.isdir(base_dir):
            print(f"[init] WARNING: baseline dir not found: {base_dir} — skipping delta section")
        else:
            base_label = args.baseline_label
            base_out = os.path.join(out_dir, f"{base_label}_core.json")
            # baseline получает СВОИ сиды (seed+1) — намеренно другая, независимая
            # выборка партий, не парная (движок всё равно не принимает сид на игру,
            # см. project_rating_noise/gate_scout_guard.py комментарий в run_pair).
            base_data = run_pool(base_dir, base_label, n_games, args.seed + 1, args.workers,
                                  only, base_out)
            base_opponents = base_data.get("opponents", {})
            base_archetypes = compute_archetype_table(base_opponents)

    summary = print_report(args.candidate_label, cand_opponents, cand_archetypes,
                            base_label, base_opponents, base_archetypes)
    summary["pool"] = POOL
    summary["excluded"] = EXCLUDED
    summary["band_weights_source"] = ("ТЗ R3 / docs/gold/PLAN_TOP30_20260808.md §1 "
                                       "(числа переданы оркестратором дословно)")
    summary["band_weights"] = BAND_WEIGHTS
    summary["n_games_per_opponent"] = n_games
    summary["seed"] = args.seed
    summary["generated"] = time.strftime("%Y-%m-%d %H:%M:%S")

    band_out_path = os.path.join(out_dir, "band_summary.json")
    with open(band_out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[done] band summary written to {band_out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
