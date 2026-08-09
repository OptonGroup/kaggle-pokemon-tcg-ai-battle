#!/usr/bin/env python3
"""r4_weight_search.py — R4: координатный спуск по весам policy.json поверх
эталона submissions/gatecfg_deeper_p0p1 (см. docs/gold/PLAN_TOP30_20260808.md §1,
рычаг R4; полный проект передан оркестратором в ТЗ этой сессии).

НЕ трогает submissions/scout_deeper_p0p1/**, submissions/scout_deeper/**,
submissions/gatecfg_*/**, scripts/gate_scout_guard.py, scripts/scout_gate_paired.py,
scripts/scout_gate_pool2.py — все они используются ТОЛЬКО на чтение / как библиотека
(importlib, тот же приём, что scout_gate_pool2.py уже применяет к gate_scout_guard.py).

=== Геном (29 непрерывных чисел, см. ТЗ weights_chosen) ===
  1. search.ucb_c                                    — абсолютное значение
  8. main_priority "приоритетные очки" по 8 типам опций (main.py:56,177-181) —
     НЕ перестановка: очки непрерывные, реальный MAIN_PRIORITY = argsort(-очки).
     argsort даёт корректный список без repair-оператора для перестановок.
  8. card_delta[id] для id из {42,89,100,343,90,93,240,1} — аддитивный сдвиг,
     применяется к записи id ВО ВСЕХ контекстах, где она встречается, потом clip[0,1].
  1. trainer_block_bias — аддитивный сдвиг для id из WIDE_IDS (13 "широких" карт
     колоды 1080..1245) во всех контекстах, где встречаются.
  1. attack_table_bias — аддитивный сдвиг для всех 35 attackId в ctx0["attack"].
 10. priors: (ctx,поле) для (0,card)(0,attack)(0,type)(7,card)(4,card)(3,card)
     (5,card)(9,card)(1,card)(8,card) — аддитивный сдвиг к дефолтному score_option().
Итого 1+8+8+1+1+10 = 29 параметров.

=== Кандидат без дублирования каталога агента ===
На каждую особь создаётся лёгкая рабочая директория out/r4_weight_search/<run>/work/<tag>/
с symlink'ами на main.py/cg/deck.csv/meta_decks.json/value_model.json/value_model_arch.json
из submissions/gatecfg_deeper_p0p1 (эталон НЕ копируется, НЕ меняется) и СВОИМ файлом
policy.json (единственный реальный файл на кандидата, ~15КБ). main.py вычисляет
AGENT_DIR = dirname(__file__) от пути символической ссылки, поэтому подхватывает
именно локальный policy.json, а cg/ (5.2МБ) — общий, без копирования.

=== Два яруса + финальный лок ===
Ярус 0 (скан, N=40×4опп, без пары): дельта band-weighted WR кандидата против
ФИКСИРОВАННОГО REFERENCE_WR (из out/gate_pool2/20260809c_scaled/BASE/BASE_core.json,
N=200/опп) > 8пп -> идёт в ярус 1. Дёшево, но не различает <8-10пп эффект.
Ярус 1 (парное подтверждение survivors, N=100×4опп): кандидат и ЖИВОЙ baseline
(=submissions/gatecfg_deeper_p0p1, эталон, БЕЗ изменений) прогоняются ПОДРЯД в
одном вызове с ОДИНАКОВЫМ cell_id/seed на пару candidate/baseline (движок сам
сид не принимает — "общий набор сидов" здесь означает общий python-side
random.seed() и отсутствие временного дрейфа между прогонами, не побитовое
совпадение партий). z>=1.96 (p<0.05, band-weighted two-prop z-test) -> принять
шаг координатного спуска в накопленный genome.
Финал (один раз, после спуска или по истечении бюджета): 2 независимых реранa
N=200×4опп кандидат-vs-baseline + holdout N=150×5 (независимые реализации тех же
архетипов + 1 off-target). Holdout правило: z<-1.0 хоть на одном -> откат к
предыдущему принятому genome (см. FALLBACK_GENOME в state.json).

Запуск:
    .venv/bin/python scripts/r4_weight_search.py --smoke
    .venv/bin/python scripts/r4_weight_search.py --run r4_20260809 --budget-hours 18 --workers 4
    .venv/bin/python scripts/r4_weight_search.py --run r4_20260809 --resume   # продолжить после остановки
"""
from __future__ import annotations

import argparse
import copy
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

OUT_ROOT = os.path.join(ROOT, "out", "r4_weight_search")
BASE_AGENT_DIR = g.resolve_dir("submissions/gatecfg_deeper_p0p1")
BASE_CORE_JSON = os.path.join(ROOT, "out", "gate_pool2", "20260809c_scaled", "BASE",
                               "BASE_core.json")

with open(os.path.join(BASE_AGENT_DIR, "policy.json")) as _fh:
    BASE_POLICY = json.load(_fh)

# ---------------------------------------------------------------------------
# Пул фитнеса (ярус 0/1) — те же 4 оппонента, что band_700_799 в
# scripts/scout_gate_pool2.py:56-59/129-146 (протяжённый файл, не импортируем
# целиком чтобы не тащить его POOL-инициализацию — дублируем 4 директории явно,
# числа весов оттуда же процитированы дословно).
# ---------------------------------------------------------------------------
FITNESS_POOL = {
    "mirror": g.resolve_dir("submissions/external_tetsutani_grimmsnarl"),
    "alakazam": g.resolve_dir("submissions/external_jazivxt_codex"),
    "lucario": g.resolve_dir("submissions/external_maulikgajera_lucario"),
    "archaludon": g.resolve_dir("submissions/v044_archaludon_abomasnow"),
}
# band_700_799 raw (scripts/scout_gate_pool2.py:56-59): lucario=.26 alakazam=.19
# archaludon=.17 dragapult=.08(нет живого опп, исключён) mirror=.09 -> нормировка
# на сумму присутствующих (.26+.19+.17+.09=.71).
_RAW_W = {"lucario": .26, "alakazam": .19, "archaludon": .17, "mirror": .09}
_SUM_W = sum(_RAW_W.values())
NORM_W = {k: v / _SUM_W for k, v in _RAW_W.items()}

# opponent-key -> BASE_core.json cell_id (см. out/gate_pool2/20260809c_scaled/BASE)
BASE_CORE_KEY = {
    "mirror": "ext2_tetsutani_grimmsnarl",
    "alakazam": "ext2_jazivxt_codex",
    "lucario": "ext_maulikgajera_lucario",
    "archaludon": "v044_archaludon",
}

with open(BASE_CORE_JSON) as _fh:
    _base_core = json.load(_fh)
REFERENCE_PER_OPP = {k: _base_core["opponents"][ck]["win_rate"] for k, ck in BASE_CORE_KEY.items()}
REFERENCE_WR_BAND = sum(NORM_W[k] * REFERENCE_PER_OPP[k] for k in NORM_W)

TIER0_N = 40
TIER0_THRESHOLD = 0.08  # 8пп, см. ТЗ fitness_design
TIER1_N = 100
TIER1_Z = 1.96
TIER2_N = 200
TIER2_HOLDOUT_N = 150
TIER2_HOLDOUT_Z_REJECT = -1.0

# ---------------------------------------------------------------------------
# Holdout (только финальный лок — НЕ участвуют в фитнесе яруса 0/1)
# ---------------------------------------------------------------------------
HOLDOUT_POOL = {
    "alakazam_holdout": g.resolve_dir("submissions/external_prvsiyan_alakazam_v9"),
    "lucario_holdout": g.resolve_dir("submissions/external_nursrijan_lucario"),
    "mirror_holdout": g.resolve_dir("submissions/external_prvsiyan_grimmsnarl_hybrid"),
    "archaludon_holdout": g.resolve_dir("submissions/external_masamikobayashi_archaludon"),
    "off_target_crustle_mill": g.resolve_dir("submissions/external_prvsiyan_router_visuals"),
}

# ---------------------------------------------------------------------------
# Геном
# ---------------------------------------------------------------------------
TYPE_IDS = list(BASE_POLICY["main_priority"])  # [9,7,10,8,11,13,14,12] — фикс. набор
CARD_DELTA_IDS = [42, 89, 100, 343, 90, 93, 240, 1]
WIDE_IDS = [1080, 1086, 1094, 1122, 1129, 1137, 1152, 1175, 1182, 1211, 1227, 1245]
PRIOR_SPECS = [
    (0, "card_prior"), (0, "attack_prior"), (0, "type_prior"),
    (7, "card_prior"), (4, "card_prior"), (3, "card_prior"),
    (5, "card_prior"), (9, "card_prior"), (1, "card_prior"), (8, "card_prior"),
]


def _mp_name(t: int) -> str:
    return f"mp_{t}"


def _cd_name(cid: int) -> str:
    return f"card_delta_{cid}"


def _prior_name(ctx: int, field: str) -> str:
    return f"prior_{ctx}_{field}"


# Порядок обхода координатного спуска — по частоте чтения (см. ТЗ weights_chosen):
# ucb_c (62% решений на каждом сэмпле) -> main_priority (миллионы чтений в playout) ->
# ctx0 приоры (самый частый контекст) -> card_delta (7-20% контекстов каждый) ->
# ctx7/4/3/5/9/1/8 приоры (по убыванию частоты контекста) -> trainer/attack biases
# (агрегаты, наименее прицельные).
PARAM_ORDER: list[str] = (
    ["ucb_c"]
    + [_mp_name(t) for t in TYPE_IDS]
    + [_prior_name(0, "card_prior"), _prior_name(0, "attack_prior"), _prior_name(0, "type_prior")]
    + [_cd_name(cid) for cid in CARD_DELTA_IDS]
    + [_prior_name(7, "card_prior"), _prior_name(4, "card_prior"), _prior_name(3, "card_prior"),
       _prior_name(5, "card_prior"), _prior_name(9, "card_prior"), _prior_name(1, "card_prior"),
       _prior_name(8, "card_prior")]
    + ["trainer_block_bias", "attack_table_bias"]
)
assert len(PARAM_ORDER) == 1 + 8 + 8 + 1 + 1 + 10 == 29, len(PARAM_ORDER)

PARAM_STEP = {}
PARAM_BOUNDS = {}
PARAM_STEP["ucb_c"] = 0.05
PARAM_BOUNDS["ucb_c"] = (0.0, 1.0)
for _t in TYPE_IDS:
    PARAM_STEP[_mp_name(_t)] = 1.5
    PARAM_BOUNDS[_mp_name(_t)] = (-5.0, 15.0)
for _cid in CARD_DELTA_IDS:
    PARAM_STEP[_cd_name(_cid)] = 0.04
    PARAM_BOUNDS[_cd_name(_cid)] = (-0.35, 0.35)
PARAM_STEP["trainer_block_bias"] = 0.04
PARAM_BOUNDS["trainer_block_bias"] = (-0.35, 0.35)
PARAM_STEP["attack_table_bias"] = 0.04
PARAM_BOUNDS["attack_table_bias"] = (-0.35, 0.35)
for _ctx, _field in PRIOR_SPECS:
    PARAM_STEP[_prior_name(_ctx, _field)] = 0.04
    PARAM_BOUNDS[_prior_name(_ctx, _field)] = (-0.35, 0.35)


def init_genome() -> dict:
    genome = {"ucb_c": BASE_POLICY["search"].get("ucb_c", 0.0)}
    n = len(TYPE_IDS)
    for i, t in enumerate(TYPE_IDS):
        genome[_mp_name(t)] = float(n - i)  # убывающие очки воспроизводят baseline-порядок
    for cid in CARD_DELTA_IDS:
        genome[_cd_name(cid)] = 0.0
    genome["trainer_block_bias"] = 0.0
    genome["attack_table_bias"] = 0.0
    for ctx, field in PRIOR_SPECS:
        genome[_prior_name(ctx, field)] = 0.0
    return genome


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def apply_genome(base_policy: dict, genome: dict) -> dict:
    pol = copy.deepcopy(base_policy)
    pol.setdefault("search", {})["ucb_c"] = genome["ucb_c"]
    mp_scores = {t: genome[_mp_name(t)] for t in TYPE_IDS}
    pol["main_priority"] = sorted(TYPE_IDS, key=lambda t: (-mp_scores[t], TYPE_IDS.index(t)))

    ctxs = pol["contexts"]
    for cdata in ctxs.values():
        card = cdata.get("card")
        if not card:
            continue
        for cid in CARD_DELTA_IDS:
            key = str(cid)
            if key in card:
                card[key] = _clip01(card[key] + genome[_cd_name(cid)])
        for wid in WIDE_IDS:
            key = str(wid)
            if key in card:
                card[key] = _clip01(card[key] + genome["trainer_block_bias"])

    ctx0 = ctxs.get("0", {})
    atk = ctx0.get("attack")
    if atk:
        for k in list(atk.keys()):
            atk[k] = _clip01(atk[k] + genome["attack_table_bias"])

    for ctx, field in PRIOR_SPECS:
        c = ctxs.get(str(ctx))
        if c is None or field not in c:
            continue
        c[field] = _clip01(c[field] + genome[_prior_name(ctx, field)])

    return pol


def genome_diff(base: dict, other: dict) -> dict:
    return {k: other[k] for k in base if base.get(k) != other.get(k)}


def mutate_value(pname: str, current: float, direction: int) -> float:
    lo, hi = PARAM_BOUNDS[pname]
    step = PARAM_STEP[pname]
    return max(lo, min(hi, current + direction * step))


# ---------------------------------------------------------------------------
# Кандидат-директория (symlink-скелет + собственный policy.json)
# ---------------------------------------------------------------------------
_SHARED_FILES = ("main.py", "cg", "deck.csv", "meta_decks.json", "value_model.json",
                  "value_model_arch.json")


def make_candidate_dir(genome: dict, tag: str, work_root: str) -> str:
    d = os.path.join(work_root, tag)
    os.makedirs(d, exist_ok=True)
    for fname in _SHARED_FILES:
        link = os.path.join(d, fname)
        target = os.path.join(BASE_AGENT_DIR, fname)
        if os.path.lexists(link):
            if os.path.islink(link) and os.readlink(link) == target:
                continue
            os.remove(link)
        os.symlink(target, link)
    policy = apply_genome(BASE_POLICY, genome)
    with open(os.path.join(d, "policy.json"), "w") as f:
        json.dump(policy, f)
    return d


# ---------------------------------------------------------------------------
# JSONL-лог
# ---------------------------------------------------------------------------

def log_jsonl(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Ярус 0 / 1 / 2
# ---------------------------------------------------------------------------

def _slim(stats: dict) -> dict:
    return {k: stats[k] for k in ("win_rate", "n_games", "wins", "losses", "draws",
                                   "avg_game_length", "crashes", "illegal_agent",
                                   "illegal_opp", "exc_agent", "exc_opp", "timeouts")}


def eval_tier0(genome: dict, tag: str, work_root: str, workers: int, seed: int,
               n: int = TIER0_N) -> dict:
    cand_dir = make_candidate_dir(genome, tag, work_root)
    per_opp = {}
    t0 = time.time()
    for key, opp_dir in FITNESS_POOL.items():
        stats = g.run_pair(cand_dir, "sub:" + opp_dir, f"t0_{tag}_{key}", n, seed, workers)
        per_opp[key] = _slim(stats)
    dt = time.time() - t0
    wr_band = sum(NORM_W[k] * per_opp[k]["win_rate"] for k in NORM_W)
    var = sum((NORM_W[k] ** 2) * (per_opp[k]["win_rate"] * (1 - per_opp[k]["win_rate"])
                                   / per_opp[k]["n_games"]) for k in NORM_W)
    se = math.sqrt(var) if var > 0 else 0.0
    delta = wr_band - REFERENCE_WR_BAND
    passed = delta > TIER0_THRESHOLD
    return dict(tier=0, tag=tag, cand_dir=cand_dir, per_opp=per_opp, wr_band=wr_band,
                reference_wr_band=REFERENCE_WR_BAND, delta=delta, se=se, passed=passed,
                elapsed_s=dt, n_per_opp=n, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))


def eval_tier1(genome: dict, tag: str, work_root: str, workers: int, seed: int,
               n: int = TIER1_N) -> dict:
    cand_dir = make_candidate_dir(genome, tag, work_root)
    per_opp_cand, per_opp_base, per_arch = {}, {}, []
    t0 = time.time()
    delta_sum, var_sum = 0.0, 0.0
    for key, opp_dir in FITNESS_POOL.items():
        # ОДИНАКОВЫЙ cell_id/seed для candidate и baseline на этой паре -> одинаковая
        # последовательность python-side random.seed(seed) по игре (движок сид не
        # принимает, см. gate_scout_guard.py:487-500). Это и есть "общий набор сидов"
        # в пределах того, что стенд физически может дать.
        cell_id = f"t1_{tag}_{key}"
        stats_c = g.run_pair(cand_dir, "sub:" + opp_dir, cell_id, n, seed, workers)
        stats_b = g.run_pair(BASE_AGENT_DIR, "sub:" + opp_dir, cell_id, n, seed, workers)
        per_opp_cand[key] = _slim(stats_c)
        per_opp_base[key] = _slim(stats_b)
        wc = NORM_W[key]
        d = stats_c["win_rate"] - stats_b["win_rate"]
        z, p = g.two_prop_ztest(stats_c["win_rate"] * stats_c["n_games"], stats_c["n_games"],
                                 stats_b["win_rate"] * stats_b["n_games"], stats_b["n_games"])
        vca = stats_c["win_rate"] * (1 - stats_c["win_rate"]) / stats_c["n_games"]
        vba = stats_b["win_rate"] * (1 - stats_b["win_rate"]) / stats_b["n_games"]
        delta_sum += wc * d
        var_sum += (wc ** 2) * (vca + vba)
        per_arch.append(dict(key=key, cand_wr=stats_c["win_rate"], base_wr=stats_b["win_rate"],
                              delta=d, z=z, p_value=p))
    dt = time.time() - t0
    se = math.sqrt(var_sum) if var_sum > 0 else 0.0
    z_band = delta_sum / se if se > 0 else 0.0
    p_band = math.erfc(abs(z_band) / math.sqrt(2))
    passed = z_band >= TIER1_Z
    return dict(tier=1, tag=tag, cand_dir=cand_dir, per_opp_cand=per_opp_cand,
                per_opp_base=per_opp_base, per_archetype=per_arch, delta_band=delta_sum,
                se_band=se, z_band=z_band, p_value=p_band, passed=passed, elapsed_s=dt,
                n_per_opp=n, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))


def eval_tier2(genome: dict, work_root: str, workers: int, base_seed: int) -> dict:
    """Финальный лок: 2 независимых реранa N=200x4 fitness-pool + holdout N=150x5."""
    cand_dir = make_candidate_dir(genome, "final", work_root)
    reruns = []
    for rerun_i in range(2):
        seed = base_seed + 7919 * (rerun_i + 1)
        per_opp_cand, per_opp_base, per_arch = {}, {}, []
        delta_sum, var_sum = 0.0, 0.0
        for key, opp_dir in FITNESS_POOL.items():
            cell_id = f"t2_r{rerun_i}_{key}"
            stats_c = g.run_pair(cand_dir, "sub:" + opp_dir, cell_id, TIER2_N, seed, workers)
            stats_b = g.run_pair(BASE_AGENT_DIR, "sub:" + opp_dir, cell_id, TIER2_N, seed, workers)
            per_opp_cand[key] = _slim(stats_c)
            per_opp_base[key] = _slim(stats_b)
            wc = NORM_W[key]
            d = stats_c["win_rate"] - stats_b["win_rate"]
            z, p = g.two_prop_ztest(stats_c["win_rate"] * stats_c["n_games"], stats_c["n_games"],
                                     stats_b["win_rate"] * stats_b["n_games"], stats_b["n_games"])
            vca = stats_c["win_rate"] * (1 - stats_c["win_rate"]) / stats_c["n_games"]
            vba = stats_b["win_rate"] * (1 - stats_b["win_rate"]) / stats_b["n_games"]
            delta_sum += wc * d
            var_sum += (wc ** 2) * (vca + vba)
            per_arch.append(dict(key=key, cand_wr=stats_c["win_rate"], base_wr=stats_b["win_rate"],
                                  delta=d, z=z, p_value=p))
        se = math.sqrt(var_sum) if var_sum > 0 else 0.0
        z_band = delta_sum / se if se > 0 else 0.0
        p_band = math.erfc(abs(z_band) / math.sqrt(2))
        reruns.append(dict(rerun=rerun_i, per_opp_cand=per_opp_cand, per_opp_base=per_opp_base,
                            per_archetype=per_arch, delta_band=delta_sum, se_band=se,
                            z_band=z_band, p_value=p_band))

    # Комбинированный p (Stouffer, 2 независимых рерана) — соответствует правилу
    # ★★★Дисциплина измерения: >=2 реранa + p<0.01 combined.
    zs = [r["z_band"] for r in reruns]
    z_combined = sum(zs) / math.sqrt(len(zs))
    p_combined = math.erfc(abs(z_combined) / math.sqrt(2))
    both_positive = all(r["delta_band"] > 0 for r in reruns)
    fitness_pool_pass = both_positive and p_combined < 0.01

    holdout = {}
    holdout_reject = False
    for key, opp_dir in HOLDOUT_POOL.items():
        cell_id = f"t2_holdout_{key}"
        stats_c = g.run_pair(cand_dir, "sub:" + opp_dir, cell_id, TIER2_HOLDOUT_N,
                              base_seed + 424242, workers)
        stats_b = g.run_pair(BASE_AGENT_DIR, "sub:" + opp_dir, cell_id, TIER2_HOLDOUT_N,
                              base_seed + 424242, workers)
        z, p = g.two_prop_ztest(stats_c["win_rate"] * stats_c["n_games"], stats_c["n_games"],
                                 stats_b["win_rate"] * stats_b["n_games"], stats_b["n_games"])
        rec = dict(cand=_slim(stats_c), base=_slim(stats_b), z=z, p_value=p)
        holdout[key] = rec
        if key != "off_target_crustle_mill" and z < TIER2_HOLDOUT_Z_REJECT:
            holdout_reject = True

    promote = fitness_pool_pass and not holdout_reject
    return dict(tier=2, cand_dir=cand_dir, reruns=reruns, z_combined=z_combined,
                p_combined=p_combined, fitness_pool_pass=fitness_pool_pass, holdout=holdout,
                holdout_reject=holdout_reject, promote=promote,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))


# ---------------------------------------------------------------------------
# Состояние / резюмирование
# ---------------------------------------------------------------------------

def load_state(run_dir: str) -> dict | None:
    p = os.path.join(run_dir, "state.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def save_state(run_dir: str, state: dict) -> None:
    p = os.path.join(run_dir, "state.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# Основной цикл координатного спуска
# ---------------------------------------------------------------------------

def run_search(run_dir: str, budget_hours: float, workers: int, max_passes: int,
               param_order: list[str] | None = None) -> None:
    os.makedirs(run_dir, exist_ok=True)
    work_root = os.path.join(run_dir, "work")
    os.makedirs(work_root, exist_ok=True)
    log_path = os.path.join(run_dir, "eval_log.jsonl")

    # [2026-08-09, диагностика R4 замедления] Реальная цена оценки яруса 0 при N=40
    # оказалась ~10 мин, тир1 ~45 мин (см. отчёт в чате/докстринг ниже) — полный проход
    # по 29 параметрам не помещается в 12ч. active_param_order позволяет сузить геном до
    # приоритетных параметров (ucb_c + main_priority) без изменения структуры остального
    # кода; по умолчанию (param_order=None) используется полный PARAM_ORDER — поведение
    # не меняется для существующих вызовов.
    active_param_order = param_order if param_order is not None else PARAM_ORDER

    state = load_state(run_dir)
    if state is None:
        state = dict(genome=init_genome(), accepted=[], pass_num=1, param_idx=0,
                     start_time=time.time(), eval_counter=0, tier2=None)
        save_state(run_dir, state)
        print(f"[r4] fresh state, run_dir={run_dir}", flush=True)
    else:
        print(f"[r4] resumed state: pass={state['pass_num']} param_idx={state['param_idx']} "
              f"accepted={len(state['accepted'])} evals_so_far={state['eval_counter']}", flush=True)

    start_time = state.get("start_time") or time.time()
    budget_s = budget_hours * 3600.0

    def elapsed():
        return time.time() - start_time

    def time_left():
        return budget_s - elapsed()

    seed_base = 20260809

    while state["pass_num"] <= max_passes:
        if time_left() <= 0:
            print(f"[r4] budget exhausted at pass={state['pass_num']} param_idx={state['param_idx']}",
                  flush=True)
            break
        pass_num = state["pass_num"]
        for idx in range(state["param_idx"], len(active_param_order)):
            if time_left() <= 0:
                state["param_idx"] = idx
                save_state(run_dir, state)
                print("[r4] budget exhausted mid-pass, saved state", flush=True)
                break
            pname = active_param_order[idx]
            current_genome = state["genome"]
            accepted_this_param = False
            for direction in (1, -1):
                trial = dict(current_genome)
                trial[pname] = mutate_value(pname, current_genome[pname], direction)
                if trial[pname] == current_genome[pname]:
                    continue  # упёрлись в границу, шаг в эту сторону не имеет смысла
                state["eval_counter"] += 1
                tag = f"p{pass_num}_i{idx:02d}_{pname}_{'p' if direction > 0 else 'm'}"
                t0_rec = eval_tier0(trial, tag, work_root, workers, seed_base + state["eval_counter"])
                t0_rec["param"] = pname
                t0_rec["direction"] = direction
                log_jsonl(log_path, t0_rec)
                print(f"[r4] T0 {tag}: wr_band={t0_rec['wr_band']*100:.1f}% "
                      f"ref={REFERENCE_WR_BAND*100:.1f}% delta={t0_rec['delta']*100:+.1f}pp "
                      f"se={t0_rec['se']*100:.1f}pp passed={t0_rec['passed']} "
                      f"({t0_rec['elapsed_s']:.0f}s)", flush=True)
                if not t0_rec["passed"]:
                    continue
                t1_rec = eval_tier1(trial, tag, work_root, workers, seed_base + state["eval_counter"])
                t1_rec["param"] = pname
                t1_rec["direction"] = direction
                log_jsonl(log_path, t1_rec)
                print(f"[r4] T1 {tag}: delta_band={t1_rec['delta_band']*100:+.1f}pp "
                      f"z={t1_rec['z_band']:+.2f} p={t1_rec['p_value']:.4f} "
                      f"passed={t1_rec['passed']} ({t1_rec['elapsed_s']:.0f}s)", flush=True)
                if t1_rec["passed"]:
                    state["genome"] = trial
                    state["accepted"].append(dict(pass_num=pass_num, idx=idx, param=pname,
                                                   direction=direction, tag=tag,
                                                   delta_band=t1_rec["delta_band"],
                                                   z_band=t1_rec["z_band"]))
                    accepted_this_param = True
                    print(f"[r4] ACCEPTED {pname} dir={direction} -> genome updated "
                          f"(accepted so far: {len(state['accepted'])})", flush=True)
                    break
            state["param_idx"] = idx + 1
            save_state(run_dir, state)
            if not accepted_this_param:
                print(f"[r4] {pname}: no improvement either direction, keep current value", flush=True)
        else:
            # полный проход по всем параметрам завершён без прерывания по бюджету
            state["pass_num"] = pass_num + 1
            state["param_idx"] = 0
            save_state(run_dir, state)
            continue
        break  # вышли из внутреннего for по бюджету

    # --- финал: ярус 2 (обязателен всегда, даже при пустом genome) -----------
    if time_left() <= 0:
        print("[r4] WARNING: бюджет уже исчерпан, ярус 2 всё равно запускается "
              "(обязателен перед любым решением о промоушене, см. ТЗ budget_plan)", flush=True)
    print(f"[r4] === TIER 2 (final lock) === genome accepted steps = {len(state['accepted'])}",
          flush=True)
    t2 = eval_tier2(state["genome"], work_root, workers, seed_base + 900000)
    log_jsonl(log_path, dict(tier=2, **{k: v for k, v in t2.items()}))
    state["tier2"] = t2
    save_state(run_dir, state)
    print(f"[r4] TIER2 done: z_combined={t2['z_combined']:+.2f} p_combined={t2['p_combined']:.4f} "
          f"fitness_pool_pass={t2['fitness_pool_pass']} holdout_reject={t2['holdout_reject']} "
          f"PROMOTE={t2['promote']}", flush=True)
    print(f"[r4] FINISHED run_dir={run_dir}", flush=True)


# ---------------------------------------------------------------------------
# Smoke self-test
# ---------------------------------------------------------------------------

def run_smoke(run_dir: str, workers: int) -> None:
    os.makedirs(run_dir, exist_ok=True)
    work_root = os.path.join(run_dir, "work")
    os.makedirs(work_root, exist_ok=True)
    log_path = os.path.join(run_dir, "smoke_log.jsonl")

    print(f"[r4-smoke] REFERENCE_WR_BAND={REFERENCE_WR_BAND*100:.2f}% "
          f"(from {BASE_CORE_JSON})", flush=True)
    print(f"[r4-smoke] NORM_W={NORM_W}", flush=True)

    base = init_genome()
    applied = apply_genome(BASE_POLICY, base)
    assert applied["main_priority"] == BASE_POLICY["main_priority"], (
        applied["main_priority"], BASE_POLICY["main_priority"])
    assert applied["search"]["ucb_c"] == BASE_POLICY["search"]["ucb_c"]
    # побитовая идентичность card-таблиц (delta=0 всюду)
    for cid, cdata in BASE_POLICY["contexts"].items():
        assert applied["contexts"][cid] == cdata, f"ctx {cid} drifted under identity genome"
    print("[r4-smoke] identity-genome self-check OK: apply_genome(base, init_genome()) "
          "== BASE_POLICY побитово", flush=True)

    n_smoke = 8

    print(f"[r4-smoke] eval #1: identity genome (=baseline) vs baseline, tier0-style scan "
          f"n={n_smoke}/opp", flush=True)
    r_identity = eval_tier0(base, "smoke_identity", work_root, workers, seed=20260809001,
                             n=n_smoke)
    log_jsonl(log_path, r_identity)
    print(f"[r4-smoke]   wr_band={r_identity['wr_band']*100:.1f}% delta_vs_ref="
          f"{r_identity['delta']*100:+.1f}pp (ожидаем ~0, REFERENCE снят на N=200, тут N={n_smoke} "
          f"— шумно, это НЕ финальная проверка)", flush=True)
    for k, o in r_identity["per_opp"].items():
        print(f"[r4-smoke]     {k}: WR={o['win_rate']:.3f} n={o['n_games']} "
              f"avg_len={o['avg_game_length']:.1f} crashes={o['crashes']} "
              f"illegal={o['illegal_agent']}/{o['illegal_opp']}", flush=True)

    print(f"[r4-smoke] eval #2: identity genome vs baseline, tier1-style PAIRED n={n_smoke}/opp",
          flush=True)
    r_pair = eval_tier1(base, "smoke_identity_pair", work_root, workers, seed=20260809002,
                         n=n_smoke)
    log_jsonl(log_path, r_pair)
    print(f"[r4-smoke]   delta_band={r_pair['delta_band']*100:+.1f}pp z={r_pair['z_band']:+.2f} "
          f"(ожидаем z≈0 — идентичный геном против самого себя, это sanity, не сигнал)",
          flush=True)

    print(f"[r4-smoke] eval #3: ЗАВЕДОМО ИСПОРЧЕННЫЙ геном (card_delta наших базовых -0.35, "
          f"main_priority переставлен так что ATTACK/EVOLVE идут последними, ucb_c=0) "
          f"vs baseline, tier1-style PAIRED n={n_smoke}/opp", flush=True)
    broken = dict(base)
    for cid in (42, 89, 100, 343):  # BASIC_POKEMON_IDS, main.py:36
        broken[_cd_name(cid)] = -0.35
    # main_priority: пусть ATTACK(13)/EVOLVE(9) станут наименее приоритетными
    for t in TYPE_IDS:
        broken[_mp_name(t)] = 1.0
    broken[_mp_name(14)] = 20.0  # END — самый приоритетный тип, агент пассует
    broken["ucb_c"] = 0.0
    r_broken = eval_tier1(broken, "smoke_broken", work_root, workers, seed=20260809003,
                           n=n_smoke)
    log_jsonl(log_path, r_broken)
    print(f"[r4-smoke]   delta_band={r_broken['delta_band']*100:+.1f}pp z={r_broken['z_band']:+.2f} "
          f"p={r_broken['p_value']:.4f} (ОЖИДАЕМ большой ОТРИЦАТЕЛЬНЫЙ delta — испорченный геном "
          f"должен заметно проигрывать)", flush=True)
    for a in r_broken["per_archetype"]:
        print(f"[r4-smoke]     {a['key']}: cand_wr={a['cand_wr']:.3f} base_wr={a['base_wr']:.3f} "
              f"delta={a['delta']*100:+.1f}pp z={a['z']:+.2f}", flush=True)

    print("[r4-smoke] DONE. Проверить вручную: grep -ac DEGRADED и src=search в stderr этого "
          "процесса (нет отдельного файла — вывод идёт в тот терминал/лог, куда direct вызов был "
          "перенаправлен).", flush=True)

    if r_broken["delta_band"] >= -0.05:
        print("[r4-smoke] *** WARNING: испорченный геном НЕ проиграл заметно (delta > -5пп). "
              "Фитнес может не различать плохие особи — см. правило ТЗ: НЕ запускать долгий "
              "поиск, если это так. ***", flush=True)
        sys.exit(3)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="Имя run_dir под out/r4_weight_search/ "
                                                  "(по умолчанию r4_<timestamp>)")
    ap.add_argument("--resume", action="store_true", help="Продолжить существующий --run")
    ap.add_argument("--smoke", action="store_true", help="2-3 особи, sanity-check, без долгого поиска")
    ap.add_argument("--budget-hours", type=float, default=18.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-passes", type=int, default=2)
    ap.add_argument("--max-params", type=int, default=None,
                     help="Сузить PARAM_ORDER до первых N параметров (по умолчанию все 29). "
                          "PARAM_ORDER уже отсортирован по приоритету (ucb_c, затем 8 весов "
                          "main_priority первыми) -- см. диагностику скорости 2026-08-09: "
                          "--max-params 9 = только ucb_c + main_priority.")
    args = ap.parse_args()

    if args.workers > 6:
        print(f"[r4] FATAL: workers={args.workers} > 6 (жёсткий потолок по ТЗ)", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("[r4] FATAL: workers < 1", file=sys.stderr)
        return 2

    run_name = args.run or ("r4_" + time.strftime("%Y%m%d_%H%M%S"))
    run_dir = os.path.join(OUT_ROOT, run_name)

    if args.smoke:
        run_smoke(os.path.join(OUT_ROOT, "smoke_" + time.strftime("%Y%m%d_%H%M%S")), args.workers)
        return 0

    if not args.resume and os.path.exists(os.path.join(run_dir, "state.json")):
        print(f"[r4] FATAL: {run_dir}/state.json уже существует, но --resume не передан. "
              f"Передай --resume, либо другое --run имя.", file=sys.stderr)
        return 2

    param_order = PARAM_ORDER[:args.max_params] if args.max_params is not None else None
    if param_order is not None:
        print(f"[r4] сужение генома: {len(param_order)}/{len(PARAM_ORDER)} параметров -> "
              f"{param_order}", flush=True)
    run_search(run_dir, args.budget_hours, args.workers, args.max_passes, param_order=param_order)
    return 0


if __name__ == "__main__":
    sys.exit(main())
