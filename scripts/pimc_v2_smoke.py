"""pimc_v2_smoke.py — H3 SMOKE (не полный гейт!) для PIMCAgentV2.

PLAN.md §H3 задача: доказать РАБОТОСПОСОБНОСТЬ (0 exceptions, 0 illegal moves)
раздельных rollout-политик (H3-1, фикс root cause S4b) + редетерминизации на
каждый rollout (H3-2, уже структурно в pimc_guided._evaluate_candidates) +
первого хода (H3-3) и собрать тайминги/статистику решений. W/L на 4 играх —
НЕ вывод (недостаточно партий для значимости), только работоспособность.

Три матчапа, наша сторона всегда PIMCAgentV2(deck=Crustle, opponent_model="auto",
prior_agent=v009a IL, epsilon=0.3), по 4 партии каждый, чередование сторон:
  (a) mirror     vs v009a_crustle   (flat IL, наш же prior без поиска)
  (b) alakazam   vs v017_alakazam_heur
  (c) grimmsnarl vs v018_grimmsnarl

Собираем на партию: решений всего / searched / cheap, rollouts/decision
(медиана, приближение через world_total из pimc_guided._evaluate_candidates —
число попыток детерминизации в этой SEARCH-развилке, ~= число rollout'ов),
ms/decision (медиана, макс), bank_spent к концу партии, счётчики деградации
rollout-политик (PIMCAgentV2.rollout_degradation), доля search/cheap.

Результаты (по партии + агрегат) -> scratchpad/h3_smoke.json.

Запуск (нативный движок, macOS arm64, БЕЗ Docker):
    .venv/bin/python scripts/pimc_v2_smoke.py
    .venv/bin/python scripts/pimc_v2_smoke.py --games 4 --matchups mirror,alakazam,grimmsnarl
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
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
from cg_native.api import to_observation_class  # noqa: E402
import cg_native  # noqa: E402
import cg_native.api  # noqa: E402
import cg_native.game  # noqa: E402
import cg_native.sim  # noqa: E402
import cg_native.utils  # noqa: E402

from exit.pimc_agent import PIMCAgentV2, CRUSTLE_DECK_APPROX  # noqa: E402

# Безопасный алиас: если opponent main.py (или его зависимости) импортируют
# `from cg.api import ...` / `from cg.game import ...` на верхнем уровне,
# на macOS это упадёт (cg -> Linux-only libcg.so). Подменяем cg* на уже
# импортированные нативные cg_native-модули ДО exec opponent main.py.
# (паттерн из src/eval/anchored_pool.py; v009a/v017/v018 сами cg не импортируют,
# но алиас оставлен как safety net на будущее.)
sys.modules.setdefault("cg", cg_native)
sys.modules.setdefault("cg.api", cg_native.api)
sys.modules.setdefault("cg.game", cg_native.game)
sys.modules.setdefault("cg.sim", cg_native.sim)
sys.modules.setdefault("cg.utils", cg_native.utils)

CRUSTLE_DECK = list(CRUSTLE_DECK_APPROX)

MAX_MOVES_PER_GAME = 2000
RESULTS_PATH = os.path.join(ROOT, "scratchpad", "h3_smoke.json")

MATCHUPS: dict[str, str] = {
    "mirror": "v009a_crustle",
    "alakazam": "v017_alakazam_heur",
    "grimmsnarl": "v018_grimmsnarl",
}

# H3-2 bank params: bank_total_sec/safety выбраны так, чтобы budget_this
# реалистично попадал в диапазон 2-4s/решение (см. PLAN.md §H3-2), но общий
# расход на партию был ограничен (~54s max на нашу сторону) — это СМОК на
# 4 партии/матчап, не полный гейт с default 600s/0.5.
BANK_KW = dict(
    bank_total_sec=90.0,
    bank_safety=0.6,
    est_total_decisions=40,
    min_decision_budget=0.5,
    max_decision_budget=3.0,
)
EPSILON = 0.3


# ---------------------------------------------------------------------------
# Fresh-module loader (H3-1 stateful-audit: v018_grimmsnarl's _adrena_state
# is real sequencing state -> fresh module per GAME avoids cross-GAME leakage;
# v017/v009a only have diagnostic counters, harmless, but reloaded anyway for
# uniformity -- same pattern as scripts/pimc_field_gauntlet.py).
# ---------------------------------------------------------------------------

_LOAD_COUNTER = 0


def _load_submission_module(sub_dir_name: str):
    global _LOAD_COUNTER
    _LOAD_COUNTER += 1
    sub_dir = os.path.join(ROOT, "submissions", sub_dir_name)
    main_path = os.path.join(sub_dir, "main.py")
    assert os.path.exists(main_path), f"не найден {main_path}"
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    mod_name = f"_h3_smoke_{sub_dir_name}_{_LOAD_COUNTER}"
    spec = importlib.util.spec_from_file_location(mod_name, main_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = main_path
    spec.loader.exec_module(mod)
    return mod


def _is_legal(choice, sel: dict | None) -> bool:
    if sel is None:
        return isinstance(choice, list) and len(choice) == 60 and all(isinstance(c, int) for c in choice)
    opts = sel.get("option") or []
    n_opts = len(opts)
    min_c = int(sel.get("minCount") or 1)
    max_c = int(sel.get("maxCount") or 1)
    return (
        isinstance(choice, list)
        and all(isinstance(i, int) and 0 <= i < n_opts for i in choice)
        and min_c <= len(choice) <= max_c
        and len(set(choice)) == len(choice)
    )


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------

def run_one_game(matchup: str, opp_sub_dir: str, our_first: bool, seed: int) -> dict:
    # Свежий v009a-инстанс КАЖДУЮ игру -- prior_agent для НАШЕЙ стороны
    # (Crustle-IL, используется и как decision-time cheap-path, и как
    # rollout-политика нашей стороны внутри PIMCAgentV2).
    v009a_prior_mod = _load_submission_module("v009a_crustle")

    pimc = PIMCAgentV2(
        deck=CRUSTLE_DECK,
        prior_agent=v009a_prior_mod.agent,
        opponent_model="auto",
        epsilon=EPSILON,
        rng_seed=seed,
        **BANK_KW,
    )
    # Форсируем _lazy_init ДО начала партии, чтобы pimc._pimc_mod был готов
    # и можно было снять baseline-длину worlds_per_decision (module-global
    # список, накапливается по ВСЕМ играм в этом процессе -- нужен срез
    # [before:after] на КОНКРЕТНУЮ партию).
    pimc._lazy_init()
    worlds_before = len(pimc._pimc_mod._stats.get("worlds_per_decision", []))

    opp_mod = _load_submission_module(opp_sub_dir)
    opp_agent_fn = opp_mod.agent
    opp_deck = list(opp_mod.my_deck)

    if our_first:
        agent0, agent1 = pimc.agent, opp_agent_fn
        deck0, deck1 = list(CRUSTLE_DECK), list(opp_deck)
    else:
        agent0, agent1 = opp_agent_fn, pimc.agent
        deck0, deck1 = list(opp_deck), list(CRUSTLE_DECK)

    obs_dict, start = battle_start(deck0, deck1)
    if obs_dict is None:
        raise RuntimeError(f"battle_start failed: errorType={getattr(start, 'errorType', None)}")

    move_count = 0
    our_decision_times: list[float] = []  # ms
    n_illegal = 0
    n_exceptions = 0
    winner = None
    exc_side = None

    try:
        while move_count < MAX_MOVES_PER_GAME:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                winner = obs.current.result
                break

            your_index = obs.current.yourIndex if obs.current else 0
            is_our_turn = (your_index == 0) == our_first
            agent_fn = agent0 if your_index == 0 else agent1

            sel = obs_dict.get("select")
            t0 = time.perf_counter()
            try:
                choice = agent_fn(obs_dict)
            except Exception as e:
                n_exceptions += 1
                exc_side = "our" if is_our_turn else "opp"
                print(f"    !!! EXCEPTION (side={your_index}, our_turn={is_our_turn}): {e!r}", flush=True)
                break
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if is_our_turn:
                our_decision_times.append(dt_ms)

            if not _is_legal(choice, sel):
                n_illegal += 1
                print(f"    !!! ILLEGAL (side={your_index}, our_turn={is_our_turn}): "
                      f"choice={choice} sel={sel}", flush=True)
                break

            obs_dict = battle_select(choice)
            move_count += 1
    finally:
        battle_finish()

    worlds_after = len(pimc._pimc_mod._stats.get("worlds_per_decision", []))
    worlds_this_game = pimc._pimc_mod._stats["worlds_per_decision"][worlds_before:worlds_after]

    our_side = 0 if our_first else 1
    if n_exceptions > 0:
        outcome = "exception"
    elif n_illegal > 0:
        outcome = "illegal"
    elif winner is None:
        outcome = "aborted"
    elif winner == our_side:
        outcome = "our_win"
    elif winner == 2:
        outcome = "draw"
    else:
        outcome = "opp_win"

    bank = pimc.bank_stats
    decisions_total = bank["decisions_made"]
    n_search = bank["n_search_decisions"]
    n_cheap = bank["n_cheap_decisions"]

    return dict(
        matchup=matchup,
        outcome=outcome,
        exc_side=exc_side,
        winner=winner,
        moves=move_count,
        our_first=our_first,
        seed=seed,
        n_illegal=n_illegal,
        n_exceptions=n_exceptions,
        decisions_total=decisions_total,
        n_search_decisions=n_search,
        n_cheap_decisions=n_cheap,
        search_share=(n_search / decisions_total) if decisions_total else None,
        rollouts_per_search_decision_median=_median([float(w) for w in worlds_this_game]),
        rollouts_per_search_decision_max=(max(worlds_this_game) if worlds_this_game else 0),
        n_search_decisions_with_worlds=len(worlds_this_game),
        ms_per_decision_median=_median(our_decision_times),
        ms_per_decision_max=(max(our_decision_times) if our_decision_times else 0.0),
        bank_spent_sec=bank["spent_sec"],
        wall_clock_sec=bank["wall_clock_sec"],
        rollout_degradation=dict(pimc.rollout_degradation),
        opp_policy_cache_keys=list(pimc._opp_policy_cache.keys()),
        locked_archetype=pimc._locked_archetype,
    )


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4, help="партий на матчап")
    ap.add_argument("--matchups", type=str, default="mirror,alakazam,grimmsnarl")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    matchups = [m.strip() for m in args.matchups.split(",") if m.strip()]
    for m in matchups:
        assert m in MATCHUPS, f"unknown matchup {m!r}, choose from {list(MATCHUPS)}"

    state: dict = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            state = json.load(f)

    total_illegal = 0
    total_exceptions = 0

    for matchup in matchups:
        opp_sub_dir = MATCHUPS[matchup]
        games = state.setdefault(matchup, [])
        print(f"\n=== {matchup} vs {opp_sub_dir} ===", flush=True)
        n_have = len(games)
        for i in range(n_have, args.games):
            our_first = (i % 2 == 0)
            seed = args.seed + hash((matchup, i)) % 1_000_000
            t0 = time.time()
            try:
                res = run_one_game(matchup, opp_sub_dir, our_first, seed)
            except Exception as e:
                res = dict(matchup=matchup, outcome="crash", error=repr(e),
                           n_illegal=0, n_exceptions=1, our_first=our_first, seed=seed)
                print(f"  !!! game crashed: {e!r}", flush=True)
            res["wall_time_game_sec"] = time.time() - t0
            games.append(res)
            total_illegal += res.get("n_illegal", 0)
            total_exceptions += res.get("n_exceptions", 0)

            with open(RESULTS_PATH, "w") as f:
                json.dump(state, f, indent=2)

            side = "OUR=P0" if our_first else "OUR=P1"
            print(f"  game {i+1}/{args.games}  {side}  outcome={res.get('outcome')}  "
                  f"moves={res.get('moves')}  decisions={res.get('decisions_total')}  "
                  f"search={res.get('n_search_decisions')}  "
                  f"ms/dec(med/max)={res.get('ms_per_decision_median', 0):.0f}/"
                  f"{res.get('ms_per_decision_max', 0):.0f}  "
                  f"bank_spent={res.get('bank_spent_sec', 0):.1f}s  "
                  f"degrad={res.get('rollout_degradation')}  "
                  f"wall={res['wall_time_game_sec']:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("H3 SMOKE SUMMARY (работоспособность, НЕ W/L-гейт)")
    print("=" * 78)
    summary: dict = {}
    for matchup in matchups:
        games = state.get(matchup, [])
        ok_games = [g for g in games if g.get("outcome") not in ("crash",)]
        n_illegal = sum(g.get("n_illegal", 0) for g in games)
        n_exc = sum(g.get("n_exceptions", 0) for g in games)
        wins = sum(1 for g in games if g.get("outcome") == "our_win")
        losses = sum(1 for g in games if g.get("outcome") == "opp_win")
        draws = sum(1 for g in games if g.get("outcome") == "draw")
        ms_meds = [g["ms_per_decision_median"] for g in ok_games if g.get("ms_per_decision_median")]
        ms_maxs = [g["ms_per_decision_max"] for g in ok_games if g.get("ms_per_decision_max")]
        rollouts_meds = [g["rollouts_per_search_decision_median"] for g in ok_games
                          if g.get("rollouts_per_search_decision_median")]
        bank_spents = [g["bank_spent_sec"] for g in ok_games if "bank_spent_sec" in g]
        all_degrad: dict = {}
        for g in ok_games:
            for k, v in (g.get("rollout_degradation") or {}).items():
                all_degrad[k] = all_degrad.get(k, 0) + v
        decisive = wins + losses
        wr = wins / decisive if decisive else float("nan")
        lo, hi = wilson_ci(wins, decisive) if decisive else (float("nan"), float("nan"))

        summ = dict(
            n_games=len(games),
            wins=wins, losses=losses, draws=draws,
            winrate=wr, wilson_lo=lo, wilson_hi=hi,
            n_illegal_total=n_illegal,
            n_exceptions_total=n_exc,
            ms_per_decision_median_of_medians=_median(ms_meds),
            ms_per_decision_max_overall=(max(ms_maxs) if ms_maxs else 0.0),
            rollouts_per_decision_median_of_medians=_median(rollouts_meds),
            bank_spent_median_sec=_median(bank_spents),
            bank_spent_max_sec=(max(bank_spents) if bank_spents else 0.0),
            rollout_degradation_total=all_degrad,
        )
        summary[matchup] = summ
        print(f"\n--- {matchup} ---")
        print(f"  games={summ['n_games']}  {wins}-{losses} (draws={draws})  "
              f"winrate={wr:.3f} [informational only, n={decisive}]")
        print(f"  illegal={n_illegal}  exceptions={n_exc}")
        print(f"  ms/decision: median-of-medians={summ['ms_per_decision_median_of_medians']:.0f}  "
              f"max-overall={summ['ms_per_decision_max_overall']:.0f}")
        print(f"  rollouts/decision (median-of-medians): {summ['rollouts_per_decision_median_of_medians']:.1f}")
        print(f"  bank_spent: median={summ['bank_spent_median_sec']:.1f}s  max={summ['bank_spent_max_sec']:.1f}s")
        print(f"  rollout_degradation (total across games): {all_degrad}")

    state["_summary"] = summary
    with open(RESULTS_PATH, "w") as f:
        json.dump(state, f, indent=2)

    print(f"\nTOTAL illegal={total_illegal}  TOTAL exceptions={total_exceptions}")
    print(f"JSON: {RESULTS_PATH}")

    ok = (total_illegal == 0 and total_exceptions == 0)
    print(f"\n{'SMOKE PASSED (0 illegal, 0 exceptions)' if ok else 'SMOKE FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
