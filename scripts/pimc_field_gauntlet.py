"""pimc_field_gauntlet.py — S4b: PIMC(Crustle) vs field, decisive search-vs-field test.

S4a (mirror) showed PIMC(Crustle IL prior) ~= its own flat IL prior in the MIRROR
(worst case for PIMC: no opponent-archetype uncertainty to exploit). This script
tests the actual thesis: does PIMC's per-opponent adaptation (opponent_model="auto"
-> meta-mix determinization + archetype lock-on from revealed cards) help against
REAL FIELD AGENTS piloting different meta decks, where there IS exploitable
asymmetric information?

Two opponents (real pilots, not raw decklists):
  - v017_alakazam_heur (submissions/v017_alakazam_heur/main.py) — Alakazam combo.
  - v018_grimmsnarl     (submissions/v018_grimmsnarl/main.py)     — Grimmsnarl toolbox.

Two arms per opponent, our side always Crustle:
  - A (search):  PIMCAgent(deck=Crustle, prior_agent=v009a IL, opponent_model="auto")
  - B (control): flat v009a IL agent, no search.

Metric: winrate of OUR side in each arm + Wilson 95% CI; delta = wr(A) - wr(B).

Resumable: persists every game to scratchpad/s4b_field.json immediately after it
finishes. Re-running this script re-reads that file and continues filling any
(opponent, arm) cell that hasn't reached --games yet. Safe to kill/timeout anytime.

Run (foreground only, chunks of a few games at a time recommended):
    .venv/bin/python scripts/pimc_field_gauntlet.py --games 20 --time-budget 420

Each invocation stops early (saving state) once --time-budget seconds have
elapsed, so repeated invocations converge on the full N games per cell.
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

from exit.pimc_agent import PIMCAgent, CRUSTLE_DECK_APPROX  # noqa: E402

CRUSTLE_DECK = list(CRUSTLE_DECK_APPROX)

MAX_MOVES_PER_GAME = 2000

RESULTS_PATH = os.path.join(ROOT, "scratchpad", "s4b_field.json")

OPPONENTS: dict[str, str] = {
    "alakazam": "v017_alakazam_heur",
    "grimmsnarl": "v018_grimmsnarl",
}
ARMS = ("pimc", "il")


# ---------------------------------------------------------------------------
# Wilson 95% CI (same formula as scripts/pimc_vs_il_h2h.py)
# ---------------------------------------------------------------------------

def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Dynamic loading of submission main.py modules (same importlib pattern as
# scripts/pimc_vs_il_h2h.py's _load_v009a_agent — reused for the field
# opponents too). Each call produces a FRESH module instance (unique spec
# name with a monotonic counter) so per-game module-level state (e.g.
# v018_grimmsnarl's `_adrena_state` sequence flag, which is never reset on
# deck-select) can never leak between games.
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
    mod_name = f"_field_gauntlet_{sub_dir_name}_{_LOAD_COUNTER}"
    spec = importlib.util.spec_from_file_location(mod_name, main_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = main_path  # guarded _DIR resolution in their main.py relies on this
    spec.loader.exec_module(mod)
    return mod


def _load_v009a_agent():
    """v009a IL — reused 1:1 pattern from pimc_vs_il_h2h.py (fresh each call, no
    known cross-game mutable state in v009a's main.py, but cheap & safe anyway)."""
    return _load_submission_module("v009a_crustle")


# ---------------------------------------------------------------------------
# Legality check (same contract as pimc_vs_il_h2h.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# One game. our_first=True -> our side is player0.
# our_agent_fn: callable(obs_dict) -> list[int]  (either PIMC.agent or flat IL agent)
# opp_agent_fn: callable(obs_dict) -> list[int]  (v017/v018 agent())
# opp_deck: their 60-card deck (from their module's my_deck)
# pimc_obj: the PIMCAgent instance if arm=="pimc" (for bank_stats), else None.
# ---------------------------------------------------------------------------

def run_one_game(our_first: bool, our_agent_fn, opp_agent_fn, opp_deck: list[int],
                  pimc_obj: "PIMCAgent | None") -> dict:
    if our_first:
        agent0, agent1 = our_agent_fn, opp_agent_fn
        deck0, deck1 = list(CRUSTLE_DECK), list(opp_deck)
    else:
        agent0, agent1 = opp_agent_fn, our_agent_fn
        deck0, deck1 = list(opp_deck), list(CRUSTLE_DECK)

    obs_dict, start = battle_start(deck0, deck1)
    if obs_dict is None:
        raise RuntimeError(f"battle_start failed: errorType={start.errorType}")

    move_count = 0
    our_decision_times: list[float] = []
    n_illegal = 0
    n_exceptions = 0
    winner = None

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
                print(f"    !!! ИСКЛЮЧЕНИЕ (side={your_index}, our_turn={is_our_turn}): {e!r}", flush=True)
                raise
            dt = time.perf_counter() - t0
            if is_our_turn:
                our_decision_times.append(dt)

            if not _is_legal(choice, sel):
                n_illegal += 1
                print(f"    !!! НЕЛЕГАЛЬНЫЙ ВЫБОР (side={your_index}, our_turn={is_our_turn}): "
                      f"choice={choice} sel={sel}", flush=True)
                break

            obs_dict = battle_select(choice)
            move_count += 1
    finally:
        battle_finish()

    our_side = 0 if our_first else 1
    if winner is None:
        outcome = "aborted"
    elif winner == our_side:
        outcome = "our_win"
    elif winner == 2:
        outcome = "draw"
    else:
        outcome = "opp_win"

    result = dict(
        outcome=outcome,
        winner=winner,
        moves=move_count,
        n_illegal=n_illegal,
        n_exceptions=n_exceptions,
        our_first=our_first,
        max_decision_time=max(our_decision_times) if our_decision_times else 0.0,
        median_decision_time=(sorted(our_decision_times)[len(our_decision_times) // 2]
                               if our_decision_times else 0.0),
    )
    if pimc_obj is not None:
        bank = pimc_obj.bank_stats
        result.update(
            bank_spent=bank["spent_sec"],
            wall_clock=bank["wall_clock_sec"],
            decisions_made=bank["decisions_made"],
            n_search=bank["n_search_decisions"],
            n_cheap=bank["n_cheap_decisions"],
        )
    return result


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _empty_state() -> dict:
    return {opp: {arm: {"games": []} for arm in ARMS} for opp in OPPONENTS}


def load_state() -> dict:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            state = json.load(f)
        # backfill any missing keys (new opponents/arms added later)
        for opp in OPPONENTS:
            state.setdefault(opp, {})
            for arm in ARMS:
                state[opp].setdefault(arm, {"games": []})
        return state
    return _empty_state()


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, RESULTS_PATH)


def cell_summary(games: list[dict]) -> dict:
    wins = sum(1 for g in games if g["outcome"] == "our_win")
    losses = sum(1 for g in games if g["outcome"] == "opp_win")
    draws = sum(1 for g in games if g["outcome"] == "draw")
    aborted = sum(1 for g in games if g["outcome"] == "aborted")
    decisive = wins + losses
    wr = wins / decisive if decisive else float("nan")
    lo, hi = wilson_ci(wins, decisive) if decisive else (float("nan"), float("nan"))
    return dict(n=len(games), wins=wins, losses=losses, draws=draws, aborted=aborted,
                decisive=decisive, winrate=wr, wilson_lo=lo, wilson_hi=hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20, help="games target PER (opponent, arm) cell")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--time-budget", type=float, default=420.0,
                     help="stop (saving state) after this many wall-clock seconds")
    ap.add_argument("--opponents", type=str, default="alakazam,grimmsnarl")
    ap.add_argument("--arms", type=str, default="pimc,il")
    args = ap.parse_args()

    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    n_target = args.games
    t_deadline = time.time() + args.time_budget

    state = load_state()

    print(f"[boot] загружаю v009a IL (общий prior для PIMC И flat-контроль)...", flush=True)
    v009a_mod = _load_v009a_agent()
    print(f"       OK, my_deck len={len(v009a_mod.my_deck)}", flush=True)

    total_games_played_this_run = 0

    for opp_name in opponents:
        if time.time() >= t_deadline:
            break
        sub_dir_name = OPPONENTS[opp_name]

        for arm in arms:
            if time.time() >= t_deadline:
                break
            cell = state[opp_name][arm]["games"]
            n_have = len(cell)
            if n_have >= n_target:
                print(f"[skip] {opp_name}/{arm}: уже {n_have}/{n_target} игр", flush=True)
                continue

            print(f"\n=== {opp_name} / arm={arm}: {n_have}/{n_target} игр сделано, продолжаю ===", flush=True)

            while n_have < n_target and time.time() < t_deadline:
                game_idx = n_have  # 0-based index across the whole cell history
                our_first = (game_idx % 2 == 0)
                seed = args.seed + hash((opp_name, arm, game_idx)) % 1_000_000

                # Fresh module instances every game -> no cross-game state leakage
                # (v018_grimmsnarl's _adrena_state is a module global never reset
                # on deck-select; v009a/v017 are almost certainly stateless but we
                # reload them too for uniformity/safety).
                opp_mod = _load_submission_module(sub_dir_name)
                opp_agent_fn = opp_mod.agent
                opp_deck = list(opp_mod.my_deck)

                pimc_obj = None
                if arm == "pimc":
                    il_mod = _load_v009a_agent()
                    pimc_obj = PIMCAgent(
                        deck=CRUSTLE_DECK,
                        prior_agent=il_mod.agent,
                        opponent_model="auto",  # S4b: THE point -- per-opponent adaptation vs field
                    )
                    our_agent_fn = pimc_obj.agent
                else:
                    il_mod = _load_v009a_agent()
                    our_agent_fn = il_mod.agent

                t0 = time.time()
                try:
                    res = run_one_game(our_first, our_agent_fn, opp_agent_fn, opp_deck, pimc_obj)
                except Exception as e:
                    res = dict(outcome="aborted", winner=None, moves=0, n_illegal=0,
                               n_exceptions=1, our_first=our_first, error=repr(e))
                    print(f"    !!! game crashed: {e!r}", flush=True)
                dt_game = time.time() - t0
                res["wall_time_game"] = dt_game
                res["seed"] = seed

                cell.append(res)
                n_have = len(cell)
                total_games_played_this_run += 1
                save_state(state)

                side = "OUR=P0" if our_first else "OUR=P1"
                bank_str = ""
                if arm == "pimc":
                    bank_str = (f"  bank={res.get('bank_spent', 0):.1f}s "
                                f"search={res.get('n_search', 0)} cheap={res.get('n_cheap', 0)}")
                print(f"  [{opp_name}/{arm}] игра {n_have:3d}/{n_target}  {side}  "
                      f"outcome={res['outcome']:9s}  moves={res['moves']:4d}  "
                      f"wall={dt_game:6.1f}s  illegal={res['n_illegal']}  exc={res['n_exceptions']}"
                      f"{bank_str}", flush=True)

                summ = cell_summary(cell)
                if summ["decisive"] > 0:
                    print(f"    --- промежуточно {opp_name}/{arm}: {summ['wins']}-{summ['losses']} "
                          f"(draws={summ['draws']}) winrate={summ['winrate']:.3f} "
                          f"Wilson95%=[{summ['wilson_lo']:.3f},{summ['wilson_hi']:.3f}] ---", flush=True)

                if time.time() >= t_deadline:
                    print(f"\n[time-budget] {args.time_budget}s исчерпан, останавливаюсь "
                          f"(состояние сохранено, перезапусти скрипт чтобы продолжить)", flush=True)
                    break

    # -----------------------------------------------------------------
    # Final report (partial or full depending on where we stopped)
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ИТОГ S4b: PIMC(Crustle, auto) vs field  |  flat v009a IL vs field")
    print("=" * 78)
    print(f"игр сыграно в ЭТОМ запуске: {total_games_played_this_run}\n")

    deltas = {}
    for opp_name in OPPONENTS:
        print(f"--- {opp_name} ---")
        summaries = {}
        for arm in ARMS:
            cell = state[opp_name][arm]["games"]
            summ = cell_summary(cell)
            summaries[arm] = summ
            if summ["decisive"]:
                print(f"  arm={arm:5s}  n={summ['n']:3d}  {summ['wins']}-{summ['losses']} "
                      f"(draws={summ['draws']}, aborted={summ['aborted']})  "
                      f"winrate={summ['winrate']:.3f}  Wilson95%=[{summ['wilson_lo']:.3f},{summ['wilson_hi']:.3f}]")
            else:
                print(f"  arm={arm:5s}  n={summ['n']:3d}  (0 decisive games yet)")
        if summaries["pimc"]["decisive"] and summaries["il"]["decisive"]:
            delta = summaries["pimc"]["winrate"] - summaries["il"]["winrate"]
            deltas[opp_name] = delta
            print(f"  delta (PIMC - IL) = {delta:+.3f}")
        print()

    if deltas:
        print("--- Вердикт по гипотезе (PLAN.md §S4b) ---")
        vals = list(deltas.values())
        if all(d >= 0.08 for d in vals):
            print(f"  ВСЕ delta >= +0.08 ({deltas}) -> ГИПОТЕЗА ЖИВА: поиск бьёт поле выше плоского IL "
                  f"-> зелёный на полный S4-гейт + live-probe.")
        elif all(-0.05 <= d <= 0.05 for d in vals):
            print(f"  ВСЕ delta в пределах +-0.05 ({deltas}) -> поиск НЕ добавляет и против поля -> "
                  f"ветка поиска МЕРТВА окончательно -> разворот на Crustle-ретюн.")
        elif any(d < 0 for d in vals):
            print(f"  Есть отрицательная delta ({deltas}) -> поиск ВРЕДИТ хотя бы против одного оппонента -> "
                  f"МЕРТВА.")
        else:
            print(f"  Смешанный/неоднозначный результат: {deltas} -> см. таблицу выше, решение вручную.")

        # ceiling effect check
        for opp_name in OPPONENTS:
            summ_pimc = cell_summary(state[opp_name]["pimc"]["games"])
            summ_il = cell_summary(state[opp_name]["il"]["games"])
            if summ_pimc["decisive"] and summ_il["decisive"]:
                if summ_pimc["winrate"] > 0.85 and summ_il["winrate"] > 0.85:
                    print(f"  [ceiling] {opp_name}: ОБА плеча >85% winrate -> оппонент слишком слаб, "
                          f"сигнал по delta слабый (добавить более сильного оппонента, напр. v002_lucario).")
    else:
        print("  Недостаточно решительных партий ни в одной ячейке для вердикта -- продолжай запускать скрипт.")

    print(f"\nJSON состояние: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
