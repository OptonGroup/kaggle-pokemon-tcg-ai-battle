"""pimc_bug_audit.py — следствие PIMC v2 bug-audit (PLAN.md §H, аномалия H3-4).

Инструментирует КАЖДУЮ search-развилку PIMCAgentV2 vs v001_random:
  - полный obs.select.option (легальные ходы)
  - кандидаты PIMC (_smart_candidates)
  - prior_choice (v009a.agent отдельно)
  - PIMC chosen
  - per-candidate rollout score (mean), N rollouts, variance, terminal rate, avg length
  - cheap vs search path

Цель: найти ROOT CAUSE аномалии WR=0.727 vs random (prior бьёт random 95%+).

Запуск (нативный движок, macOS arm64, один процесс):
    .venv/bin/python scripts/pimc_bug_audit.py --games 20
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

sys.modules.setdefault("cg", cg_native)
sys.modules.setdefault("cg.api", cg_native.api)
sys.modules.setdefault("cg.game", cg_native.game)
sys.modules.setdefault("cg.sim", cg_native.sim)
sys.modules.setdefault("cg.utils", cg_native.utils)

CRUSTLE_DECK = list(CRUSTLE_DECK_APPROX)
MAX_MOVES_PER_GAME = 2000
LOG_PATH = os.path.join(ROOT, "scratchpad", "pimc_audit_log.json")

# Bank как в smoke (быстрее игры), но достаточно для реального поиска
BANK_KW = dict(
    bank_total_sec=90.0,
    bank_safety=0.6,
    est_total_decisions=40,
    min_decision_budget=0.5,
    max_decision_budget=3.0,
)
EPSILON = 0.3

_LOAD_COUNTER = 0


def _load_submission_module(sub_dir_name: str):
    global _LOAD_COUNTER
    _LOAD_COUNTER += 1
    sub_dir = os.path.join(ROOT, "submissions", sub_dir_name)
    main_path = os.path.join(sub_dir, "main.py")
    assert os.path.exists(main_path), f"не найден {main_path}"
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    mod_name = f"_audit_{sub_dir_name}_{_LOAD_COUNTER}"
    spec = importlib.util.spec_from_file_location(mod_name, main_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = main_path
    spec.loader.exec_module(mod)
    return mod


# OptionType int-константы (для человекочитаемого лога)
_OT_NAMES = {
    1: "YES", 2: "NO", 3: "CARD", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE",
    10: "ABILITY", 12: "RETREAT", 13: "ATTACK", 14: "END",
}
_SC_NAMES = {0: "MAIN", 41: "IS_FIRST"}
_ST_NAMES = {8: "COUNT", 9: "YES_NO"}


def _opt_summary(o):
    """Человекочитаемое описание опции."""
    if isinstance(o, dict):
        t = o.get("type")
        aid = o.get("attackId")
        return f"type={_OT_NAMES.get(t, t)} attackId={aid}"
    t = getattr(o, "type", None)
    aid = getattr(o, "attackId", None)
    return f"type={_OT_NAMES.get(int(t), int(t) if t is not None else None)} attackId={aid}"


def _is_legal(choice, sel):
    if sel is None:
        return isinstance(choice, list) and len(choice) == 60
    opts = sel.get("option") or []
    n = len(opts)
    min_c = int(sel.get("minCount") or 1)
    max_c = int(sel.get("maxCount") or 1)
    return (isinstance(choice, list)
            and all(isinstance(i, int) and 0 <= i < n for i in choice)
            and min_c <= len(choice) <= max_c
            and len(set(choice)) == len(choice))


# ---------------------------------------------------------------------------
# Инструментированный _evaluate_candidates — копия pimc_guided._evaluate_candidates
# с полным логированием per-candidate scores, rollout lengths, terminal rate.
# ---------------------------------------------------------------------------

def _make_instrumented_evaluate(pimc_mod, audit_log, prior_agent_fn):
    """Создаёт инструментированную версию _evaluate_candidates.

    Копирует логику pimc_guided._evaluate_candidates, но логирует:
      - candidates, obs.select.option
      - per-candidate: scores, counts, rollout_results (list of {score, steps, terminal})
      - chosen idx, prior_choice
    """
    # Используем уже-патченные функции из pimc_mod (включая V2 _do_guided_rollout)
    _build_search_inputs = pimc_mod._build_search_inputs
    _do_guided_rollout = pimc_mod._do_guided_rollout
    _opp_active_ids = pimc_mod._opp_active_ids

    try:
        from cg_native.api import search_begin, search_step, search_end, search_release
    except ImportError:
        from cg.api import search_begin, search_step, search_end, search_release

    N_WORLDS_MIN = lambda: pimc_mod.N_WORLDS_MIN

    def _instrumented_evaluate(obs, candidates, time_budget):
        your_index = obs.current.yourIndex
        opp_idx = 1 - your_index
        n_cands = len(candidates)

        if n_cands == 1:
            return 0, 0

        scores = [0.0] * n_cands
        counts = [0] * n_cands
        # per-candidate: list of (score, steps, terminal_reached)
        rollout_details = [[] for _ in range(n_cands)]
        t_start = time.time()
        world_total = 0

        while time.time() - t_start < time_budget:
            try:
                your_deck, your_prize, opp_deck, opp_prize, opp_hand = \
                    _build_search_inputs(obs, your_index)
                opp_act = _opp_active_ids(obs, opp_idx)
            except Exception:
                world_total += 1
                continue

            cand_idx = world_total % n_cands
            action = candidates[cand_idx]

            try:
                root = search_begin(
                    obs,
                    your_deck=your_deck,
                    your_prize=your_prize,
                    opponent_deck=opp_deck,
                    opponent_prize=opp_prize,
                    opponent_hand=opp_hand,
                    opponent_active=opp_act,
                    manual_coin=False,
                )
            except Exception:
                world_total += 1
                continue

            try:
                try:
                    after = search_step(root.searchId, action)
                except Exception:
                    search_release(root.searchId)
                    world_total += 1
                    continue

                # Инструментируем rollout: оборачиваем _do_guided_rollout чтобы считать шаги
                steps_taken = [0]
                terminal = [False]

                # Обёртка над search_step для подсчёта шагов
                _orig_search_step = search_step
                rollout_score = [None]

                def _counting_rollout(sid, obs_after, yi):
                    # Копия _do_guided_rollout логики с подсчётом шагов
                    cur_obs = obs_after
                    cur_sid = sid
                    max_depth = pimc_mod.MAX_ROLLOUT_DEPTH
                    for step_i in range(max_depth):
                        steps_taken[0] = step_i + 1
                        if cur_obs.current and cur_obs.current.result != -1:
                            terminal[0] = True
                            r = cur_obs.current.result
                            if r == yi:
                                return 1.0
                            elif r == 2:
                                return 0.0
                            else:
                                return -1.0
                        # Используем патченный _do_guided_rollout для ОДНОГО шага?
                        # НЕТ — _do_guided_rollout делает весь rollout. Вызовем его,
                        # но он не возвращает steps. Поэтому вызовем оригинал и
                        # отдельно посчитаем steps через monkey-patch search_step.
                        break
                    # Если не terminal за 1 шаг — вызовем оригинальный rollout
                    return _do_guided_rollout(sid, obs_after, yi)

                # Простой подход: вызываем оригинальный _do_guided_rollout, но
                # подсчёт шагов делаем через обёрнутый search_step.
                # Однако search_step — глобальная функция, не передаётся в rollout.
                # _do_guided_rollout вызывает pimc_mod.search_step (импорт).
                # Патчим pimc_mod.search_step временно.
                orig_ss = pimc_mod.search_step
                step_counter = [0]
                term_flag = [False]

                def _counting_ss(sid, choice):
                    nxt = orig_ss(sid, choice)
                    step_counter[0] += 1
                    if nxt.observation.current and nxt.observation.current.result != -1:
                        term_flag[0] = True
                    return nxt

                pimc_mod.search_step = _counting_ss
                try:
                    result = _do_guided_rollout(after.searchId, after.observation, your_index)
                finally:
                    pimc_mod.search_step = orig_ss

                scores[cand_idx] += result
                counts[cand_idx] += 1
                rollout_details[cand_idx].append({
                    "score": result,
                    "steps": step_counter[0],
                    "terminal": term_flag[0],
                })
                pimc_mod._stats["rollouts_total"] += 1

            finally:
                try:
                    search_end()
                except Exception:
                    pass

            world_total += 1

            if min(counts) >= N_WORLDS_MIN() and time.time() - t_start > time_budget * 0.5:
                break

        avg_scores = [(scores[i] / counts[i] if counts[i] > 0 else -999.0, i)
                      for i in range(n_cands)]
        avg_scores.sort(reverse=True)
        best_idx = avg_scores[0][1]

        # --- ЛОГИРОВАНИЕ ---
        # obs.select.option
        sel = obs.select
        opts = sel.option
        legal_options = []
        for i, o in enumerate(opts):
            legal_options.append({"idx": i, "desc": _opt_summary(o)})

        # candidates
        cand_desc = []
        for c in candidates:
            if len(c) == 1:
                idx = c[0]
                cand_desc.append({"candidate": legal_options[idx]["desc"], "idx": idx})
            else:
                cand_desc.append({"candidate": f"multi{c}", "idx": c})

        # prior choice (v009a.agent на dict)
        obs_dict_for_prior = None
        try:
            # Конвертируем obs → dict для prior
            import dataclasses
            def _to_dict(obj):
                if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                    return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
                elif isinstance(obj, list):
                    return [_to_dict(x) for x in obj]
                elif hasattr(obj, 'value'):
                    return obj.value
                return obj
            obs_dict_for_prior = _to_dict(obs)
        except Exception:
            pass

        prior_choice = None
        if prior_agent_fn is not None and obs_dict_for_prior is not None:
            try:
                prior_choice = prior_agent_fn(obs_dict_for_prior)
            except Exception as e:
                prior_choice = f"EXC:{e}"

        # per-candidate summary
        per_cand = []
        for i in range(n_cands):
            details = rollout_details[i]
            if not details:
                per_cand.append({"idx": i, "n": 0, "avg_score": None,
                                 "terminal_rate": None, "avg_steps": None})
                continue
            scs = [d["score"] for d in details]
            terms = [d["terminal"] for d in details]
            steps = [d["steps"] for d in details]
            avg = sum(scs) / len(scs)
            var = sum((s - avg) ** 2 for s in scs) / len(scs) if len(scs) > 1 else 0.0
            per_cand.append({
                "idx": i,
                "n": len(details),
                "avg_score": round(avg, 3),
                "variance": round(var, 4),
                "terminal_rate": round(sum(terms) / len(terms), 3),
                "avg_steps": round(sum(steps) / len(steps), 1),
                "scores": [round(s, 1) for s in scs[:20]],  # первые 20 для лога
            })

        # all-zero-score check
        all_zero = all((c == 0 or per_cand[c]["avg_score"] == 0.0)
                       and counts[c] > 0 and per_cand[c]["avg_score"] == 0.0
                       for c in range(n_cands) if counts[c] > 0) and any(counts[c] > 0 for c in range(n_cands))

        # diverges from prior?
        diverges = (prior_choice is not None and isinstance(prior_choice, list)
                    and len(prior_choice) > 0 and candidates[best_idx] != prior_choice)

        entry = {
            "decision_id": len(audit_log["decisions"]),
            "game_id": audit_log.get("_current_game", -1),
            "move_in_game": audit_log.get("_current_move", -1),
            "your_index": your_index,
            "n_legal": len(opts),
            "legal_options": legal_options,
            "n_candidates": n_cands,
            "candidates": cand_desc,
            "chosen_idx": best_idx,
            "chosen_desc": cand_desc[best_idx]["candidate"] if best_idx < len(cand_desc) else "?",
            "prior_choice": prior_choice,
            "prior_desc": (legal_options[prior_choice[0]]["desc"]
                           if isinstance(prior_choice, list) and prior_choice
                           and prior_choice[0] < len(legal_options) else str(prior_choice)),
            "diverges_from_prior": diverges,
            "n_worlds": world_total,
            "per_cand": per_cand,
            "all_zero_scores": all_zero,
            "time_budget": time_budget,
        }
        audit_log["decisions"].append(entry)

        return best_idx, world_total

    return _instrumented_evaluate


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------

def run_one_game(game_id, our_first, seed, prior_mod_for_prior, audit_log):
    v009a_prior_mod = _load_submission_module("v009a_crustle")
    pimc = PIMCAgentV2(
        deck=CRUSTLE_DECK,
        prior_agent=v009a_prior_mod.agent,
        opponent_model="auto",
        epsilon=EPSILON,
        rng_seed=seed,
        **BANK_KW,
    )
    pimc._lazy_init()

    # Патчим _evaluate_candidates на инструментированную версию
    instr_eval = _make_instrumented_evaluate(pimc._pimc_mod, audit_log, prior_mod_for_prior.agent)
    pimc._pimc_mod._evaluate_candidates = instr_eval

    # Оппонент: v001_random
    opp_mod = _load_submission_module("v001_random")
    opp_agent_fn = opp_mod.agent
    opp_deck = list(opp_mod.my_deck) if hasattr(opp_mod, "my_deck") else None
    if opp_deck is None:
        # v001_random read_deck_csv — прочитаем сами
        deck_path = os.path.join(ROOT, "submissions", "v001_random", "deck.csv")
        with open(deck_path) as f:
            opp_deck = [int(x.strip()) for x in f if x.strip()][:60]

    if our_first:
        agent0, agent1 = pimc.agent, opp_agent_fn
        deck0, deck1 = list(CRUSTLE_DECK), list(opp_deck)
    else:
        agent0, agent1 = opp_agent_fn, pimc.agent
        deck0, deck1 = list(opp_deck), list(CRUSTLE_DECK)

    obs_dict, start = battle_start(deck0, deck1)
    if obs_dict is None:
        return dict(outcome="crash", error=f"battle_start: {getattr(start, 'errorType', None)}")

    move_count = 0
    winner = None
    audit_log["_current_game"] = game_id
    game_decision_start = len(audit_log["decisions"])

    try:
        while move_count < MAX_MOVES_PER_GAME:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                winner = obs.current.result
                break

            your_index = obs.current.yourIndex if obs.current else 0
            is_our_turn = (your_index == 0) == our_first
            agent_fn = agent0 if your_index == 0 else agent1
            audit_log["_current_move"] = move_count

            try:
                choice = agent_fn(obs_dict)
            except Exception as e:
                return dict(outcome="exception", exc=str(e), moves=move_count)

            if not _is_legal(choice, obs_dict.get("select")):
                return dict(outcome="illegal", moves=move_count)

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

    game_decisions = audit_log["decisions"][game_decision_start:]
    return dict(
        game_id=game_id, outcome=outcome, winner=winner, moves=move_count,
        our_first=our_first, seed=seed,
        n_decisions_this_game=len(game_decisions),
        bank_spent=pimc.bank_stats["spent_sec"],
        n_search=pimc.bank_stats["n_search_decisions"],
        n_cheap=pimc.bank_stats["n_cheap_decisions"],
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(audit_log):
    decisions = audit_log["decisions"]
    games = audit_log["games"]

    n_search = len(decisions)
    if n_search == 0:
        print("NO SEARCH DECISIONS LOGGED")
        return

    # rollout informativeness
    all_variances = []
    all_terminal_rates = []
    all_avg_steps = []
    all_zero_count = 0
    diverge_count = 0
    pimc_chose_non_attack_when_attack_available = 0
    chose_end_or_retreat = 0
    n_with_data = 0

    for d in decisions:
        per_cand = d["per_cand"]
        cands_with_data = [p for p in per_cand if p["n"] > 0]
        if not cands_with_data:
            continue
        n_with_data += 1

        variances = [p["variance"] for p in cands_with_data]
        avg_var = sum(variances) / len(variances) if variances else 0.0
        all_variances.append(avg_var)

        term_rates = [p["terminal_rate"] for p in cands_with_data]
        all_terminal_rates.append(sum(term_rates) / len(term_rates))

        steps = [p["avg_steps"] for p in cands_with_data]
        all_avg_steps.append(sum(steps) / len(steps))

        if d["all_zero_scores"]:
            all_zero_count += 1

        if d["diverges_from_prior"]:
            diverge_count += 1

        # chosen type
        chosen_desc = d["chosen_desc"]
        # check if ATTACK was available
        has_attack = any("ATTACK" in opt["desc"] for opt in d["legal_options"])
        chosen_is_attack = "ATTACK" in chosen_desc
        if has_attack and not chosen_is_attack:
            pimc_chose_non_attack_when_attack_available += 1
        if "END" in chosen_desc or "RETREAT" in chosen_desc:
            chose_end_or_retreat += 1

    # wins/losses
    wins = sum(1 for g in games if g["outcome"] == "our_win")
    losses = sum(1 for g in games if g["outcome"] == "opp_win")
    draws = sum(1 for g in games if g["outcome"] == "draw")
    decisive = wins + losses
    wr = wins / decisive if decisive else float("nan")

    print("=" * 78)
    print("PIMC BUG AUDIT — РЕЗУЛЬТАТЫ")
    print("=" * 78)
    print(f"\nИгр: {len(games)}  W={wins} L={losses} D={draws}  WR={wr:.3f} (vs random)")
    print(f"Search-развилок инструментировано: {n_search}")
    print(f"  из них с rollout-данными: {n_with_data}")
    print()
    print("ROLLOUT INFORMATIVENESS:")
    print(f"  средняя variance оценок по кандидатам: {sum(all_variances)/len(all_variances):.4f}" if all_variances else "  N/A")
    print(f"  средний terminal rate: {sum(all_terminal_rates)/len(all_terminal_rates):.3f}" if all_terminal_rates else "  N/A")
    print(f"  средняя длина rollout (шаги): {sum(all_avg_steps)/len(all_avg_steps):.1f}" if all_avg_steps else "  N/A")
    print(f"  доля решений где ВСЕ score=0.0 (timeout): {all_zero_count}/{n_with_data} = {all_zero_count/max(n_with_data,1):.3f}")
    print()
    print("ВЫБОР PIMC:")
    print(f"  доля расхождений с prior: {diverge_count}/{n_with_data} = {diverge_count/max(n_with_data,1):.3f}")
    print(f"  PIMC выбрал не-ATTACK при доступном ATTACK: {pimc_chose_non_attack_when_attack_available}/{n_with_data} = {pimc_chose_non_attack_when_attack_available/max(n_with_data,1):.3f}")
    print(f"  PIMC выбрал END/RETREAT: {chose_end_or_retreat}/{n_with_data} = {chose_end_or_retreat/max(n_with_data,1):.3f}")
    print()

    # Топ-5 worst decisions (PIMC выбрал END/RETREAT при доступном ATTACK, all_zero)
    print("ПРИМЕРЫ ПЛОХИХ РЕШЕНИЙ (all_zero_scores=True, выбрал END/RETREAT при ATTACK):")
    bad = [d for d in decisions if d["all_zero_scores"]
           and ("END" in d["chosen_desc"] or "RETREAT" in d["chosen_desc"])
           and any("ATTACK" in o["desc"] for o in d["legal_options"])]
    for d in bad[:5]:
        print(f"  decision #{d['decision_id']} game={d['game_id']} move={d['move_in_game']}")
        print(f"    legal: {[(o['idx'], o['desc']) for o in d['legal_options']]}")
        print(f"    candidates: {[(c['idx'], c['candidate']) for c in d['candidates']]}")
        print(f"    chosen: idx={d['chosen_idx']} desc={d['chosen_desc']}")
        print(f"    prior: {d['prior_desc']}")
        print(f"    per_cand: {[(p['idx'], p['n'], p['avg_score'], p['terminal_rate'], p['avg_steps']) for p in d['per_cand']]}")
        print()

    # Примеры где PIMC расходится с prior и prior = ATTACK
    print("ПРИМЕРЫ РАСХОЖДЕНИЙ (prior=ATTACK, PIMC chose other):")
    diverge_attack = [d for d in decisions if d["diverges_from_prior"]
                      and isinstance(d["prior_choice"], list) and d["prior_choice"]
                      and "ATTACK" in d["prior_desc"]]
    for d in diverge_attack[:5]:
        print(f"  decision #{d['decision_id']} game={d['game_id']} move={d['move_in_game']}")
        print(f"    chosen: {d['chosen_desc']}  prior: {d['prior_desc']}")
        print(f"    per_cand: {[(p['idx'], p['n'], p['avg_score'], p['terminal_rate']) for p in d['per_cand']]}")
        print()

    # Итог по loss turning points
    losing_games = [g for g in games if g["outcome"] == "opp_win"]
    print(f"ПРОИГРЫШЕЙ vs random: {len(losing_games)}")
    for g in losing_games[:3]:
        g_decisions = [d for d in decisions if d["game_id"] == g["game_id"]]
        bad_in_g = [d for d in g_decisions if d["all_zero_scores"]
                    and ("END" in d["chosen_desc"] or "RETREAT" in d["chosen_desc"])
                    and any("ATTACK" in o["desc"] for o in d["legal_options"])]
        print(f"  game {g['game_id']}: moves={g['moves']} bad_decisions={len(bad_in_g)}")
        for d in bad_in_g[:2]:
            print(f"    move {d['move_in_game']}: chose {d['chosen_desc']} (prior={d['prior_desc']}, all_zero={d['all_zero_scores']})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260711)
    args = ap.parse_args()

    audit_log = {"decisions": [], "games": []}

    # prior модуль для prior_choice логирования (fresh, отдельный от pimc's prior)
    prior_mod_for_log = _load_submission_module("v009a_crustle")

    for i in range(args.games):
        our_first = (i % 2 == 0)
        seed = args.seed + i * 7919
        t0 = time.time()
        res = run_one_game(i, our_first, seed, prior_mod_for_log, audit_log)
        res["wall_time"] = time.time() - t0
        audit_log["games"].append(res)
        print(f"game {i+1}/{args.games} first={'Y' if our_first else 'N'} "
              f"outcome={res['outcome']} moves={res['moves']} "
              f"search={res['n_search']} cheap={res['n_cheap']} "
              f"bank={res['bank_spent']:.1f}s wall={res['wall_time']:.1f}s", flush=True)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    # Сохраняем лог
    with open(LOG_PATH, "w") as f:
        json.dump(audit_log, f, indent=2, default=str)
    print(f"\nЛог сохранён: {LOG_PATH}")

    analyze(audit_log)
    return 0


if __name__ == "__main__":
    sys.exit(main())