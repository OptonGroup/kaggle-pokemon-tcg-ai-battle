"""measure_search_passivity.py — честный инструмент END-пассивности ВНУТРИ поиска.

Мотивация (ТЗ 2026-08-08/09): счётчик `end_at_atk` в submissions/scout_r1a_endturn/
main.py:791-795 считает значение ПОСЛЕ _p0_p1_guard (main.py:211-238), который сам
подменяет END на ATTACK, когда ATTACK доступна — поэтому structurally он ~0% и не
воспроизводит замер, на который ссылается PLAN_TOP30_20260808.md ("48%, 461 случай").

Этот скрипт снимает ОБЕ величины раздельно через monkey-patch `search_choice` в
свежезагруженном инстансе кандидата (эталоны submissions/scout_deeper[_p0p1] и
submissions/scout_r1a_endturn НЕ модифицируются):
  - RAW-argmax: то, что вернул сырой `search_choice(obs)` — ПОСЛЕ поиска, но ДО
    любого guard'а (scout_deeper вообще не вызывает guard; p0p1/r1a вызывают его
    в agent() уже после того, как search_choice отработал).
  - FINAL: то, что реально вернул `agent(obs_dict)` (после guard'а, если он есть).

Оба репортятся раздельно, по типам опций (ATTACK-only и "широкое" определение —
ATTACK/EVOLVE/ATTACH-на-активного/PLAY-базового-при-пустой-скамейке — то же, что
`is_useful()` внутри `_p0_p1_guard`), и только для решений с src=search (agent()
переходит в src=search тогда и только тогда, когда search_choice() вернул не-None —
тот же критерий воспроизведён здесь).

Движок: cg_native через sys.modules-алиасы (см. scripts/gate_scout_guard.py,
project_macos_stand_silently_disables_search) — работает на macOS ARM, поиск НЕ
молча отключается (проверяется по наличию src=search строк в отчёте).

Запуск:
    .venv/bin/python scripts/measure_search_passivity.py \
        --configs deeper,deeper_p0p1,r1a_off,r1a_on \
        --opponents ext2_tetsutani_grimmsnarl,v044_archaludon \
        --n 16 --samples 16 --workers 4 --seed-base 1000 \
        --out out/search_passivity/run1.json

cg_native не принимает сид и его RNG не сбрасывается random.seed() (project note:
13/25 расхождений на "идентичных" снимках) — сравнения ниже интерпретируются как
НЕЗАВИСИМЫЕ выборки, не как парный дизайн.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
    from cg_native.api import to_observation_class  # noqa: E402
    import cg_native  # noqa: E402
    import cg_native.api, cg_native.game, cg_native.sim, cg_native.utils  # noqa: E402
    sys.modules.setdefault("cg", cg_native)
    sys.modules.setdefault("cg.api", cg_native.api)
    sys.modules.setdefault("cg.game", cg_native.game)
    sys.modules.setdefault("cg.sim", cg_native.sim)
    sys.modules.setdefault("cg.utils", cg_native.utils)
    ENGINE = "cg_native"
except Exception:
    from cg.game import battle_start, battle_finish, battle_select  # noqa: E402
    from cg.api import to_observation_class  # noqa: E402
    ENGINE = "cg"

MAX_MOVES_PER_GAME = 2000
CTX_MAIN = 0
OT_PLAY, OT_ATTACH, OT_EVOLVE, OT_ATTACK, OT_END = 7, 8, 9, 13, 14
BASIC_POKEMON_IDS = {42, 89, 100, 343}

CONFIGS = {
    "deeper": dict(dir="submissions/scout_deeper", env={}),
    "deeper_p0p1": dict(dir="submissions/scout_deeper_p0p1", env={}),
    "r1a_off": dict(dir="submissions/scout_r1a_endturn", env={"SCOUT_R1A_ENDTURN": "0"}),
    "r1a_on": dict(dir="submissions/scout_r1a_endturn", env={"SCOUT_R1A_ENDTURN": "1"}),
}

OPPONENTS = {
    "ext2_tetsutani_grimmsnarl": "submissions/external_tetsutani_grimmsnarl",
    "v044_archaludon": "submissions/v044_archaludon_abomasnow",
}


def resolve_dir(spec: str) -> str:
    return spec if os.path.isabs(spec) else os.path.join(ROOT, spec)


_LOAD_COUNTER = 0


def _load_submission_module(sub_dir: str, env_overrides: dict, mod_tag: str):
    global _LOAD_COUNTER
    _LOAD_COUNTER += 1
    for k, v in env_overrides.items():
        os.environ[k] = v
    main_path = os.path.join(sub_dir, "main.py")
    assert os.path.exists(main_path), f"не найден {main_path}"
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    mod_name = f"_msp_{mod_tag}_{os.path.basename(sub_dir)}_{_LOAD_COUNTER}"
    original_cwd = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(mod_name, main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(original_cwd)
    return mod


def _load_deck_csv(path: str) -> list[int]:
    with open(path) as f:
        ids = [int(x.strip()) for x in f if x.strip()]
    if len(ids) != 60:
        raise ValueError(f"{path}: expected 60 cards, got {len(ids)}")
    return ids


def _resolve_deck(sub_dir: str, mod) -> list[int]:
    deck = getattr(mod, "my_deck", None)
    if deck:
        return list(deck)
    return _load_deck_csv(os.path.join(sub_dir, "deck.csv"))


def _is_legal(choice, sel: dict | None) -> bool:
    if sel is None:
        return isinstance(choice, list) and len(choice) == 60 and all(isinstance(c, int) for c in choice)
    opts = sel.get("option") or []
    n_opts = len(opts)
    _min_raw = sel.get("minCount")
    _max_raw = sel.get("maxCount")
    min_c = 0 if _min_raw is None else int(_min_raw)
    max_c = 1 if _max_raw is None else int(_max_raw)
    return (
        isinstance(choice, list)
        and all(isinstance(i, int) and 0 <= i < n_opts for i in choice)
        and min_c <= len(choice) <= max_c
        and len(set(choice)) == len(choice)
    )


def _g(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        v = obj.get(key, default)
    else:
        v = getattr(obj, key, default)
    return default if v is None else v


def _is_useful_broad(o, bench_empty: bool, hand: list) -> bool:
    t = _g(o, "type")
    if t == OT_ATTACK or t == OT_EVOLVE:
        return True
    if t == OT_ATTACH:
        return _g(o, "inPlayArea") == 4  # AREA_ACTIVE
    if t == OT_PLAY and bench_empty:
        idx = _g(o, "index")
        if idx is not None and 0 <= idx < len(hand):
            return _g(hand[idx], "id") in BASIC_POKEMON_IDS
    return False


def _classify_decision(obs_dict: dict, sel: dict) -> dict | None:
    """Возвращает дескриптор MAIN-решения кандидата ДО его хода, или None если не CTX_MAIN."""
    if sel is None or sel.get("context") != CTX_MAIN:
        return None
    options = sel.get("option") or []
    if len(options) < 2:
        return None
    state = obs_dict.get("current") or {}
    you = state.get("yourIndex", 0)
    players = state.get("players") or []
    mine = players[you] if len(players) > you else None
    bench_empty = not (_g(mine, "bench", []) or [])
    hand = _g(mine, "hand", []) or []
    attack_avail = any(_g(o, "type") == OT_ATTACK for o in options)
    broad_avail = any(_is_useful_broad(o, bench_empty, hand) for o in options)
    return dict(options=options, attack_avail=attack_avail, broad_avail=broad_avail)


def _is_end(options: list, choice) -> bool | None:
    if not isinstance(choice, list) or len(choice) != 1:
        return False
    idx = choice[0]
    if not (0 <= idx < len(options)):
        return False
    return _g(options[idx], "type") == OT_END


def _play_one_game(task: dict) -> dict:
    cfg_key = task["cfg_key"]
    cfg = CONFIGS[cfg_key]
    candidate_dir = resolve_dir(cfg["dir"])
    opp_dir = resolve_dir(OPPONENTS[task["opp_key"]])
    our_first = task["our_first"]
    seed = task["seed"]
    samples_override = task.get("samples_override")

    import random
    random.seed(seed)

    cand_mod = _load_submission_module(candidate_dir, cfg["env"], "cand")
    if samples_override is not None and hasattr(cand_mod, "SEARCH_CFG"):
        cand_mod.SEARCH_CFG["samples"] = samples_override

    # monkey-patch: перехватываем RAW pre-guard argmax, не трогая файл на диске.
    _raw_capture = {"v": None}
    if hasattr(cand_mod, "search_choice"):
        _orig_search_choice = cand_mod.search_choice

        def _wrapped_search_choice(obs, _orig=_orig_search_choice, _cap=_raw_capture):
            r = _orig(obs)
            _cap["v"] = r
            return r

        cand_mod.search_choice = _wrapped_search_choice

    cand_agent = cand_mod.agent
    cand_deck = _resolve_deck(candidate_dir, cand_mod)

    opp_mod = _load_submission_module(opp_dir, {}, "opp")
    opp_agent = opp_mod.agent
    opp_deck = _resolve_deck(opp_dir, opp_mod)

    if our_first:
        deck0, deck1 = list(cand_deck), list(opp_deck)
        agent0, agent1 = cand_agent, opp_agent
    else:
        deck0, deck1 = list(opp_deck), list(cand_deck)
        agent0, agent1 = opp_agent, cand_agent

    t0 = time.perf_counter()
    obs_dict, start = battle_start(deck0, deck1)
    if obs_dict is None:
        return dict(cfg_key=cfg_key, opp_key=task["opp_key"], game_idx=task["game_idx"],
                    outcome="crash", winner=None, moves=0, decisions=[],
                    elapsed_s=time.perf_counter() - t0)

    decisions = []
    move_count = 0
    n_exceptions = 0
    exc_side = None
    n_illegal = 0
    illegal_side = None
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

            pre = _classify_decision(obs_dict, sel) if is_our_turn else None
            if pre is not None:
                _raw_capture["v"] = None

            try:
                choice = agent_fn(obs_dict)
            except Exception:
                n_exceptions += 1
                exc_side = "our" if is_our_turn else "opp"
                break

            if pre is not None:
                raw = _raw_capture["v"]
                src = "search" if raw is not None else "heur"
                rec = dict(
                    attack_avail=pre["attack_avail"],
                    broad_avail=pre["broad_avail"],
                    src=src,
                    final_end=_is_end(pre["options"], choice),
                )
                if raw is not None:
                    rec["raw_end"] = _is_end(pre["options"], raw)
                decisions.append(rec)

            if not _is_legal(choice, sel):
                n_illegal += 1
                illegal_side = "our" if is_our_turn else "opp"
                break

            obs_dict = battle_select(choice)
            move_count += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass

    our_side = 0 if our_first else 1
    aborted = (winner is None and n_exceptions == 0 and n_illegal == 0)
    if n_exceptions > 0:
        outcome = "exception"
    elif n_illegal > 0:
        outcome = "illegal"
    elif aborted:
        outcome = "timeout"
    elif winner == our_side:
        outcome = "our_win"
    elif winner == 2:
        outcome = "draw"
    else:
        outcome = "opp_win"

    return dict(
        cfg_key=cfg_key, opp_key=task["opp_key"], game_idx=task["game_idx"],
        our_first=our_first, seed=seed, outcome=outcome, exc_side=exc_side,
        illegal_side=illegal_side, winner=winner, moves=move_count,
        decisions=decisions, elapsed_s=time.perf_counter() - t0,
    )


def run_cell(cfg_key: str, opp_key: str, n_games: int, base_seed: int, workers: int,
             samples_override: int | None) -> dict:
    ctx = multiprocessing.get_context("spawn")
    queue = []
    for game_idx in range(n_games):
        our_first = (game_idx % 2 == 0)
        seed = base_seed + (zlib.crc32(f"{cfg_key}|{opp_key}|{game_idx}".encode()) % 1_000_000)
        queue.append(dict(cfg_key=cfg_key, opp_key=opp_key, our_first=our_first,
                           seed=seed, game_idx=game_idx, samples_override=samples_override))

    t_start = time.time()
    games = []
    q_iter = iter(queue)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = {}
        for _ in range(workers):
            try:
                t = next(q_iter)
            except StopIteration:
                break
            futures[ex.submit(_play_one_game, t)] = t
        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                t = futures.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:
                    res = dict(cfg_key=t["cfg_key"], opp_key=t["opp_key"], game_idx=t["game_idx"],
                                outcome="worker_crash", winner=None, moves=0, decisions=[],
                                elapsed_s=0.0, error=str(e))
                games.append(res)
                try:
                    t2 = next(q_iter)
                    futures[ex.submit(_play_one_game, t2)] = t2
                except StopIteration:
                    pass
    elapsed = time.time() - t_start

    wins = sum(1 for g in games if g["outcome"] == "our_win")
    losses = sum(1 for g in games if g["outcome"] == "opp_win")
    draws = sum(1 for g in games if g["outcome"] == "draw")
    crashes = sum(1 for g in games if g["outcome"] in ("crash", "worker_crash", "exception", "illegal", "timeout"))

    all_decisions = [d for g in games for d in g.get("decisions", [])]
    search_dec = [d for d in all_decisions if d["src"] == "search"]
    heur_dec = [d for d in all_decisions if d["src"] == "heur"]

    def _rate(dec_list, avail_key, end_key):
        elig = [d for d in dec_list if d[avail_key] and end_key in d]
        if not elig:
            return None, 0
        hits = sum(1 for d in elig if d[end_key])
        return hits / len(elig), len(elig)

    raw_attack_rate, raw_attack_n = _rate(search_dec, "attack_avail", "raw_end")
    raw_broad_rate, raw_broad_n = _rate(search_dec, "broad_avail", "raw_end")
    final_attack_rate_search, final_attack_n_search = _rate(search_dec, "attack_avail", "final_end")
    final_broad_rate_search, final_broad_n_search = _rate(search_dec, "broad_avail", "final_end")
    final_attack_rate_all, final_attack_n_all = _rate(all_decisions, "attack_avail", "final_end")

    n_src_search_lines = len(search_dec)

    return dict(
        cfg=cfg_key, opponent=opp_key, n_games=len(games),
        wins=wins, losses=losses, draws=draws, crashes_or_aborts=crashes,
        elapsed_s=elapsed,
        n_main_decisions=len(all_decisions),
        n_search_decisions=n_src_search_lines,
        n_heur_decisions=len(heur_dec),
        search_share=(n_src_search_lines / len(all_decisions) if all_decisions else None),
        raw_end_rate_given_attack=raw_attack_rate, raw_end_rate_given_attack_n=raw_attack_n,
        raw_end_rate_given_broad=raw_broad_rate, raw_end_rate_given_broad_n=raw_broad_n,
        final_end_rate_given_attack_searchonly=final_attack_rate_search, final_end_rate_given_attack_searchonly_n=final_attack_n_search,
        final_end_rate_given_broad_searchonly=final_broad_rate_search, final_end_rate_given_broad_searchonly_n=final_broad_n_search,
        final_end_rate_given_attack_all=final_attack_rate_all, final_end_rate_given_attack_all_n=final_attack_n_all,
        samples_override=samples_override,
        engine=ENGINE,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS.keys()))
    ap.add_argument("--opponents", default=",".join(OPPONENTS.keys()))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--samples", type=int, default=None, help="override SEARCH_CFG.samples (prod=64)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_keys = [c.strip() for c in args.configs.split(",") if c.strip()]
    opp_keys = [o.strip() for o in args.opponents.split(",") if o.strip()]

    print(f"[measure] engine={ENGINE} configs={cfg_keys} opponents={opp_keys} "
          f"n={args.n} samples_override={args.samples} workers={args.workers}", file=sys.stderr)

    results = []
    for cfg_key in cfg_keys:
        for opp_key in opp_keys:
            t0 = time.time()
            r = run_cell(cfg_key, opp_key, args.n, args.seed_base, args.workers, args.samples)
            dt = time.time() - t0
            print(f"[measure] {cfg_key:14s} vs {opp_key:28s} "
                  f"games={r['n_games']} W-L-D={r['wins']}-{r['losses']}-{r['draws']} "
                  f"search_dec={r['n_search_decisions']}/{r['n_main_decisions']} "
                  f"RAW-END|atk={_fmt(r['raw_end_rate_given_attack'])}(n={r['raw_end_rate_given_attack_n']}) "
                  f"RAW-END|broad={_fmt(r['raw_end_rate_given_broad'])}(n={r['raw_end_rate_given_broad_n']}) "
                  f"FINAL-END|atk(search)={_fmt(r['final_end_rate_given_attack_searchonly'])} "
                  f"FINAL-END|atk(all)={_fmt(r['final_end_rate_given_attack_all'])} "
                  f"[{dt:.0f}s]", file=sys.stderr)
            results.append(r)

    out = dict(engine=ENGINE, args=vars(args), results=results)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[measure] written {args.out}", file=sys.stderr)
    else:
        print(json.dumps(out, indent=2))
    return 0


def _fmt(x):
    return "n/a" if x is None else f"{x*100:.1f}%"


if __name__ == "__main__":
    sys.exit(main())
