"""gate_v062_regression.py — Phase 4.1 regression-гейт v062_grimmsnarl_playbook.

Кандидат: submissions/v062_grimmsnarl_playbook (код не трогаем — только читаем).
Движок: cg_native (libcg.dylib, macOS arm64), multiprocessing spawn per-game tasks
(та же схема, что scripts/h3_gate.py::_play_one_game / run_queue — threading запрещён,
движок не thread-safe).

Две группы пар:
  CORE (N=300 каждая, держим полный N всегда):
    v062 vs v017_alakazam_heur
    v062 vs v016_alakazam
    v062 vs v009a_crustle
    v062 vs v044_archaludon_abomasnow   (абсолютный путь — не в этом worktree)
    v062 vs v049_bench_sticky_guard     (абсолютный путь — не в этом worktree)
    v062 vs v061_grimmsnarl_heuristic   (head-to-head, та же дека)
  PROXY (N=300 или меньше при нехватке времени — см. --n-proxy):
    v062 vs meta:MegaLucario_live / TRMewtwo_live / Abomasnow_live (greedy-прокси)
    v061 vs те же 3 прокси (парный бейзлайн)

Вывод:
  results/gates/v062_core_gate.json   (формат как v061_core_gate.json)
  results/gates/v062_proxy_gate.json  (candidate v062/v061 x 3 прокси + delta)

Резюмируемо: пары, уже присутствующие в выходном JSON с n_games >= целевого N,
пропускаются при повторном запуске.

Запуск:
    .venv/bin/python scripts/gate_v062_regression.py --smoke        # 4g/pair sanity
    .venv/bin/python scripts/gate_v062_regression.py --n 300 --n-proxy 300 --workers 8
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
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Нативный движок (macOS arm64). Threading НЕБЕЗОПАСЕН — только multiprocessing spawn.
from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
from cg_native.api import to_observation_class  # noqa: E402
import cg_native  # noqa: E402
import cg_native.api  # noqa: E402
import cg_native.game  # noqa: E402
import cg_native.sim  # noqa: E402
import cg_native.utils  # noqa: E402

# submissions/*/main.py делают `from cg.api import ...` — на macOS это Linux-only .so,
# алиасим на нативные модули (паттерн из scripts/h3_gate.py).
sys.modules.setdefault("cg", cg_native)
sys.modules.setdefault("cg.api", cg_native.api)
sys.modules.setdefault("cg.game", cg_native.game)
sys.modules.setdefault("cg.sim", cg_native.sim)
sys.modules.setdefault("cg.utils", cg_native.utils)

MAX_MOVES_PER_GAME = 2000
GATES_DIR = os.path.join(ROOT, "results", "gates")

MAIN_REPO = "/Users/max/kaggle-pokemon-tcg-ai-battle"

# ---------------------------------------------------------------------------
# Мета-дек greedy-attack прокси (ПОРТ 1:1 из scripts/h3_gate.py — без lgb,
# без cg.api top-level импорта).
# ---------------------------------------------------------------------------

_ATTACK_DAMAGE_MAP: dict[int, int] | None = None


def _get_attack_damage_map() -> dict[int, int]:
    global _ATTACK_DAMAGE_MAP
    if _ATTACK_DAMAGE_MAP is None:
        try:
            from cg_native.api import all_attack
        except ImportError:
            from cg.api import all_attack  # type: ignore[no-redef]
        _ATTACK_DAMAGE_MAP = {a.attackId: (a.damage or 0) for a in all_attack()}
    return _ATTACK_DAMAGE_MAP


def _opt_get(o, key: str, default=None):
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


_OT_YES = 1
_OT_CARD = 3
_OT_PLAY = 7
_OT_ATTACH = 8
_OT_EVOLVE = 9
_OT_ABILITY = 10
_OT_RETREAT = 12
_OT_ATTACK = 13
_OT_END = 14
_ST_YES_NO = 9
_ST_COUNT = 8
_SC_MAIN = 0
_GREEDY_PRIORITY: tuple[int, ...] = (
    _OT_ATTACK, _OT_ABILITY, _OT_EVOLVE, _OT_PLAY, _OT_ATTACH, _OT_RETREAT, _OT_CARD, _OT_END,
)


def greedy_meta_agent(obs_dict: dict) -> list[int]:
    """Generic greedy-attack пилот для мета-дек-оппонентов (см. scripts/h3_gate.py)."""
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return []
    opts = sel.get("option") or []
    n = len(opts)
    if n == 0:
        return []
    sel_type = sel.get("type")
    if sel_type == _ST_YES_NO:
        for i in range(n):
            if _opt_get(opts[i], "type") == _OT_YES:
                return [i]
        return [0]
    if sel_type == _ST_COUNT:
        best = max(range(n), key=lambda i: (_opt_get(opts[i], "number", 0) or 0))
        return [best]
    min_c = int(sel.get("minCount") or 1)
    max_c = int(sel.get("maxCount") or 1)
    context = sel.get("context")
    if context == _SC_MAIN:
        atk_idxs = [i for i in range(n) if _opt_get(opts[i], "type") == _OT_ATTACK]
        if atk_idxs:
            dmg_map = _get_attack_damage_map()
            best = max(atk_idxs, key=lambda i: (dmg_map.get(_opt_get(opts[i], "attackId"), 0) or 0))
            return [best]
        for ot in _GREEDY_PRIORITY:
            idxs = [i for i in range(n) if _opt_get(opts[i], "type") == ot]
            if idxs:
                return [idxs[0]]
    k = max(1, min_c)
    k = min(k, max_c, n)
    return list(range(k))


# ---------------------------------------------------------------------------
# Загрузка submission-модулей и дек
# ---------------------------------------------------------------------------

_LOAD_COUNTER = 0


def _load_submission_module(sub_dir: str):
    """Загружает <sub_dir>/main.py свежим instance'ом модуля (fresh per call —
    защита от межигрового stateful-протекания, как в h3_gate.py)."""
    global _LOAD_COUNTER
    _LOAD_COUNTER += 1
    main_path = os.path.join(sub_dir, "main.py")
    assert os.path.exists(main_path), f"не найден {main_path}"
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    mod_name = f"_gatev062_{os.path.basename(sub_dir)}_{_LOAD_COUNTER}"
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


def _load_meta_deck(deck_name: str) -> list[int]:
    return _load_deck_csv(os.path.join(ROOT, "decks", "meta", f"{deck_name}.csv"))


def resolve_dir(spec: str) -> str:
    """Абсолютный путь как есть; относительный — от ROOT (worktree)."""
    if os.path.isabs(spec):
        return spec
    return os.path.join(ROOT, spec)


# ---------------------------------------------------------------------------
# Вспомогательные функции статистики
# ---------------------------------------------------------------------------

def _is_legal(choice, sel: dict | None) -> bool:
    if sel is None:
        return isinstance(choice, list) and len(choice) == 60 and all(isinstance(c, int) for c in choice)
    opts = sel.get("option") or []
    n_opts = len(opts)
    # ВАЖНО: minCount=0 легален (пустой выбор) — `sel.get(...) or 1` идиома (как в
    # scripts/h3_gate.py::_is_legal) ломает это, т.к. 0 falsy. Использовать `is None`.
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


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def wilson_ci(wins: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def two_prop_ztest(x1: float, n1: float, x2: float, n2: float) -> tuple[float, float]:
    """Двухсторонний двухвыборочный z-тест разности долей p1-p2. Возвращает (z, p_value).
    x1/x2 могут быть 'win-equivalent' (wins + 0.5*draws) — дробные, тест на пропорциях
    это допускает (используется как приближение)."""
    if n1 <= 0 or n2 <= 0:
        return (0.0, 1.0)
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(max(p_pool * (1 - p_pool), 1e-12) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    # двухсторонний p-value через erf
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return (z, p_value)


# ---------------------------------------------------------------------------
# Worker: _play_one_game — всё создаётся внутри воркера (spawn-safe)
# ---------------------------------------------------------------------------

def _play_one_game(task: dict) -> dict:
    cell_id = task["cell_id"]
    candidate_dir = task["candidate_dir"]
    opp_spec = task["opp_spec"]
    our_first = task["our_first"]
    seed = task["seed"]
    game_idx = task["game_idx"]

    import random
    random.seed(seed)

    cand_mod = _load_submission_module(candidate_dir)
    cand_agent = cand_mod.agent
    cand_deck = _resolve_deck(candidate_dir, cand_mod)

    if opp_spec.startswith("meta:"):
        deck_name = opp_spec[5:]
        opp_deck = _load_meta_deck(deck_name)
        opp_agent = greedy_meta_agent
    else:
        assert opp_spec.startswith("sub:"), f"bad opp_spec: {opp_spec}"
        opp_dir = opp_spec[4:]
        opp_mod = _load_submission_module(opp_dir)
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
        return dict(
            cell_id=cell_id, game_idx=game_idx, our_first=our_first, seed=seed,
            outcome="crash", exc_side=None, winner=None, moves=0,
            n_illegal=0, illegal_side=None, n_exceptions=1,
            elapsed_s=time.perf_counter() - t0,
        )

    move_count = 0
    n_illegal = 0
    illegal_side = None
    n_exceptions = 0
    exc_side = None
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

            try:
                choice = agent_fn(obs_dict)
            except Exception:
                n_exceptions += 1
                exc_side = "our" if is_our_turn else "opp"
                break

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
        cell_id=cell_id, game_idx=game_idx, our_first=our_first, seed=seed,
        outcome=outcome, exc_side=exc_side, illegal_side=illegal_side, winner=winner,
        moves=move_count, n_illegal=n_illegal, n_exceptions=n_exceptions,
        elapsed_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Раннер одной пары (N игр, seat-balanced, ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def run_pair(candidate_dir: str, opp_spec: str, cell_id: str, n_games: int,
             base_seed: int, workers: int) -> dict:
    ctx = multiprocessing.get_context("spawn")
    queue: list[dict] = []
    for game_idx in range(n_games):
        our_first = (game_idx % 2 == 0)
        seed = base_seed + (hash((cell_id, game_idx)) % 1_000_000)
        queue.append(dict(
            cell_id=cell_id, candidate_dir=candidate_dir, opp_spec=opp_spec,
            our_first=our_first, seed=seed, game_idx=game_idx,
        ))

    t_start = time.time()
    games: list[dict] = []
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
                    res = dict(
                        cell_id=t["cell_id"], game_idx=t["game_idx"], our_first=t["our_first"],
                        seed=t["seed"], outcome="worker_crash", exc_side="our", illegal_side=None,
                        winner=None, moves=0, n_illegal=0, n_exceptions=1, elapsed_s=0.0,
                        error=str(e),
                    )
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
    exc_agent = sum(1 for g in games if g["outcome"] == "exception" and g.get("exc_side") == "our")
    exc_opp = sum(1 for g in games if g["outcome"] == "exception" and g.get("exc_side") == "opp")
    illegal_agent = sum(1 for g in games if g["outcome"] == "illegal" and g.get("illegal_side") == "our")
    illegal_opp = sum(1 for g in games if g["outcome"] == "illegal" and g.get("illegal_side") == "opp")
    timeouts = sum(1 for g in games if g["outcome"] == "timeout")
    crashes = sum(1 for g in games if g["outcome"] in ("crash", "worker_crash"))
    lengths = [g["moves"] for g in games if g["moves"] > 0]

    n = len(games)
    we = wins + 0.5 * draws
    win_rate = we / n if n else 0.0
    wlo, whi = wilson_ci(we, n)
    decisive_n = wins + losses
    decisive_win_rate = wins / decisive_n if decisive_n else 0.0

    return dict(
        opponent=cell_id,
        opponent_dir=opp_spec,
        n_games=n,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=win_rate,
        wilson95_lo=wlo,
        wilson95_hi=whi,
        decisive_win_rate=decisive_win_rate,
        decisive_n=decisive_n,
        avg_game_length=(sum(lengths) / len(lengths) if lengths else 0.0),
        min_game_length=(min(lengths) if lengths else 0),
        max_game_length=(max(lengths) if lengths else 0),
        exc_agent=exc_agent,
        exc_opp=exc_opp,
        illegal_agent=illegal_agent,
        illegal_opp=illegal_opp,
        timeouts=timeouts,
        crashes=crashes,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Pair configs
# ---------------------------------------------------------------------------

V062_DIR = resolve_dir("submissions/v062_grimmsnarl_playbook")
V061_DIR = resolve_dir("submissions/v061_grimmsnarl_heuristic")

# Baseline v061_core_gate.json exact counts (2026-07-21, N=300, cg_native) — для z-теста.
BASELINE = {
    "v017": dict(wins=281, losses=19, draws=0, n=300, win_rate=0.9366666666666666),
    "v016": dict(wins=284, losses=16, draws=0, n=300, win_rate=0.9466666666666667),
    "v009a": dict(wins=50, losses=249, draws=1, n=300, win_rate=0.16833333333333333),
    "v044": dict(wins=58, losses=242, draws=0, n=300, win_rate=0.19333333333333333),
    "v049": dict(wins=57, losses=243, draws=0, n=300, win_rate=0.19),
}

CORE_PAIRS = [
    dict(key="v017", opp_dir=resolve_dir("submissions/v017_alakazam_heur"), baseline="v017"),
    dict(key="v016", opp_dir=resolve_dir("submissions/v016_alakazam"), baseline="v016"),
    dict(key="v009a", opp_dir=resolve_dir("submissions/v009a_crustle"), baseline="v009a"),
    dict(key="v044", opp_dir=resolve_dir(MAIN_REPO + "/submissions/v044_archaludon_abomasnow"), baseline="v044"),
    dict(key="v049", opp_dir=resolve_dir(MAIN_REPO + "/submissions/v049_bench_sticky_guard"), baseline="v049"),
    dict(key="v061_h2h", opp_dir=V061_DIR, baseline=None),
]

PROXY_DECKS = ["MegaLucario_live", "TRMewtwo_live", "Abomasnow_live"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_existing(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="Игр на CORE-пару")
    ap.add_argument("--n-proxy", type=int, default=300, help="Игр на PROXY-пару")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--groups", default="core,proxy", help="core,proxy")
    ap.add_argument("--only", default=None, help="Запустить только эту cell_id (debug)")
    ap.add_argument("--smoke", action="store_true", help="4 игры/пара, sanity-check")
    ap.add_argument("--out-core", default=os.path.join(GATES_DIR, "v062_core_gate.json"))
    ap.add_argument("--out-proxy", default=os.path.join(GATES_DIR, "v062_proxy_gate.json"))
    args = ap.parse_args()

    n_core = 4 if args.smoke else args.n
    n_proxy = 4 if args.smoke else args.n_proxy
    groups = set(args.groups.split(","))

    if "core" in groups:
        out_path = args.out_core
        existing = load_existing(out_path)
        opponents = existing.get("opponents", {})
        for cfg in CORE_PAIRS:
            key = cfg["key"]
            if args.only and key != args.only:
                continue
            have = opponents.get(key, {})
            if have.get("n_games", 0) >= n_core:
                print(f"[core] {key}: already have n_games={have['n_games']} >= {n_core}, skip")
                continue
            print(f"[core] {key}: running n={n_core} vs {cfg['opp_dir']} ...", flush=True)
            t0 = time.time()
            stats = run_pair(V062_DIR, "sub:" + cfg["opp_dir"], key, n_core, args.seed, args.workers)
            print(f"[core] {key}: done in {time.time()-t0:.1f}s  "
                  f"WR={stats['win_rate']:.4f} [{stats['wilson95_lo']:.3f},{stats['wilson95_hi']:.3f}]  "
                  f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']}  "
                  f"exc={stats['exc_agent']}/{stats['exc_opp']} illegal={stats['illegal_agent']}/{stats['illegal_opp']} "
                  f"timeouts={stats['timeouts']} crashes={stats['crashes']}", flush=True)
            opponents[key] = stats
            existing = dict(
                tag="v062_core_gate",
                agent=dict(ckpt=None, agent_dir="v062_grimmsnarl_playbook",
                            deck=os.path.join(V062_DIR, "deck.csv"),
                            label="submission:v062_grimmsnarl_playbook"),
                agent_ckpt_exceptions=None,
                opponents=opponents,
                overall_win_rate=(sum(o["wins"] + 0.5 * o["draws"] for o in opponents.values())
                                    / max(1, sum(o["n_games"] for o in opponents.values()))),
                generated=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            save_json(out_path, existing)

    if "proxy" in groups:
        out_path = args.out_proxy
        existing = load_existing(out_path)
        proxies = existing.get("proxies", {})
        for deck_name in PROXY_DECKS:
            proxies.setdefault(deck_name, {})
            for cand_label, cand_dir in (("v062", V062_DIR), ("v061", V061_DIR)):
                cell_id = f"{cand_label}_vs_{deck_name}"
                if args.only and cell_id != args.only:
                    continue
                have = proxies[deck_name].get(cand_label, {})
                if have.get("n_games", 0) >= n_proxy:
                    print(f"[proxy] {cell_id}: already have n_games={have['n_games']} >= {n_proxy}, skip")
                    continue
                print(f"[proxy] {cell_id}: running n={n_proxy} ...", flush=True)
                t0 = time.time()
                stats = run_pair(cand_dir, "meta:" + deck_name, cell_id, n_proxy, args.seed, args.workers)
                print(f"[proxy] {cell_id}: done in {time.time()-t0:.1f}s  "
                      f"WR={stats['win_rate']:.4f} [{stats['wilson95_lo']:.3f},{stats['wilson95_hi']:.3f}]  "
                      f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']}  "
                      f"exc={stats['exc_agent']}/{stats['exc_opp']} illegal={stats['illegal_agent']}/{stats['illegal_opp']} "
                      f"timeouts={stats['timeouts']} crashes={stats['crashes']}", flush=True)
                proxies[deck_name][cand_label] = stats
                # delta v062 - v061 (parametric two-prop z-test on win-equivalent counts)
                if "v062" in proxies[deck_name] and "v061" in proxies[deck_name]:
                    a = proxies[deck_name]["v062"]
                    b = proxies[deck_name]["v061"]
                    we_a = a["wins"] + 0.5 * a["draws"]
                    we_b = b["wins"] + 0.5 * b["draws"]
                    z, p = two_prop_ztest(we_a, a["n_games"], we_b, b["n_games"])
                    proxies[deck_name]["delta_v062_minus_v061"] = a["win_rate"] - b["win_rate"]
                    proxies[deck_name]["ztest_z"] = z
                    proxies[deck_name]["ztest_p_value"] = p
                existing = dict(
                    tag="v062_proxy_gate",
                    candidates=dict(v062=V062_DIR, v061=V061_DIR),
                    proxy_decks=PROXY_DECKS,
                    proxies=proxies,
                    generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                save_json(out_path, existing)

    print("\n=== DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
