"""gate_v073_rockinn.py — regression + band-weighted gate for the v073 Rock-Inn candidates.

Отличия от родителя scripts/gate_v072_band.py (скопирован 2026-07-24, логика прогона пар,
бэнд-агрегация и acceptance не тронуты):
  * кандидат параметризован: --candidate <dir> --label <tag> (по умолчанию
    submissions/v073_rockinn_damage / "v073"), выходные JSON именуются от label;
  * baseline для регрессий — results/gates/v072_core_gate.json (фул N=1000 гейт v072,
    залитый на ладдер), а не v065;
  * в CORE добавлена пара v072_h2h (прямое зеркало против действующего сабмита) — именно
    она отвечает на вопрос "не хуже ли предшественника"; ключ v065_h2h оставлен для
    сопоставимости с прошлым прогоном.

Правки, которые гейт должен проверить (см. out/v072_wr_check/ANALYSIS_v072.md, ТЗ v073):
  P0 (обе сборки): единый Rock-Inn-корректный расчёт нашего урона — ожидаемое движение
     ТОЛЬКО в crustle-парах (v009a, ext_tusk_crustle); остальные пары = контроль на
     отсутствие побочек.
  P1 (только v073b_park): добровольная парковка здоровой ex при матчпойнте и OHKO-теле
     на скамейке соперника — меряется как дельта v073b − v073.

Ниже — докстринг родителя (актуален по механике прогона).

gate_v072_band.py — regression + band-weighted gate for submissions/v072_mirror_gauntlet
(адаптация scripts/gate_v065_regression.py — cg_native движок, multiprocessing spawn per-game,
sys.modules алиасы cg->cg_native, resolve_dir, резюмируемость по существующему JSON).

Кандидат: submissions/v072_mirror_gauntlet (код не трогаем — только читаем).

Две группы пар (как в v065-гейте):
  CORE (N=300 по умолчанию, держим полный N всегда):
    v017, v016, v009a, v044, v049                              — базовый пул
    v065_h2h  (submissions/v065_defensive_swap)                 — БАЗА, регрессии h2h меряем
                                                                   относительно неё (см. ниже)
    v063_h2h  (submissions/v063_forensic86)
    v070_h2h  (.claude/worktrees/v070-grimmsnarl-hybrid/submissions/v070_grimmsnarl_hybrid —
               attention-BC зеркало, torch; НЕ в submissions/ этого чекаута, т.к. живёт на
               отдельной незамерженной ветке worktree-v070-grimmsnarl-hybrid)
    ext_prvsiyan_alak_v8 / v9, ext_maulikgajera_lucario, ext_nursrijan_lucario, v036_h2h
    ext_tusk_crustle  (data/external_agents/tusk_crustle_libout/tusk_crustle_submission —
               ранее подтверждена рабочей парой: scripts/gate_v070_vs_tusk.py, 500 игр,
               0 crashes/exceptions, см. results/gates/v070_vs_tusk.json)
  PROXY (N=300 по умолчанию):
    v072 vs meta:Starmie_live (greedy-прокси, единственный архетип без готового ready-агента —
               используется как компонент "starmie" в бэнд-агрегации)

БЕЙЗЛАЙН для регрессий: results/gates/v065_core_gate.json (фул N=1000 гейт v065). Совпадающие
по имени ключи (v017/v016/v009a/v044/v049/v063_h2h/ext_*/v036_h2h) используются как historical
baseline win_rate; двухпропорционный z-тест (см. two_prop_ztest), регрессия = p<0.05 И
candidate_wr < baseline_wr. Ключи v065_h2h/v070_h2h/ext_tusk_crustle не встречаются в
v065_core_gate.json (v065 никогда не играл сам с собой/tusk/v070) -> baseline=None, только отчёт.
Если файла v065_core_gate.json нет вовсе — все пары без baseline, п.3 acceptance vacuously PASS.

БЭНД-АГРЕГАЦИЯ (results/gates/v072_band_summary.json + печать таблицы):
  archetype WR = среднее WR по CORE-парам (см. ARCHETYPE_MAP), starmie = PROXY Starmie_live,
  dragapult/rest = фиксировано 0.50 (нет данных). Если у архетипа отсутствуют исходные данные
  (пара исключена/ещё не гонялась) — используется дефолт 0.50 с пометкой "degraded" в JSON.
  Бэнд-миксы (доли, myso1987 07-22) — см. BAND_MIX. weighted_wr(band) = Σ share×WR(archetype).

ACCEPTANCE (печать PASS/FAIL по каждому пункту + JSON):
  1. weighted_wr >= 0.52 для КАЖДОГО из 3 бэндов (band_700_799/800_899/900_1099);
  2. mirror: v063_h2h win_rate >= 0.55 И v070_h2h win_rate >= 0.55;
  3. 0 значимых регрессий vs v065 baseline (двухпропорционный z-тест, p<0.05, направление вниз);
  4. 0 исключений/крашей/нелегальных ходов/таймаутов суммарно по всем сыгранным играм
     (CORE + PROXY).

Резюмируемо: пары, уже присутствующие в выходном JSON с n_games >= целевого N, пропускаются
при повторном запуске.

Запуск:
    .venv/bin/python scripts/gate_v072_band.py --smoke --workers 4 --groups core   # sanity
    .venv/bin/python scripts/gate_v072_band.py --n 1000 --n-proxy 1000 --workers 8  # full run
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

# Нативный движок (macOS arm64). Threading НЕБЕЗОПАСЕН — только multiprocessing spawn.
from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
from cg_native.api import to_observation_class  # noqa: E402
import cg_native  # noqa: E402
import cg_native.api  # noqa: E402
import cg_native.game  # noqa: E402
import cg_native.sim  # noqa: E402
import cg_native.utils  # noqa: E402

# submissions/*/main.py делают `from cg.api import ...` — на macOS это Linux-only .so,
# алиасим на нативные модули (паттерн из scripts/h3_gate.py / gate_v065_regression.py).
sys.modules.setdefault("cg", cg_native)
sys.modules.setdefault("cg.api", cg_native.api)
sys.modules.setdefault("cg.game", cg_native.game)
sys.modules.setdefault("cg.sim", cg_native.sim)
sys.modules.setdefault("cg.utils", cg_native.utils)

MAX_MOVES_PER_GAME = 2000
GATES_DIR = os.path.join(ROOT, "results", "gates")

CANDIDATE_LABEL = "v073"          # переопределяется --label в main()


def resolve_dir(spec: str) -> str:
    """Абсолютный путь как есть; относительный — от ROOT."""
    if os.path.isabs(spec):
        return spec
    return os.path.join(ROOT, spec)


CANDIDATE_DIR = resolve_dir("submissions/v073_rockinn_damage")   # переопределяется --candidate
V072_DIR = resolve_dir("submissions/v072_mirror_gauntlet")       # действующий сабмит = h2h-база

# v070_grimmsnarl_hybrid не закоммичен в main (живёт на незамерженной ветке
# worktree-v070-grimmsnarl-hybrid, worktree заблокирован/locked, но читаем).
V070_DIR = resolve_dir(".claude/worktrees/v070-grimmsnarl-hybrid/submissions/v070_grimmsnarl_hybrid")

# tusk_crustle library-out (data/external_agents/, см. project_tusk_crustle... memory) —
# подтверждена рабочей парой ранее (scripts/gate_v070_vs_tusk.py, 500 игр 0 crashes).
TUSK_DIR = resolve_dir("data/external_agents/tusk_crustle_libout/tusk_crustle_submission")


def _pair_dir_valid(path: str) -> tuple[bool, str | None]:
    if not os.path.isdir(path):
        return False, f"dir not found: {path}"
    if not os.path.isfile(os.path.join(path, "main.py")):
        return False, f"main.py not found in {path}"
    if not os.path.isfile(os.path.join(path, "deck.csv")):
        return False, f"deck.csv not found in {path}"
    return True, None


# ---------------------------------------------------------------------------
# Мета-дек greedy-attack прокси (ПОРТ 1:1 из scripts/h3_gate.py / gate_v065_regression.py).
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
    mod_name = f"_gatev072_{os.path.basename(sub_dir)}_{_LOAD_COUNTER}"
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


# ---------------------------------------------------------------------------
# Вспомогательные функции статистики
# ---------------------------------------------------------------------------

def _is_legal(choice, sel: dict | None) -> bool:
    if sel is None:
        return isinstance(choice, list) and len(choice) == 60 and all(isinstance(c, int) for c in choice)
    opts = sel.get("option") or []
    n_opts = len(opts)
    # ВАЖНО: minCount=0 легален (пустой выбор) — `sel.get(...) or 1` идиома ломает это,
    # т.к. 0 falsy. Использовать `is None`.
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
        # [v073] БЫЛО: base_seed + hash((cell_id, game_idx)) % 1_000_000. Встроенный hash()
        # для СТРОК рандомизирован по процессу (PYTHONHASHSEED), поэтому набор сидов
        # менялся от прогона к прогону без всякой причины. crc32 детерминирован между
        # процессами и релизами — сид-набор теперь стабилен.
        #
        # ВАЖНО, ЧТО ЭТО НЕ ДАЁТ (проверено экспериментом 2026-07-24): парности прогонов.
        # `_play_one_game` сидит только `random.seed(seed)` питоновской стороны, а
        # `battle_start(deck0, deck1)` НЕ принимает сид вовсе — RNG движка (libcg /
        # cg_native) внутренний и извне не управляется. Два прогона ОДНОГО И ТОГО ЖЕ
        # кандидата с одним сидом дали 50/0/0 и 49/0/1 на паре v017 (N=50). Поэтому любое
        # сравнение версий здесь — это ДВЕ НЕЗАВИСИМЫЕ ВЫБОРКИ, а не парный дизайн:
        # единственный корректный вывод даёт двухпропорционный z-тест (compute_regressions),
        # а дельты ±1-2пп на отдельной паре при N=1000 — шум, даже если правка физически
        # не может касаться этого оппонента.
        seed = base_seed + (zlib.crc32(f"{cell_id}|{game_idx}".encode()) % 1_000_000)
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

_RAW_CORE_PAIRS = [
    dict(key="v017", opp_dir=resolve_dir("submissions/v017_alakazam_heur"), baseline="v017"),
    dict(key="v016", opp_dir=resolve_dir("submissions/v016_alakazam"), baseline="v016"),
    dict(key="v009a", opp_dir=resolve_dir("submissions/v009a_crustle"), baseline="v009a"),
    dict(key="v044", opp_dir=resolve_dir("submissions/v044_archaludon_abomasnow"), baseline="v044"),
    dict(key="v049", opp_dir=resolve_dir("submissions/v049_bench_sticky_guard"), baseline="v049"),
    # БАЗА для h2h-регрессий (см. модуль docstring) — сам v065 не встречается как ключ
    # в v065_core_gate.json (он был кандидатом, а не оппонентом там) -> baseline=None.
    dict(key="v065_h2h", opp_dir=resolve_dir("submissions/v065_defensive_swap"), baseline="v065_h2h"),
    # [v073] действующий сабмит — прямое зеркало, главная h2h-проверка "не хуже предшественника".
    # baseline=None: в v072_core_gate.json v072 фигурирует как кандидат, а не как оппонент,
    # так что исторического WR для этой пары нет — только отчёт + acceptance-порог ниже.
    dict(key="v072_h2h", opp_dir=V072_DIR, baseline=None),
    dict(key="v063_h2h", opp_dir=resolve_dir("submissions/v063_forensic86"), baseline="v063_h2h"),
    dict(key="v070_h2h", opp_dir=V070_DIR, baseline="v070_h2h"),
    dict(key="ext_prvsiyan_alak_v8", opp_dir=resolve_dir("submissions/external_prvsiyan_alakazam_v8"), baseline="ext_prvsiyan_alak_v8"),
    dict(key="ext_prvsiyan_alak_v9", opp_dir=resolve_dir("submissions/external_prvsiyan_alakazam_v9"), baseline="ext_prvsiyan_alak_v9"),
    dict(key="ext_maulikgajera_lucario", opp_dir=resolve_dir("submissions/external_maulikgajera_lucario"), baseline="ext_maulikgajera_lucario"),
    dict(key="ext_nursrijan_lucario", opp_dir=resolve_dir("submissions/external_nursrijan_lucario"), baseline="ext_nursrijan_lucario"),
    dict(key="v036_h2h", opp_dir=resolve_dir("submissions/v036_archaludon_patched"), baseline="v036_h2h"),
    dict(key="ext_tusk_crustle", opp_dir=TUSK_DIR, baseline="ext_tusk_crustle"),
]

# Валидация в момент импорта: пары с недоступным/неполным opp_dir исключаются (не падаем),
# причина фиксируется в EXCLUDED_PAIRS для докстринга/консоли/JSON.
CORE_PAIRS: list[dict] = []
EXCLUDED_PAIRS: list[dict] = []
for _cfg in _RAW_CORE_PAIRS:
    _ok, _reason = _pair_dir_valid(_cfg["opp_dir"])
    if _ok:
        CORE_PAIRS.append(_cfg)
    else:
        EXCLUDED_PAIRS.append(dict(key=_cfg["key"], opp_dir=_cfg["opp_dir"], reason=_reason))

PROXY_DECKS = ["Starmie_live"]

BASELINE_PATH = os.path.join(GATES_DIR, "v072_core_gate.json")   # переопределяется --baseline

# ---------------------------------------------------------------------------
# Бэнд-агрегация (архетип <- среднее WR по CORE-парам; см. докстринг модуля).
# ---------------------------------------------------------------------------

ARCHETYPE_MAP: dict[str, list[str]] = {
    "alakazam": ["v017", "v016", "ext_prvsiyan_alak_v8", "ext_prvsiyan_alak_v9"],
    "lucario": ["ext_maulikgajera_lucario", "ext_nursrijan_lucario"],
    "archaludon": ["v044", "v036_h2h"],
    "crustle": ["v009a", "ext_tusk_crustle"],
    # [v073] действующий сабмит входит в зеркальный архетип наравне с v063/v070 —
    # на ладдере 07-24 это ровно та дека, против которой мы играем 9% игр.
    "mirror": ["v063_h2h", "v070_h2h", "v072_h2h"],
    # "starmie" не входит сюда — берётся отдельно из PROXY-группы (meta:Starmie_live).
}

FIXED_ARCHETYPES = {"dragapult", "rest"}  # нет данных -> зафиксировано 0.50

# Доли бэнд-миксов (myso1987, снимок 2026-07-22 — см. memory reference_score_band_meta).
BAND_MIX: dict[str, dict[str, float]] = {
    "band_700_799": dict(lucario=.2625, archaludon=.1775, alakazam=.17, dragapult=.115, starmie=.0588, rest=.2162),
    "band_800_899": dict(archaludon=.2903, alakazam=.2716, lucario=.0719, crustle=.0479, mirror=.0519, rest=.3664),
    "band_900_1099": dict(alakazam=.38, mirror=.23, crustle=.09, archaludon=.09, lucario=.07, starmie=.026, rest=.114),
}

ACCEPTANCE_WR_THRESHOLD = 0.52
MIRROR_WR_THRESHOLD = 0.55


def load_baseline() -> dict:
    if not os.path.exists(BASELINE_PATH):
        print(f"[baseline] WARNING: {BASELINE_PATH} not found — regressions section will be empty", flush=True)
        return {}
    with open(BASELINE_PATH) as f:
        data = json.load(f)
    return data.get("opponents", {})


def compute_regressions(core_opponents: dict, baseline_opponents: dict) -> list[dict]:
    out = []
    for cfg in CORE_PAIRS:
        key = cfg["key"]
        base_key = cfg["baseline"]
        cand = core_opponents.get(key)
        if not cand or not cand.get("n_games"):
            continue
        if base_key is None or base_key not in baseline_opponents:
            out.append(dict(key=key, has_baseline=False))
            continue
        base = baseline_opponents[base_key]
        cand_we = cand["wins"] + 0.5 * cand["draws"]
        base_we = base["wins"] + 0.5 * base["draws"]
        z, p = two_prop_ztest(cand_we, cand["n_games"], base_we, base["n_games"])
        is_regression = (p < 0.05) and (cand["win_rate"] < base["win_rate"])
        out.append(dict(
            key=key, has_baseline=True,
            candidate_wr=cand["win_rate"], candidate_n=cand["n_games"],
            baseline_wr=base["win_rate"], baseline_n=base["n_games"],
            ztest_z=z, ztest_p_value=p, is_regression=is_regression,
        ))
    return out


def _archetype_wr(key: str, core_opponents: dict, proxy_data: dict) -> tuple[float | None, bool]:
    """Возвращает (wr, is_degraded_default)."""
    if key in FIXED_ARCHETYPES:
        return 0.50, False
    if key == "starmie":
        entry = proxy_data.get("Starmie_live", {}).get(CANDIDATE_LABEL)
        if entry and entry.get("n_games"):
            return entry["win_rate"], False
        return 0.50, True
    pair_keys = ARCHETYPE_MAP.get(key, [])
    vals = []
    for pk in pair_keys:
        o = core_opponents.get(pk)
        if o and o.get("n_games"):
            vals.append(o["win_rate"])
    if not vals:
        return 0.50, True
    return sum(vals) / len(vals), False


def compute_band_summary(core_opponents: dict, proxy_data: dict) -> dict:
    all_archetype_keys = set()
    for mix in BAND_MIX.values():
        all_archetype_keys.update(mix.keys())
    all_archetype_keys.add("starmie")

    archetype_wr = {}
    archetype_degraded = {}
    for k in sorted(all_archetype_keys):
        wr, degraded = _archetype_wr(k, core_opponents, proxy_data)
        archetype_wr[k] = wr
        archetype_degraded[k] = degraded

    bands = {}
    for band_key, mix in BAND_MIX.items():
        total = 0.0
        degraded_components = []
        for arche, share in mix.items():
            wr = archetype_wr.get(arche, 0.50)
            if archetype_degraded.get(arche):
                degraded_components.append(arche)
            total += share * wr
        bands[band_key] = dict(
            mix=mix,
            weighted_wr=total,
            degraded_components=degraded_components,
            passed=total >= ACCEPTANCE_WR_THRESHOLD,
        )

    return dict(archetype_wr=archetype_wr, archetype_degraded=archetype_degraded, bands=bands)


def compute_acceptance(core_opponents: dict, proxy_data: dict, band_summary: dict,
                        regressions: list[dict]) -> dict:
    band_pass = {bk: b["passed"] for bk, b in band_summary["bands"].items()}
    all_bands_pass = all(band_pass.values())

    v063 = core_opponents.get("v063_h2h", {})
    v070 = core_opponents.get("v070_h2h", {})
    v063_wr = v063.get("win_rate")
    v070_wr = v070.get("win_rate")
    mirror_pass = (
        v063_wr is not None and v063_wr >= MIRROR_WR_THRESHOLD
        and v070_wr is not None and v070_wr >= MIRROR_WR_THRESHOLD
    )

    # [v073] Прямое зеркало против ДЕЙСТВУЮЩЕГО сабмита. Порог здесь принципиально другой,
    # чем MIRROR_WR_THRESHOLD: P0 скоупнут на crustle, а v072 крустлом не играет, поэтому
    # ожидаемый результат этой пары — ровно паритет. Критерий = "не хуже": Wilson95_lo >= 0.45
    # (тот же порог, что в фул-прогоне v048/v065, memory project_full_gate_pool_config).
    v072 = core_opponents.get("v072_h2h", {})
    v072_wr = v072.get("win_rate")
    v072_lo = v072.get("wilson95_lo")
    incumbent_pass = v072_lo is not None and v072_lo >= 0.45

    sig_regressions = [r for r in regressions if r.get("is_regression")]
    no_regressions_pass = len(sig_regressions) == 0

    crash_total = 0
    for o in core_opponents.values():
        crash_total += (o.get("exc_agent", 0) + o.get("exc_opp", 0) + o.get("illegal_agent", 0)
                         + o.get("illegal_opp", 0) + o.get("timeouts", 0) + o.get("crashes", 0))
    for deck_data in proxy_data.values():
        for cand_label, o in deck_data.items():
            if not isinstance(o, dict) or "n_games" not in o:
                continue
            crash_total += (o.get("exc_agent", 0) + o.get("exc_opp", 0) + o.get("illegal_agent", 0)
                             + o.get("illegal_opp", 0) + o.get("timeouts", 0) + o.get("crashes", 0))
    no_crashes_pass = crash_total == 0

    overall = (all_bands_pass and mirror_pass and incumbent_pass
               and no_regressions_pass and no_crashes_pass)

    return dict(
        band_wr_pass=band_pass,
        band_wr_pass_all=all_bands_pass,
        mirror_pass=mirror_pass,
        mirror_v063_h2h_wr=v063_wr,
        mirror_v070_h2h_wr=v070_wr,
        incumbent_pass=incumbent_pass,
        incumbent_v072_h2h_wr=v072_wr,
        incumbent_v072_h2h_lo=v072_lo,
        no_regressions_pass=no_regressions_pass,
        n_significant_regressions=len(sig_regressions),
        regression_keys=[r["key"] for r in sig_regressions],
        no_crashes_pass=no_crashes_pass,
        crash_total=crash_total,
        overall_pass=overall,
    )


def print_band_table(band_summary: dict, acceptance: dict, regressions: list[dict]) -> None:
    print("\n=== ARCHETYPE WIN RATES ===")
    for k in sorted(band_summary["archetype_wr"].keys()):
        wr = band_summary["archetype_wr"][k]
        tag = " (degraded default 0.50, no data)" if band_summary["archetype_degraded"].get(k) else ""
        print(f"  {k:12s} WR={wr:.4f}{tag}")

    print("\n=== BAND-WEIGHTED WIN RATE (acceptance >= {:.2f}) ===".format(ACCEPTANCE_WR_THRESHOLD))
    for bk, b in band_summary["bands"].items():
        status = "PASS" if b["passed"] else "FAIL"
        deg = f"  degraded={b['degraded_components']}" if b["degraded_components"] else ""
        print(f"  {bk:16s} weighted_wr={b['weighted_wr']:.4f}  [{status}]{deg}")

    print(f"\n=== REGRESSIONS vs baseline ({BASELINE_PATH}) ===")
    if not regressions:
        print("  (no data yet)")
    for r in regressions:
        if not r.get("has_baseline"):
            print(f"  {r['key']:24s} no baseline — report only")
            continue
        flag = "REGRESSION" if r["is_regression"] else "ok"
        print(f"  {r['key']:24s} cand={r['candidate_wr']:.4f} base={r['baseline_wr']:.4f} "
              f"z={r['ztest_z']:.2f} p={r['ztest_p_value']:.4f}  [{flag}]")

    print("\n=== ACCEPTANCE ===")
    print(f"  1. band_wr>=0.52 for each band: {'PASS' if acceptance['band_wr_pass_all'] else 'FAIL'} "
          f"({acceptance['band_wr_pass']})")
    print(f"  2. mirror v063_h2h>={MIRROR_WR_THRESHOLD} ({acceptance['mirror_v063_h2h_wr']}) AND "
          f"v070_h2h>={MIRROR_WR_THRESHOLD} ({acceptance['mirror_v070_h2h_wr']}): "
          f"{'PASS' if acceptance['mirror_pass'] else 'FAIL'}")
    print(f"  2b. incumbent v072_h2h wilson95_lo>=0.45 "
          f"(wr={acceptance['incumbent_v072_h2h_wr']}, lo={acceptance['incumbent_v072_h2h_lo']}): "
          f"{'PASS' if acceptance['incumbent_pass'] else 'FAIL'}")
    print(f"  3. 0 significant regressions vs baseline: "
          f"{'PASS' if acceptance['no_regressions_pass'] else 'FAIL'} "
          f"(n={acceptance['n_significant_regressions']}, keys={acceptance['regression_keys']})")
    print(f"  4. 0 exceptions/illegal/timeouts/crashes: "
          f"{'PASS' if acceptance['no_crashes_pass'] else 'FAIL'} (total={acceptance['crash_total']})")
    print(f"\n  OVERALL: {'PASS' if acceptance['overall_pass'] else 'FAIL'}")

    if EXCLUDED_PAIRS:
        print("\n=== EXCLUDED PAIRS (invalid opp_dir, skipped — not a crash) ===")
        for e in EXCLUDED_PAIRS:
            print(f"  {e['key']}: {e['reason']}")


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
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--groups", default="core,proxy", help="core,proxy")
    ap.add_argument("--only", default=None, help="Запустить только эту cell_id (debug)")
    ap.add_argument("--smoke", action="store_true", help="4 игры/пара, sanity-check")
    # [v073] кандидат параметризован — один и тот же гейт гоняет обе сборки
    # (v073_rockinn_damage = P0, v073b_park = P0+P1) в отдельные JSON.
    ap.add_argument("--candidate", default="submissions/v073_rockinn_damage",
                    help="Каталог сабмита-кандидата (относительный путь — от ROOT)")
    ap.add_argument("--label", default="v073", help="Метка кандидата (имена JSON и cell_id)")
    ap.add_argument("--baseline", default=None,
                    help="JSON-гейт для регрессионного сравнения (по умолчанию v072_core_gate.json)")
    ap.add_argument("--out-core", default=None)
    ap.add_argument("--out-proxy", default=None)
    ap.add_argument("--out-band", default=None)
    args = ap.parse_args()

    global CANDIDATE_DIR, CANDIDATE_LABEL, BASELINE_PATH
    CANDIDATE_DIR = resolve_dir(args.candidate)
    CANDIDATE_LABEL = args.label
    if args.baseline:
        BASELINE_PATH = resolve_dir(args.baseline)
    if not os.path.isdir(CANDIDATE_DIR):
        print(f"[init] FATAL: candidate dir not found: {CANDIDATE_DIR}", flush=True)
        return 2
    if args.out_core is None:
        args.out_core = os.path.join(GATES_DIR, f"{CANDIDATE_LABEL}_core_gate.json")
    if args.out_proxy is None:
        args.out_proxy = os.path.join(GATES_DIR, f"{CANDIDATE_LABEL}_proxy_gate.json")
    if args.out_band is None:
        args.out_band = os.path.join(GATES_DIR, f"{CANDIDATE_LABEL}_band_summary.json")
    print(f"[init] candidate={CANDIDATE_DIR} label={CANDIDATE_LABEL} baseline={BASELINE_PATH}",
          flush=True)

    n_core = 4 if args.smoke else args.n
    n_proxy = 4 if args.smoke else args.n_proxy
    groups = set(args.groups.split(","))

    if EXCLUDED_PAIRS:
        print("[init] excluded pairs (invalid opp_dir, skipped):", flush=True)
        for e in EXCLUDED_PAIRS:
            print(f"  {e['key']}: {e['reason']}", flush=True)

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
            stats = run_pair(CANDIDATE_DIR, "sub:" + cfg["opp_dir"], key, n_core, args.seed, args.workers)
            print(f"[core] {key}: done in {time.time()-t0:.1f}s  "
                  f"WR={stats['win_rate']:.4f} [{stats['wilson95_lo']:.3f},{stats['wilson95_hi']:.3f}]  "
                  f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']}  "
                  f"exc={stats['exc_agent']}/{stats['exc_opp']} illegal={stats['illegal_agent']}/{stats['illegal_opp']} "
                  f"timeouts={stats['timeouts']} crashes={stats['crashes']}", flush=True)
            opponents[key] = stats
            existing = dict(
                tag=f"{CANDIDATE_LABEL}_core_gate",
                agent=dict(ckpt=None, agent_dir=os.path.basename(CANDIDATE_DIR),
                            deck=os.path.join(CANDIDATE_DIR, "deck.csv"),
                            label="submission:" + os.path.basename(CANDIDATE_DIR)),
                agent_ckpt_exceptions=None,
                excluded_pairs=EXCLUDED_PAIRS,
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
            cell_id = f"{CANDIDATE_LABEL}_vs_{deck_name}"
            if args.only and cell_id != args.only:
                continue
            have = proxies[deck_name].get(CANDIDATE_LABEL, {})
            if have.get("n_games", 0) >= n_proxy:
                print(f"[proxy] {cell_id}: already have n_games={have['n_games']} >= {n_proxy}, skip")
                continue
            print(f"[proxy] {cell_id}: running n={n_proxy} ...", flush=True)
            t0 = time.time()
            stats = run_pair(CANDIDATE_DIR, "meta:" + deck_name, cell_id, n_proxy, args.seed, args.workers)
            print(f"[proxy] {cell_id}: done in {time.time()-t0:.1f}s  "
                  f"WR={stats['win_rate']:.4f} [{stats['wilson95_lo']:.3f},{stats['wilson95_hi']:.3f}]  "
                  f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']}  "
                  f"exc={stats['exc_agent']}/{stats['exc_opp']} illegal={stats['illegal_agent']}/{stats['illegal_opp']} "
                  f"timeouts={stats['timeouts']} crashes={stats['crashes']}", flush=True)
            proxies[deck_name][CANDIDATE_LABEL] = stats
            existing = dict(
                tag=f"{CANDIDATE_LABEL}_proxy_gate",
                candidates={CANDIDATE_LABEL: CANDIDATE_DIR},
                proxy_decks=PROXY_DECKS,
                proxies=proxies,
                generated=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            save_json(out_path, existing)

    # ------------------------------------------------------------------
    # Бэнд-сводка: пересчитывается из того, что реально накоплено на диске
    # (даже если этот запуск затронул только часть пар/групп через --only или --groups).
    # ------------------------------------------------------------------
    core_existing = load_existing(args.out_core)
    core_opponents = core_existing.get("opponents", {})
    proxy_existing = load_existing(args.out_proxy)
    proxy_data = proxy_existing.get("proxies", {})

    baseline_opponents = load_baseline()
    regressions = compute_regressions(core_opponents, baseline_opponents)
    band_summary = compute_band_summary(core_opponents, proxy_data)
    acceptance = compute_acceptance(core_opponents, proxy_data, band_summary, regressions)

    print_band_table(band_summary, acceptance, regressions)

    band_out = dict(
        tag=f"{CANDIDATE_LABEL}_band_summary",
        candidate=os.path.basename(CANDIDATE_DIR),
        baseline_source=BASELINE_PATH if os.path.exists(BASELINE_PATH) else None,
        excluded_pairs=EXCLUDED_PAIRS,
        archetype_wr=band_summary["archetype_wr"],
        archetype_degraded=band_summary["archetype_degraded"],
        bands=band_summary["bands"],
        regressions=regressions,
        acceptance=acceptance,
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    save_json(args.out_band, band_out)

    print("\n=== DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
