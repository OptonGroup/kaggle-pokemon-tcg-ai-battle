"""field_viability.py — дек-viability против расширенного мета-поля (post-PIMC-KILL).

Вопрос: какая НАША дека+пилот выживает в текущей мете (07-12)?
PIMC закрыт (§H3-4-re KILL). Вектор — deck-swap/политика. Прежде чем вкладывать
в улучшение политики, честный скрин: наш чемпион-пилот на нашей деке против
16 top-cut дек (yu0307/deck-csv, recon 07-11) + 3 старые мета-деки.

МЕТОД: каждый наш агент-кандидат (v009a Crustle-IL, v018 Grimmsnarl-IL,
v017 Alakazam) vs каждая мета-дека (generic greedy-пилот из h3_gate).
Seat-swap (чередование сторон). N игр/матчап.

ВАЖНАЯ ОГОВОРКА: generic greedy-пилот СЛАБЕЕ реального IL-пилота мета-деки →
наши WR ОПТИМИСТИЧНО ЗАВЫШЕНЫ. Но как ОТНОСИТЕЛЬНЫЙ скрин (какая наша дека
лучше держит поле, против каких архетипов проваливается) — валидно: слабый
пилот одинаково слаб против всех наших дек. Live остаётся арбитром.

Переиспользует инфраструктуру scripts/h3_gate.py: greedy_meta_agent, loaders,
game loop, cg-алиас, multiprocessing spawn.

Запуск (нативный движок macOS arm64):
    .venv/bin/python scripts/field_viability.py --games 40 --workers 4
    .venv/bin/python scripts/field_viability.py --games 1 --candidates v009a_crustle --decks MegaStarmie  # sanity
    .venv/bin/python scripts/field_viability.py --report-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Переиспользуем проверенные компоненты h3_gate (loaders, greedy-пилот, cg-алиас).
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
from cg_native.api import to_observation_class  # noqa: E402

import h3_gate  # noqa: E402  — greedy_meta_agent, _load_*, _is_legal, wilson_ci, cg-алиас
from h3_gate import (  # noqa: E402
    greedy_meta_agent,
    _load_submission_module,
    _load_deck_csv,
    _resolve_opp_deck,
    _is_legal,
    _median,
    wilson_ci,
)

MAX_MOVES_PER_GAME = 2000

RESULTS_DIR = os.path.join(ROOT, "results", "gates")
STATE_PATH = os.path.join(RESULTS_DIR, "field_viability_state.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "field_viability_report.json")
META_DIR = os.path.join(ROOT, "decks", "meta")

# Наши деки+пилоты с РЕАЛЬНЫМ пилотом (не random). sub_dir в submissions/.
DEFAULT_CANDIDATES = [
    "v009a_crustle",       # Crustle-IL — действующий чемпион (~733 live, был 930)
    "v018_grimmsnarl",     # Grimmsnarl-IL — BC-friendly midrange
    "v017_alakazam_heur",  # Alakazam — комбо (recon: дека >1100, наш BC был слаб)
]


def _list_meta_decks() -> list[str]:
    """Все decks/meta/*.csv → имена без расширения."""
    out = []
    for fn in sorted(os.listdir(META_DIR)):
        if fn.endswith(".csv"):
            out.append(fn[:-4])
    return out


def _load_candidate(sub_dir: str):
    """Возвращает (agent_fn, deck) для нашего кандидата."""
    mod = _load_submission_module(sub_dir)
    agent_fn = mod.agent
    deck = list(getattr(mod, "my_deck", None) or _resolve_opp_deck(sub_dir))
    return agent_fn, deck


def _load_meta_deck_by_name(name: str) -> list[int]:
    return _load_deck_csv(os.path.join(META_DIR, f"{name}.csv"))


# ---------------------------------------------------------------------------
# Одна игра: наш кандидат vs greedy-пилот на мета-деке (seat-swap)
# ---------------------------------------------------------------------------

def play_one(task: dict) -> dict:
    cand = task["candidate"]
    deck_name = task["deck"]
    our_first = task["our_first"]

    our_agent, our_deck = _load_candidate(cand)
    opp_deck = _load_meta_deck_by_name(deck_name)
    opp_agent = greedy_meta_agent

    if our_first:
        deck0, deck1 = our_deck, opp_deck
        agent0, agent1 = our_agent, opp_agent
    else:
        deck0, deck1 = opp_deck, our_deck
        agent0, agent1 = opp_agent, our_agent

    obs_dict, start = battle_start(list(deck0), list(deck1))
    if obs_dict is None:
        return dict(candidate=cand, deck=deck_name, our_first=our_first,
                    outcome="crash", error=f"battle_start:{getattr(start,'errorType',None)}",
                    n_illegal=0, n_exceptions=1, moves=0)

    move_count = 0
    n_illegal = n_exceptions = 0
    winner = None
    our_times: list[float] = []
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
            except Exception:
                n_exceptions += 1
                exc_side = "our" if is_our_turn else "opp"
                break
            if is_our_turn:
                our_times.append((time.perf_counter() - t0) * 1000.0)
            if not _is_legal(choice, sel):
                n_illegal += 1
                break
            obs_dict = battle_select(choice)
            move_count += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass

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

    return dict(candidate=cand, deck=deck_name, our_first=our_first,
                outcome=outcome, winner=winner, moves=move_count,
                n_illegal=n_illegal, n_exceptions=n_exceptions, exc_side=exc_side,
                ms_median=_median(our_times), ms_max=(max(our_times) if our_times else 0.0))


# ---------------------------------------------------------------------------
# State (resumable) + queue
# ---------------------------------------------------------------------------

def cell_key(cand: str, deck: str) -> str:
    return f"{cand}__vs__{deck}"


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"cells": {}}


def save_state(state: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


def build_queue(state: dict, candidates: list[str], decks: list[str], n_games: int) -> list[dict]:
    """Round-robin: interleave по (cand, deck), чтобы промежуточные срезы были равномерны."""
    cells = state.get("cells", {})
    tasks: list[dict] = []
    # По game_idx во внешнем цикле → все ячейки растут параллельно.
    for gi in range(n_games):
        for cand in candidates:
            for deck in decks:
                ck = cell_key(cand, deck)
                have = len(cells.get(ck, {}).get("games", []))
                if have > gi:
                    continue  # эта партия уже сыграна (resume)
                tasks.append(dict(candidate=cand, deck=deck, game_idx=gi,
                                  our_first=(gi % 2 == 0)))
    return tasks


def record(state: dict, res: dict) -> None:
    ck = cell_key(res["candidate"], res["deck"])
    cells = state.setdefault("cells", {})
    if ck not in cells:
        cells[ck] = {"candidate": res["candidate"], "deck": res["deck"], "games": []}
    cells[ck]["games"].append(res)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(state: dict, candidates: list[str], decks: list[str]) -> dict:
    cells = state.get("cells", {})
    total = sum(len(c["games"]) for c in cells.values())
    ill = sum(g.get("n_illegal", 0) for c in cells.values() for g in c["games"])
    exc = sum(g.get("n_exceptions", 0) for c in cells.values() for g in c["games"])

    per_candidate: dict = {}
    matrix: dict = {}
    for cand in candidates:
        rows = {}
        wins = losses = draws = other = n = 0
        for deck in decks:
            games = cells.get(cell_key(cand, deck), {}).get("games", [])
            w = sum(1 for g in games if g["outcome"] == "our_win")
            l = sum(1 for g in games if g["outcome"] == "opp_win")
            d = sum(1 for g in games if g["outcome"] == "draw")
            o = sum(1 for g in games if g["outcome"] in ("crash", "illegal", "exception", "aborted"))
            gn = len(games)
            decisive = w + l
            wr = (w / decisive) if decisive else float("nan")
            lo, hi = wilson_ci(w, decisive) if decisive else (float("nan"), float("nan"))
            rows[deck] = dict(n=gn, w=w, l=l, d=d, bad=o, wr=wr, wilson_lo=lo, wilson_hi=hi)
            wins += w; losses += l; draws += d; other += o; n += gn
        dec = wins + losses
        overall_wr = (wins / dec) if dec else float("nan")
        lo, hi = wilson_ci(wins, dec) if dec else (float("nan"), float("nan"))
        # худшие матчапы (по WR, где есть данные)
        ranked = sorted(
            [(dk, r) for dk, r in rows.items() if (r["w"] + r["l"]) >= 4],
            key=lambda kv: kv[1]["wr"],
        )
        worst = [(dk, round(r["wr"], 3), f"{r['w']}-{r['l']}") for dk, r in ranked[:5]]
        best = [(dk, round(r["wr"], 3), f"{r['w']}-{r['l']}") for dk, r in ranked[-5:][::-1]]
        per_candidate[cand] = dict(
            n_games=n, wins=wins, losses=losses, draws=draws, bad=other,
            overall_wr=overall_wr, wilson_lo=lo, wilson_hi=hi,
            worst_matchups=worst, best_matchups=best,
        )
        matrix[cand] = rows

    return dict(total_games=total, total_illegal=ill, total_exceptions=exc,
                candidates=candidates, decks=decks,
                per_candidate=per_candidate, matrix=matrix)


def print_report(rep: dict) -> None:
    print("\n" + "=" * 80)
    print("FIELD VIABILITY — наш пилот+дека vs 19 top-cut мета-дек (generic greedy-пилот)")
    print("ОГОВОРКА: greedy-пилот слабее реального IL → WR завышены; относительный скрин.")
    print("=" * 80)
    print(f"total games={rep['total_games']}  illegal={rep['total_illegal']}  exceptions={rep['total_exceptions']}")
    for cand, c in rep["per_candidate"].items():
        wr = c["overall_wr"]
        print(f"\n### {cand}")
        print(f"  overall: {c['wins']}-{c['losses']} (draws={c['draws']}, bad={c['bad']})  "
              f"WR={wr:.3f} Wilson[{c['wilson_lo']:.2f},{c['wilson_hi']:.2f}]  n={c['n_games']}")
        print(f"  ХУДШИЕ матчапы: {c['worst_matchups']}")
        print(f"  ЛУЧШИЕ матчапы: {c['best_matchups']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40, help="партий на матчап (seat-swap)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATES),
                    help="через запятую sub_dir наших агентов")
    ap.add_argument("--decks", type=str, default="", help="через запятую имена мета-дек (пусто=все)")
    ap.add_argument("--seed", type=int, default=20260712)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    decks = [d.strip() for d in args.decks.split(",") if d.strip()] or _list_meta_decks()

    state = load_state()

    if args.report_only:
        rep = build_report(state, candidates, decks)
        print_report(rep)
        return 0

    queue = build_queue(state, candidates, decks, args.games)
    print(f"Кандидаты: {candidates}")
    print(f"Мета-дек: {len(decks)}  ({', '.join(decks)})")
    print(f"Партий/матчап: {args.games}  →  осталось сыграть: {len(queue)} игр "
          f"(всего целевых {len(candidates)*len(decks)*args.games})")
    if not queue:
        print("Всё уже сыграно. --report-only для отчёта.")
        rep = build_report(state, candidates, decks)
        print_report(rep)
        return 0

    ctx = __import__("multiprocessing").get_context("spawn")
    t_start = time.time()
    n_done = 0
    save_every = max(4, len(queue) // 50)

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = {}
        it = iter(queue)
        # заполняем начальное окно
        for _ in range(min(args.workers * 2, len(queue))):
            try:
                task = next(it)
            except StopIteration:
                break
            futures[ex.submit(play_one, task)] = task

        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                task = futures.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:
                    res = dict(candidate=task["candidate"], deck=task["deck"],
                               our_first=task["our_first"], outcome="crash",
                               error=repr(e), n_illegal=0, n_exceptions=1, moves=0)
                record(state, res)
                n_done += 1
                if n_done % save_every == 0:
                    save_state(state)
                    el = time.time() - t_start
                    rate = n_done / el if el else 0
                    eta = (len(queue) - n_done) / rate if rate else 0
                    print(f"  [{n_done}/{len(queue)}] {rate:.1f} игр/с  ETA {eta/60:.0f}мин  "
                          f"last: {res['candidate'][:12]} vs {res['deck'][:14]} → {res['outcome']}", flush=True)
                # доливаем
                try:
                    nt = next(it)
                    futures[ex.submit(play_one, nt)] = nt
                except StopIteration:
                    pass

    save_state(state)
    rep = build_report(state, candidates, decks)
    with open(REPORT_PATH, "w") as f:
        json.dump(rep, f, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print_report(rep)
    print(f"\nState: {STATE_PATH}\nReport: {REPORT_PATH}")
    print(f"Wall: {(time.time()-t_start)/60:.1f} мин")
    return 0


if __name__ == "__main__":
    sys.exit(main())
