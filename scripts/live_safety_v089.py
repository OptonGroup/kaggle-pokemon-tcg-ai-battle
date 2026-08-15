#!/usr/bin/env python3
"""live_safety_v089.py — предсабмитная проверка артефакта на РЕАЛЬНЫХ играх движка.

Проверяет ровно то, что ломается в проде и НЕ ловится офлайн-метриками
(см. память проекта: project_kaggle_live_safety, project_engine_search):
  1. агент не кидает исключений ни на одном решении (в live исключение = молчаливый
     greedy-fallback или поражение);
  2. КАЖДЫЙ возвращённый список индексов легален: все в [0, len(option)), длина в
     [minCount, maxCount], без дубликатов (невалидный индекс = мгновенное поражение);
  3. время на решение и суммарный банк на эпизод (live: 600с, 2 vCPU) — с запасом;
  4. первый вызов (obs.select is None) возвращает ровно 60 id деки.

Каждый эпизод грузит main.py ЗАНОВО (в проде так и происходит), чтобы поймать
межигровое протекание состояния — известная ловушка кэша sys.modules
(project_gate_loader_module_cache_trap): main пересоздаётся на игру, его импорты НЕТ.

Пример:
  python scripts/live_safety_v089.py --agent-dir submissions/v089_bc2b \
      --opp-dir submissions/v088_bc2b --games 6
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import cg_native  # noqa: E402
import cg_native.api  # noqa: E402

sys.modules.setdefault("cg", cg_native)
sys.modules.setdefault("cg.api", cg_native.api)

from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402


class IllegalMove(Exception):
    pass


def load_agent(sub_dir: Path, tag: str):
    """Загружает main.py сабмита как свежий модуль (уникальное имя => свой sys.modules-слот)."""
    main_py = sub_dir / "main.py"
    sys.path.insert(0, str(sub_dir))
    try:
        name = f"_live_{tag}_{time.perf_counter_ns()}"
        spec = importlib.util.spec_from_file_location(name, str(main_py))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod.agent
    finally:
        sys.path.remove(str(sub_dir))


def check_legal(select, obs_dict, where: str) -> None:
    sel = (obs_dict.get("select") or {})
    options = sel.get("option") or []
    lo = sel.get("minCount", 1)
    hi = sel.get("maxCount", 1)
    if not isinstance(select, list):
        raise IllegalMove(f"{where}: вернул {type(select).__name__}, ожидался list")
    if len(select) != len(set(select)):
        raise IllegalMove(f"{where}: дубликаты в {select}")
    if not (lo <= len(select) <= hi):
        raise IllegalMove(f"{where}: длина {len(select)} вне [{lo},{hi}] (опций {len(options)})")
    for i in select:
        if not isinstance(i, int) or not (0 <= i < len(options)):
            raise IllegalMove(f"{where}: индекс {i} вне [0,{len(options)})")


def run_game(agent0, agent1, deck0, deck1, bank_s=600.0) -> dict:
    obs_dict, start = battle_start(deck0, deck1)
    if obs_dict is None:
        raise RuntimeError(f"battle_start отказал: {start.errorPlayer}/{start.errorType}")
    bank = {0: bank_s, 1: bank_s}
    times = {0: [], 1: []}
    moves = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                return {"winner": obs.current.result, "moves": moves,
                        "max_move_s": {k: (max(v) if v else 0.0) for k, v in times.items()},
                        "spent_s": {k: round(bank_s - bank[k], 1) for k in bank}}
            me = obs.current.yourIndex if obs.current else 0
            agent = agent0 if me == 0 else agent1
            obs_dict["remainingOverageTime"] = bank[me]
            t0 = time.perf_counter()
            select = agent(obs_dict)
            dt = time.perf_counter() - t0
            times[me].append(dt)
            bank[me] -= dt
            check_legal(select, obs_dict, f"игрок{me} ход{moves}")
            if bank[me] <= 0:
                raise IllegalMove(f"игрок{me} исчерпал банк 600с")
            obs_dict = battle_select(select)
            moves += 1
    finally:
        battle_finish()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-dir", required=True)
    ap.add_argument("--opp-dir", required=True)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    a_dir, o_dir = REPO / args.agent_dir, REPO / args.opp_dir
    deck_a = [int(x) for x in (a_dir / "deck.csv").read_text().split() if x.strip()]
    deck_o = [int(x) for x in (o_dir / "deck.csv").read_text().split() if x.strip()]
    print(f"[live] дека кандидата: {len(deck_a)} карт; оппонента: {len(deck_o)}")
    if len(deck_a) != 60:
        print(f"[live] ПРОВАЛ: deck.csv кандидата содержит {len(deck_a)} карт, нужно 60")
        return 1

    results, all_max, exceptions = [], [], []
    for g in range(args.games):
        # свежая загрузка обоих агентов на КАЖДУЮ игру — как в проде
        ag = load_agent(a_dir, "cand")
        op = load_agent(o_dir, "base")
        first, second = (ag, op) if g % 2 == 0 else (op, ag)
        cand_idx = 0 if g % 2 == 0 else 1
        try:
            r = run_game(first, second, deck_a if g % 2 == 0 else deck_o,
                         deck_o if g % 2 == 0 else deck_a)
        except Exception as e:  # noqa: BLE001 — намеренно широко: любое исключение = провал
            exceptions.append(f"игра{g}: {type(e).__name__}: {e}")
            print(f"[live] игра {g+1}/{args.games}: ИСКЛЮЧЕНИЕ {type(e).__name__}: {e}")
            continue
        won = (r["winner"] == cand_idx)
        results.append(won)
        all_max.append(r["max_move_s"][cand_idx])
        print(f"[live] игра {g+1}/{args.games}: {'победа' if won else 'поражение'} "
              f"ходов={r['moves']} макс.решение={r['max_move_s'][cand_idx]:.2f}с "
              f"банк потрачен={r['spent_s'][cand_idx]}с/600")

    n = len(results)
    rep = {"games_ok": n, "games_requested": args.games, "exceptions": exceptions,
           "wins": sum(results), "max_move_s": round(max(all_max), 2) if all_max else None,
           "median_max_move_s": round(statistics.median(all_max), 2) if all_max else None}
    print(f"\n[live] сыграно без ошибок: {n}/{args.games}; побед кандидата: {sum(results)}")
    if all_max:
        print(f"[live] худшее решение: {max(all_max):.2f}с (лимит эпизода 600с суммарно)")
    ok = (n == args.games) and not exceptions
    print("ИТОГ:", "OK — исключений нет, все ходы легальны, время с запасом" if ok
          else "ПРОВАЛ — см. исключения выше, САБМИТИТЬ НЕЛЬЗЯ")
    rep["verdict"] = "OK" if ok else "FAIL"
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
