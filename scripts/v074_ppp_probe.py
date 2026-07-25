"""v074_ppp_probe.py — предусловие P0 из обновлённого ТЗ (docs/gold/PLAN_v074.md, врезка 07-25).

Вопрос, без ответа на который гейт по lucario-матчапу слеп (§9.1 отчёта
out/v073_full_download/ANALYSIS_v073.md):

  1. Разыгрывают ли lucario-оппоненты ГЕЙТ-ПУЛА карту Premium Power Pro (Item 1141,
     «+30 урона от {F}-покемонов в актив на этот ход») — и как часто? В live она была
     сыграна в 25 из 26 партий.
  2. Стакается ли она у них в пределах одного хода (в live наблюдались +60 и +90)?
  3. Видна ли она в `PlayerState.discard` соперника ДО первого крупного удара — то есть
     работает ли наблюдение как предиктор (в live: 12 игр из 26).
  4. Какое распределение ФАКТИЧЕСКОГО входящего урона по нашему активному телу, и
     соответствует ли надбавка +30 появлению копии в сбросе.

Метод: играем кандидата против оппонента, на каждом шаге снимаем (a) счётчик копий 1141
в сбросе соперника, (b) нашу активную (id, hp). Падение hp нашей активной при неизменном
id = входящий урон; сопоставляем его с приростом счётчика 1141 в том же окне.

Запуск:
    .venv/bin/python scripts/v074_ppp_probe.py --opp submissions/external_nursrijan_lucario \
        --candidate submissions/v074_fullmetal --games 30
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter

# ROOT можно переопределить через PTCG_ROOT: скрипт живёт в worktree, а пул оппонентов и
# libcg.dylib — в основном чекауте.
ROOT = os.environ.get("PTCG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
from cg_native.api import to_observation_class, all_card_data  # noqa: E402
import cg_native  # noqa: E402
import cg_native.api  # noqa: E402
import cg_native.game  # noqa: E402
import cg_native.sim  # noqa: E402
import cg_native.utils  # noqa: E402

sys.modules.setdefault("cg", cg_native)
sys.modules.setdefault("cg.api", cg_native.api)
sys.modules.setdefault("cg.game", cg_native.game)
sys.modules.setdefault("cg.sim", cg_native.sim)
sys.modules.setdefault("cg.utils", cg_native.utils)

PREMIUM_POWER_PRO = 1141
GRAVITY_MOUNTAIN = 1252
LUCARIO_LINE = {677, 678}
MAX_MOVES = 2000

CARD_DB = {c.cardId: c for c in all_card_data()}
_LOAD_COUNTER = 0


def _load_submission_module(sub_dir: str):
    global _LOAD_COUNTER
    _LOAD_COUNTER += 1
    main_path = os.path.join(sub_dir, "main.py")
    assert os.path.exists(main_path), f"не найден {main_path}"
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    mod_name = f"_pppprobe_{os.path.basename(sub_dir)}_{_LOAD_COUNTER}"
    cwd = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(mod_name, main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


def _load_deck(sub_dir: str, mod) -> list[int]:
    deck = getattr(mod, "my_deck", None)
    if deck:
        return list(deck)
    with open(os.path.join(sub_dir, "deck.csv")) as f:
        return [int(x.strip()) for x in f if x.strip()]


def _stadium_id(cur: dict):
    for c in (cur.get("stadium") or []):
        if isinstance(c, dict) and c.get("id") is not None:
            return int(c["id"])
    return None


def _discard_count(pl: dict, card_id: int) -> int:
    return sum(1 for c in (pl.get("discard") or [])
               if isinstance(c, dict) and int(c.get("id") or 0) == card_id)


def _active(pl: dict):
    act = (pl.get("active") or [None])[0]
    if not isinstance(act, dict):
        return None
    return (int(act.get("id") or 0), int(act.get("hp") or 0))


def run_game(cand_dir: str, opp_dir: str, our_first: bool, seed: int,
             hits: list, games: list, stats: Counter):
    random.seed(seed)
    cand_mod = _load_submission_module(cand_dir)
    opp_mod = _load_submission_module(opp_dir)
    cand_deck = _load_deck(cand_dir, cand_mod)
    opp_deck = _load_deck(opp_dir, opp_mod)

    if our_first:
        deck0, deck1 = list(cand_deck), list(opp_deck)
        agent0, agent1 = cand_mod.agent, opp_mod.agent
    else:
        deck0, deck1 = list(opp_deck), list(cand_deck)
        agent0, agent1 = opp_mod.agent, cand_mod.agent
    our_side = 0 if our_first else 1

    obs_dict, _ = battle_start(deck0, deck1)
    if obs_dict is None:
        return

    g = dict(seed=seed, ppp_max=0, ppp_max_step_delta=0, first_ppp_move=None,
             first_big_hit_move=None, first_big_hit_dmg=0, gravity_seen=False,
             our_steps=0, big_hits=0)
    prev_ppp = 0
    prev_active = None
    moves = 0
    try:
        while moves < MAX_MOVES:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                break
            cur = obs_dict.get("current") or {}
            yi = int(cur.get("yourIndex") or 0)
            players = cur.get("players") or [{}, {}]
            while len(players) < 2:
                players = players + [{}]
            opp_pl = players[1 - our_side]
            my_pl = players[our_side]

            ppp = _discard_count(opp_pl, PREMIUM_POWER_PRO)
            if ppp > g["ppp_max"]:
                g["ppp_max"] = ppp
            delta = ppp - prev_ppp
            if delta > g["ppp_max_step_delta"]:
                g["ppp_max_step_delta"] = delta
            if ppp > 0 and g["first_ppp_move"] is None:
                g["first_ppp_move"] = moves

            if _stadium_id(cur) == GRAVITY_MOUNTAIN:
                g["gravity_seen"] = True

            my_act = _active(my_pl)
            opp_act = _active(opp_pl)
            if (prev_active is not None and my_act is not None
                    and prev_active[0] == my_act[0] and my_act[1] < prev_active[1]):
                dmg = prev_active[1] - my_act[1]
                hits.append(dict(
                    seed=seed, move=moves, dmg=dmg,
                    target_id=prev_active[0], hp_before=prev_active[1],
                    opp_active_id=(opp_act[0] if opp_act else 0),
                    ppp_before=prev_ppp, ppp_now=ppp, ppp_delta=delta,
                    gravity=(_stadium_id(cur) == GRAVITY_MOUNTAIN),
                ))
                if dmg >= 200:
                    g["big_hits"] += 1
                    if g["first_big_hit_move"] is None:
                        g["first_big_hit_move"] = moves
                        g["first_big_hit_dmg"] = dmg
            prev_active = my_act
            prev_ppp = ppp

            if yi == our_side:
                g["our_steps"] += 1

            agent_fn = agent0 if yi == 0 else agent1
            try:
                choice = agent_fn(obs_dict)
            except Exception as e:
                stats["exceptions"] += 1
                stats[f"exc:{type(e).__name__}"] += 1
                break
            if not isinstance(choice, list):
                break
            obs_dict = battle_select(choice)
            moves += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass
    games.append(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="submissions/v074_fullmetal")
    ap.add_argument("--opp", default="submissions/external_nursrijan_lucario")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cand_dir = args.candidate if os.path.isabs(args.candidate) else os.path.join(ROOT, args.candidate)
    opp_dir = args.opp if os.path.isabs(args.opp) else os.path.join(ROOT, args.opp)

    hits: list = []
    games: list = []
    stats: Counter = Counter()
    t0 = time.perf_counter()
    for i in range(args.games):
        run_game(cand_dir, opp_dir, our_first=(i % 2 == 0), seed=7000 + i,
                 hits=hits, games=games, stats=stats)
        if (i + 1) % 5 == 0:
            print(f"  game {i+1}/{args.games} hits={len(hits)} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)

    n = len(games)
    with_ppp = [g for g in games if g["ppp_max"] > 0]
    print(f"\n=== PPP PROBE: {os.path.basename(cand_dir)} vs {os.path.basename(opp_dir)} ===")
    print(f"игр: {n}")
    print(f"игр, где 1141 попал в сброс соперника: {len(with_ppp)} "
          f"({100.0*len(with_ppp)/max(1,n):.1f}%)   [в live 25/26 = 96%]")
    print(f"распределение max копий 1141 за игру: "
          f"{dict(sorted(Counter(g['ppp_max'] for g in games).items()))}")
    print(f"распределение max прироста копий между наблюдениями (прокси стакинга за ход): "
          f"{dict(sorted(Counter(g['ppp_max_step_delta'] for g in games).items()))}")
    print(f"игр с Gravity Mountain на столе: {sum(1 for g in games if g['gravity_seen'])}")

    pre = sum(1 for g in with_ppp
              if g["first_big_hit_move"] is not None
              and g["first_ppp_move"] is not None
              and g["first_ppp_move"] < g["first_big_hit_move"])
    big = sum(1 for g in games if g["first_big_hit_move"] is not None)
    print(f"игр с ударом >=200 по нашему активному: {big}; из них копия 1141 была видна "
          f"в сбросе ДО этого удара: {pre}   [в live 12/26]")

    print(f"\nвсего зафиксированных падений hp нашей активной: {len(hits)}")
    by_dmg = Counter(h["dmg"] for h in hits)
    print("распределение входящего урона (>=100):")
    for d in sorted(by_dmg):
        if d >= 100:
            print(f"  {d:4d} -> {by_dmg[d]}")
    luc = [h for h in hits if h["opp_active_id"] in LUCARIO_LINE and h["dmg"] >= 100]
    print(f"\nудары >=100 при {{F}}-lucario в активе: {len(luc)}")
    print(f"  их распределение: {dict(sorted(Counter(h['dmg'] for h in luc).items()))}")
    print(f"  из них с приростом 1141 в том же окне: "
          f"{sum(1 for h in luc if h['ppp_delta'] > 0)}")
    print(f"  из них при уже >=1 копии в сбросе: {sum(1 for h in luc if h['ppp_before'] > 0)}")
    if stats["exceptions"]:
        print(f"\nИСКЛЮЧЕНИЯ: {stats['exceptions']} {[k for k in stats if k.startswith('exc:')]}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(dict(candidate=cand_dir, opponent=opp_dir, games=games, hits=hits,
                           stats=dict(stats)), f, indent=2)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
