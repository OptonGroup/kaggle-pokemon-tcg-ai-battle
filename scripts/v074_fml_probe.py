"""v074_fml_probe.py — эмпирическая проверка предпосылки P0 (ТЗ docs/gold/PLAN_v074.md).

Вопросы, на которые отвечает прогон (движок cg_native, macOS arm64):

  1. Выкладывают ли archaludon-оппоненты гейт-пула (v044/v036) Full Metal Lab (1244)
     на практике, и в какой доле наших ходов он реально лежит на столе?
  2. Режет ли движок наш Shadow Bullet (180) до 150 по {M}-активному под этим стадионом?
  3. Режет ли он +30 сплеша по {M}-скамейке ДО НУЛЯ (в живом реплее 87839152 сплеш
     наблюдался только по {W}-Relicanth, то есть прямого доказательства по {M} не было)?

Метод: играем кандидата против оппонента, на каждом шаге снимаем доску соперника
(active/bench id+hp) и текущий стадион. Когда НАШ агент выбирает ATTACK, запоминаем
снимок; первый последующий шаг, где доска соперника изменилась, даёт фактический урон
по каждому телу. Урон сопоставляется с типом цели (CardData.energyType) и стадионом.

Запуск:
    .venv/bin/python scripts/v074_fml_probe.py --opp submissions/v044_archaludon_abomasnow \
        --candidate submissions/v073_rockinn_damage --games 20
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

FULL_METAL_LAB = 1244
METAL = 8            # EnergyType.METAL
OPT_ATTACK = 13
MAX_MOVES = 2000

CARD_DB = {c.cardId: c for c in all_card_data()}
_LOAD_COUNTER = 0


def energy_type_of(card_id: int):
    cd = CARD_DB.get(int(card_id or 0))
    et = getattr(cd, "energyType", None) if cd else None
    if et is None:
        return None
    return int(getattr(et, "value", et))


def _load_submission_module(sub_dir: str):
    global _LOAD_COUNTER
    _LOAD_COUNTER += 1
    main_path = os.path.join(sub_dir, "main.py")
    assert os.path.exists(main_path), f"не найден {main_path}"
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    mod_name = f"_fmlprobe_{os.path.basename(sub_dir)}_{_LOAD_COUNTER}"
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
    st = cur.get("stadium") or []
    for c in st:
        if isinstance(c, dict) and c.get("id") is not None:
            return int(c["id"])
    return None


def _board(pl: dict):
    """(active(id,hp) | None, [(id,hp) по слотам скамейки])"""
    act = (pl.get("active") or [None])[0]
    a = (int(act.get("id") or 0), int(act.get("hp") or 0)) if isinstance(act, dict) else None
    bench = []
    for pk in (pl.get("bench") or []):
        bench.append((int(pk.get("id") or 0), int(pk.get("hp") or 0)) if isinstance(pk, dict) else None)
    return a, bench


def run_game(cand_dir: str, opp_dir: str, our_first: bool, seed: int, ev: list, stats: Counter):
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
    pending = None       # контекст последней объявленной нами атаки
    prev = None          # снимок доски соперника на предыдущем шаге
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
            stadium = _stadium_id(cur)
            act, bench = _board(opp_pl)

            if yi == our_side:
                stats["our_steps"] += 1
                if stadium == FULL_METAL_LAB:
                    stats["our_steps_under_fml"] += 1

            # разрешение уже объявленной атаки: первое изменение доски соперника
            # Каждое падение hp на доске соперника пишем как событие и приписываем ему
            # контекст ПОСЛЕДНЕЙ нашей объявленной атаки (steps_since — сколько шагов
            # прошло; урон атаки прилетает в пределах 1-3 шагов, дальше это уже чекап/
            # способности). Не очищаем контекст на первом изменении: сплеш Shadow Bullet
            # приходит отдельным шагом от урона по активному.
            if prev is not None and pending is not None:
                p_act, p_bench = prev
                p_stad, p_attacker, p_atkid, p_step = pending
                since = moves - p_step
                if since <= 6:
                    if p_act and act and p_act[0] == act[0] and act[1] < p_act[1]:
                        ev.append(dict(kind="active", stadium=p_stad, target_id=p_act[0],
                                       etype=energy_type_of(p_act[0]), attacker=p_attacker,
                                       attack_id=p_atkid, steps_since=since,
                                       hp_before=p_act[1], dmg=p_act[1] - act[1]))
                    for i, before in enumerate(p_bench):
                        after = bench[i] if i < len(bench) else None
                        if before and after and before[0] == after[0] and after[1] < before[1]:
                            ev.append(dict(kind="bench", stadium=p_stad, target_id=before[0],
                                           etype=energy_type_of(before[0]), attacker=p_attacker,
                                           attack_id=p_atkid, steps_since=since,
                                           hp_before=before[1], dmg=before[1] - after[1]))
            prev = (act, bench)

            sel = obs_dict.get("select")
            agent_fn = agent0 if yi == 0 else agent1
            try:
                choice = agent_fn(obs_dict)
            except Exception as e:
                stats["exceptions"] += 1
                stats[f"exc:{type(e).__name__}"] += 1
                break
            if not isinstance(choice, list):
                break

            if yi == our_side and sel:
                opts = sel.get("option") or []
                my_act = (players[our_side].get("active") or [None])[0]
                my_act_id = int(my_act.get("id") or 0) if isinstance(my_act, dict) else 0
                for i in choice:
                    if 0 <= i < len(opts) and int((opts[i] or {}).get("type") or 0) == OPT_ATTACK:
                        atk_id = (opts[i] or {}).get("attackId")
                        pending = (stadium, my_act_id,
                                   int(atk_id) if atk_id is not None else None, moves)
                        stats["our_attacks"] += 1
                        if stadium == FULL_METAL_LAB:
                            stats["our_attacks_under_fml"] += 1
                        break

            obs_dict = battle_select(choice)
            moves += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="submissions/v073_rockinn_damage")
    ap.add_argument("--opp", default="submissions/v044_archaludon_abomasnow")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cand_dir = os.path.join(ROOT, args.candidate) if not os.path.isabs(args.candidate) else args.candidate
    opp_dir = os.path.join(ROOT, args.opp) if not os.path.isabs(args.opp) else args.opp

    ev: list = []
    stats: Counter = Counter()
    t0 = time.perf_counter()
    for g in range(args.games):
        run_game(cand_dir, opp_dir, our_first=(g % 2 == 0), seed=1000 + g, ev=ev, stats=stats)
        print(f"  game {g+1}/{args.games} events={len(ev)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    print("\n=== СТАТИСТИКА ===")
    print(f"наших шагов: {stats['our_steps']}, из них под Full Metal Lab: "
          f"{stats['our_steps_under_fml']} "
          f"({100.0*stats['our_steps_under_fml']/max(1,stats['our_steps']):.1f}%)")
    print(f"наших атак: {stats['our_attacks']}, из них под FML: {stats['our_attacks_under_fml']} "
          f"({100.0*stats['our_attacks_under_fml']/max(1,stats['our_attacks']):.1f}%)")
    if stats["exceptions"]:
        print(f"ИСКЛЮЧЕНИЯ: {stats['exceptions']} {[k for k in stats if k.startswith('exc:')]}")

    # распределение фактического урона: (kind, стадион=FML?, тип цели METAL?) -> Counter(dmg)
    buckets: dict = defaultdict(Counter)
    for e in ev:
        if e.get("steps_since", 0) > 3:
            continue   # дальше 3 шагов это уже чекап/способности, не урон атаки
        key = (e.get("attacker"), e.get("attack_id"), e["kind"],
               e["stadium"] == FULL_METAL_LAB, e["etype"] == METAL)
        buckets[key][e["dmg"]] += 1
    print("\n=== ФАКТИЧЕСКИЙ УРОН (атакующий, attackId, kind, под_FML, цель_{M}) -> {урон: раз} ===")
    for key in sorted(buckets, key=lambda k: (k[0] or 0, k[1] or 0, k[2], k[3], k[4])):
        attacker, atk_id, kind, fml, metal = key
        dist = dict(sorted(buckets[key].items()))
        print(f"  attacker={attacker} atk={atk_id} {kind:6s} FML={str(fml):5s} "
              f"target_M={str(metal):5s} -> {dist}")

    if args.out:
        out = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(dict(stats=dict(stats), events=ev), f, indent=1)
        print(f"\nсырые события -> {out}")


if __name__ == "__main__":
    main()
