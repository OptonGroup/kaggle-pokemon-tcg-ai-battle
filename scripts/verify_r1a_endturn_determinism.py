#!/usr/bin/env python3
"""verify_r1a_endturn_determinism.py — ПОПЫТКА доказательства п.3 ТЗ R1(a) на живом
движке; РЕЗУЛЬТАТ (запуск 2026-08-08): МЕТОДОЛОГИЯ НЕВАЛИДНА для этой цели — не
использовать вывод этого скрипта как признак бага в SCOUT_R1A_ENDTURN.

Почему: cg_native/libcg.so хранит СОБСТВЕННОЕ, не управляемое Python'ом RNG-состояние,
общее на весь процесс между последовательными search_begin/search_step (см.
scripts/gate_scout_guard.py:487-500 — "два прогона ОДНОГО И ТОГО ЖЕ кандидата с одним
сидом дали 50/0/0 и 49/0/1"). Контрольный прогон здесь ЭТО ПОДТВЕРДИЛ: cand_off
(ENV не задан) и cand_off0 (ENV="0") — ОДИН И ТОТ ЖЕ код, endturn=False оба раза —
разошлись в 13/25 снимков ТОЛЬКО из-за порядка вызовов в одном процессе (каждый вызов
agent() тратит engine-side RNG движка, который random.seed() не резетит). Значит
"ref != off" в выводе ниже — это тот же самый шум, а НЕ признак нарушения п.3.

Авторитетное, действительно детерминированное доказательство п.3 — см.
scripts/verify_r1a_endturn_unit.py (без движка, мок search_step, синтетический сценарий
транзитного yourIndex-флипа) — ТАМ playout() с endturn=False побитово идентичен
эталону (одинаковый evaluate()-score с точностью до бита), а endturn=True доказуемо
доигрывает мимо транзитного флипа до настоящей границы хода.

Этот файл оставлен как диагностика самого ФЕНОМЕНА (движок недетерминирован между
последовательными вызовами в одном процессе) — полезно знать при интерпретации ЛЮБОГО
живого A/B на cg_native, не только R1(a).

Метод (снимки, не полные live-игры — исключает нестабильность внутреннего RNG
движка cg_native/libcg.so, который НЕ управляется random.seed() и делает два
прогона ОДНОЙ live-партии невоспроизводимыми, см. scripts/gate_scout_guard.py:487-500):
  1. Играем ОДНУ реальную партию (кандидат=эталон vs meta-оппонент) на cg_native,
     собираем все наблюдения в CTX_MAIN с >=2 опциями (это ровно то подмножество,
     где main.py вообще пробует search_choice()).
  2. На каждом таком снимке дважды (эталон / кандидат-OFF), с ОДИНАКОВЫМ
     random.seed() непосредственно перед вызовом, вызываем agent(obs) на СВЕЖЕМ
     инстансе модуля (чтобы эпизодный бюджет _SPENT не тёк между вызовами) —
     сравниваем возвращаемый список индексов.
  3. Дополнительно (позитивный контроль): кандидат-ON на тех же снимках — должен
     хотя бы иногда ОТличаться от OFF, иначе флаг ничего не делает.

Запуск:
    .venv/bin/python scripts/verify_r1a_endturn_determinism.py
"""
from __future__ import annotations

import copy
import importlib.util
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cg_native.game import battle_start, battle_finish, battle_select  # noqa: E402
import cg_native, cg_native.api, cg_native.game, cg_native.sim, cg_native.utils  # noqa: E402
sys.modules.setdefault("cg", cg_native)
sys.modules.setdefault("cg.api", cg_native.api)
sys.modules.setdefault("cg.game", cg_native.game)
sys.modules.setdefault("cg.sim", cg_native.sim)
sys.modules.setdefault("cg.utils", cg_native.utils)

REF_DIR = os.path.join(ROOT, "submissions", "scout_deeper_p0p1")
CAND_DIR = os.path.join(ROOT, "submissions", "scout_r1a_endturn")
OPP_DECK = os.path.join(ROOT, "decks", "meta", "Dragapult.csv")

CTX_MAIN = 0
SNAP_SEED = 424242          # сид для СБОРА снимков (одна живая партия)
CALL_SEED = 7               # сид перед КАЖДЫМ изолированным вызовом agent()
MAX_SNAPSHOTS = 25
MAX_MOVES = 2000


def _load_deck(path):
    with open(path) as f:
        return [int(x.strip()) for x in f if x.strip()]


def _load_module(sub_dir, tag):
    """Свежий инстанс main.py — чтобы _SPENT/эпизодный бюджет не тёк между вызовами."""
    main_path = os.path.join(sub_dir, "main.py")
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    mod_name = f"_verify_{tag}_{id(object())}"
    cwd0 = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(mod_name, main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = main_path
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd0)
    return mod


def collect_snapshots() -> list:
    """Одна живая партия (эталон-агент vs greedy meta), снимки CTX_MAIN с >=2 опций."""
    ref_mod = _load_module(REF_DIR, "collect")
    d0 = ref_mod.read_deck()
    d1 = _load_deck(OPP_DECK)

    def greedy(obs):
        sel = obs.get("select")
        if sel is None:
            return []
        opts = sel.get("option") or []
        n = len(opts)
        if n == 0:
            return []
        prio = (13, 10, 9, 7, 8, 12, 14)  # ATTACK, ABILITY, EVOLVE, PLAY, ATTACH, RETREAT, END
        for ot in prio:
            idxs = [i for i, o in enumerate(opts) if o.get("type") == ot]
            if idxs:
                return [idxs[0]]
        lo, hi = sel.get("minCount") or 0, min(sel.get("maxCount") or 1, n)
        return list(range(max(lo, min(1, hi))))

    random.seed(SNAP_SEED)
    obs_dict, _ = battle_start(d0, d1)
    snapshots = []
    moves = 0
    try:
        while obs_dict is not None and moves < MAX_MOVES:
            cur = obs_dict.get("current") or {}
            if cur.get("result", -1) != -1:
                break
            sel = obs_dict.get("select")
            if sel is None:
                break
            you = cur.get("yourIndex", 0)
            if you == 0 and sel.get("context") == CTX_MAIN and len(sel.get("option") or []) >= 2:
                if len(snapshots) < MAX_SNAPSHOTS:
                    snapshots.append(copy.deepcopy(obs_dict))
            act = ref_mod.agent(obs_dict) if you == 0 else greedy(obs_dict)
            obs_dict = battle_select(act)
            moves += 1
    finally:
        try:
            battle_finish()
        except Exception:
            pass
    return snapshots


def decide(sub_dir: str, tag: str, obs_dict: dict, env_value: str | None):
    """Свежий модуль + фиксированный random.seed() + опциональный ENV-флаг."""
    if env_value is None:
        os.environ.pop("SCOUT_R1A_ENDTURN", None)
    else:
        os.environ["SCOUT_R1A_ENDTURN"] = env_value
    mod = _load_module(sub_dir, tag)
    random.seed(CALL_SEED)
    return mod.agent(copy.deepcopy(obs_dict))


def main() -> int:
    print("Собираем снимки (одна живая партия, cg_native)...")
    snapshots = collect_snapshots()
    print(f"Снимков CTX_MAIN (>=2 опций, наш ход): {len(snapshots)}")
    if not snapshots:
        print("НЕТ снимков — партия не дошла до нужных решений, проверь движок/деки.")
        return 1

    n_match_off = 0
    n_diff_on = 0
    mismatches = []
    for i, snap in enumerate(snapshots):
        ref = decide(REF_DIR, "ref", snap, None)
        cand_off = decide(CAND_DIR, "off", snap, None)          # ENV не установлен
        cand_off0 = decide(CAND_DIR, "off0", snap, "0")          # ENV="0" явно
        cand_on = decide(CAND_DIR, "on", snap, "1")

        ok_off = (ref == cand_off == cand_off0)
        if ok_off:
            n_match_off += 1
        else:
            mismatches.append((i, ref, cand_off, cand_off0))
        if cand_on != cand_off:
            n_diff_on += 1
        n_opts = len(snap["select"]["option"])
        print(f"  snap {i:2d} n_opt={n_opts:2d} ref={ref} off={cand_off} off(env=0)={cand_off0} "
              f"on={cand_on} {'OK' if ok_off else 'MISMATCH'}"
              f"{'  <-- ON differs' if cand_on != cand_off else ''}")

    print()
    print(f"OFF == эталон: {n_match_off}/{len(snapshots)}")
    print(f"ON  != OFF   : {n_diff_on}/{len(snapshots)}  (позитивный контроль: флаг реально что-то меняет)")
    if mismatches:
        print("MISMATCHES (флаг OFF не побитово идентичен эталону!):")
        for m in mismatches:
            print("  ", m)
        return 1
    if n_diff_on == 0:
        print("ПРЕДУПРЕЖДЕНИЕ: ON никогда не отличался от OFF на этих снимках — "
              "либо все снимки не доходили до реальной смены хода при depth, либо "
              "правка не активна. Проверить вручную.")
    print("PASS: OFF побитово идентичен эталону на всех снимках.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
