"""pimc_determinizer_smoke.py — S2 смок: юнит-проверки детектора оппонента /
мета-микса в src/exit/pimc_agent.py, БЕЗ движка (чистый Python, работает и на macOS).

Что проверяем (S2, см. PLAN.md §S):
  1. identify_archetype(revealed) на синтетических revealed-множествах:
     {741}->alakazam, {648}->grimmsnarl, {506}->cubchoo, {345}->crustle, {}->None.
  2. sample_meta_archetype()/sample_meta_deck() за N сэмплов бьётся с META_PRIORS
     (архетипы без своего дека — "archaludon"/"generic" — оба маппятся на
     CRUSTLE_DECK_APPROX, поэтому сверяем агрегированную долю по РЕЗУЛЬТИРУЮЩЕЙ деке).
  3. Накопление revealed + лок на архетип на уровне PIMCAgent-инстанса (без запуска
     движка — просто дёргаем внутреннюю логику через синтетические obs_dict).

Запуск:
    .venv/bin/python scripts/pimc_determinizer_smoke.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from exit.pimc_agent import (  # noqa: E402
    ALAKAZAM_DECK,
    GRIMMSNARL_DECK,
    CUBCHOO_DECK,
    CRUSTLE_DECK_APPROX,
    META_PRIORS,
    identify_archetype,
    sample_meta_archetype,
    sample_meta_deck,
    _extract_revealed_ids,
    _deck_for_archetype,
)

n_fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global n_fail
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        n_fail += 1


# ---------------------------------------------------------------------------
# 1. identify_archetype на синтетических revealed-множествах
# ---------------------------------------------------------------------------
print("[1/3] identify_archetype(revealed) — сигнатурная детекция", flush=True)

check("741 (Abra) -> alakazam", identify_archetype({741}) == "alakazam")
check("742/743 (Kadabra/Alakazam) -> alakazam", identify_archetype({742, 743}) == "alakazam")
check("648 (Grimmsnarl ex) -> grimmsnarl", identify_archetype({648}) == "grimmsnarl")
check("646/647 (Impidimp/Morgrem) -> grimmsnarl", identify_archetype({646, 647}) == "grimmsnarl")
check("506 (Cubchoo) -> cubchoo", identify_archetype({506}) == "cubchoo")
check("345 (Crustle) -> crustle", identify_archetype({345}) == "crustle")
check("344 (Dwebble) -> crustle", identify_archetype({344}) == "crustle")
check("пустое множество -> None", identify_archetype(set()) is None)
check("нерелевантные ID (напр. 4=энергия) -> None", identify_archetype({4, 999}) is None)

# Деки за архетипами верны и по 60 карт
check("_deck_for_archetype('alakazam') is ALAKAZAM_DECK (60)",
      _deck_for_archetype("alakazam") == list(ALAKAZAM_DECK) and len(ALAKAZAM_DECK) == 60)
check("_deck_for_archetype('grimmsnarl') is GRIMMSNARL_DECK (60)",
      _deck_for_archetype("grimmsnarl") == list(GRIMMSNARL_DECK) and len(GRIMMSNARL_DECK) == 60)
check("_deck_for_archetype('cubchoo') is CUBCHOO_DECK (60)",
      _deck_for_archetype("cubchoo") == list(CUBCHOO_DECK) and len(CUBCHOO_DECK) == 60)
check("_deck_for_archetype('crustle') is CRUSTLE_DECK_APPROX (60)",
      _deck_for_archetype("crustle") == list(CRUSTLE_DECK_APPROX) and len(CRUSTLE_DECK_APPROX) == 60)
check("_deck_for_archetype('archaludon') -> CRUSTLE placeholder (TODO)",
      _deck_for_archetype("archaludon") == list(CRUSTLE_DECK_APPROX))
check("_deck_for_archetype(None) -> CRUSTLE placeholder",
      _deck_for_archetype(None) == list(CRUSTLE_DECK_APPROX))


# ---------------------------------------------------------------------------
# 2. Распределение мета-микса за N сэмплов vs META_PRIORS
# ---------------------------------------------------------------------------
print("\n[2/3] sample_meta_archetype() — распределение за N сэмплов vs META_PRIORS", flush=True)

N = 4000
counts: dict[str, int] = {k: 0 for k in META_PRIORS}
for _ in range(N):
    counts[sample_meta_archetype()] += 1

print(f"  N={N} сэмплов:")
TOL = 0.035  # абсолютный допуск (std для p=0.33,N=4000 ~ 0.0074 -> 0.035 щедрый запас)
for name, prior in META_PRIORS.items():
    freq = counts[name] / N
    ok = abs(freq - prior) <= TOL
    check(f"archetype={name:11s} prior={prior:.2f} observed={freq:.3f}", ok)

# Также агрегированная доля per-DECK (archaludon+generic+crustle все -> CRUSTLE_DECK_APPROX)
n_alakazam = n_grimmsnarl = n_cubchoo = n_crustle_placeholder = 0
for _ in range(N):
    d = sample_meta_deck()
    if d == list(ALAKAZAM_DECK):
        n_alakazam += 1
    elif d == list(GRIMMSNARL_DECK):
        n_grimmsnarl += 1
    elif d == list(CUBCHOO_DECK):
        n_cubchoo += 1
    elif d == list(CRUSTLE_DECK_APPROX):
        n_crustle_placeholder += 1

expected_crustle_placeholder = META_PRIORS["archaludon"] + META_PRIORS["crustle"] + META_PRIORS["generic"]
print(f"\n  Агрегация по результирующей деке (N={N}):")
check(f"alakazam deck freq~{META_PRIORS['alakazam']:.2f}", abs(n_alakazam / N - META_PRIORS["alakazam"]) <= TOL,
      f"observed={n_alakazam / N:.3f}")
check(f"grimmsnarl deck freq~{META_PRIORS['grimmsnarl']:.2f}", abs(n_grimmsnarl / N - META_PRIORS["grimmsnarl"]) <= TOL,
      f"observed={n_grimmsnarl / N:.3f}")
check(f"cubchoo deck freq~{META_PRIORS['cubchoo']:.2f}", abs(n_cubchoo / N - META_PRIORS["cubchoo"]) <= TOL,
      f"observed={n_cubchoo / N:.3f}")
check(f"CRUSTLE-placeholder (archaludon+crustle+generic) freq~{expected_crustle_placeholder:.2f}",
      abs(n_crustle_placeholder / N - expected_crustle_placeholder) <= TOL,
      f"observed={n_crustle_placeholder / N:.3f}")


# ---------------------------------------------------------------------------
# 3. Накопление revealed за "партию" (несколько синтетических obs_dict) + лок
# ---------------------------------------------------------------------------
print("\n[3/3] Накопление revealed card ID оппонента за несколько ходов + лок на архетип", flush=True)


def _fake_obs(your_index: int, opp_discard: list[int], opp_active_id: int | None,
              opp_bench_ids: list[int] | None = None) -> dict:
    opp_bench_ids = opp_bench_ids or []

    def _poke(cid):
        return {"id": cid, "energyCards": []} if cid is not None else None

    def _player(discard_ids, active_id, bench_ids):
        return {
            "discard": [{"id": c} for c in discard_ids],
            "active": [_poke(active_id)],
            "bench": [_poke(c) for c in bench_ids],
            "prize": [None] * 6,
        }

    me = _player([], None, [])
    opp = _player(opp_discard, opp_active_id, opp_bench_ids)
    players = [me, opp] if your_index == 0 else [opp, me]
    return {"current": {"yourIndex": your_index, "players": players}}


# Ход 1: revealed пусто -> не опознан
obs1 = _fake_obs(0, opp_discard=[], opp_active_id=None)
revealed = set()
revealed |= _extract_revealed_ids(obs1)
check("ход1: revealed пуст -> identify_archetype None", identify_archetype(revealed) is None)

# Ход 2: оппонент разыгрывает Kadabra (742) на bench -> сигнатура alakazam появляется
obs2 = _fake_obs(0, opp_discard=[1086], opp_active_id=None, opp_bench_ids=[742])
revealed |= _extract_revealed_ids(obs2)
check("ход2: revealed содержит 742 -> identify_archetype == alakazam",
      identify_archetype(revealed) == "alakazam", detail=str(sorted(revealed)))

# Ход 3: дальнейшие карты не должны сбить уже опознанный архетип (лок держится на PIMCAgent,
# здесь просто проверяем что повторный вызов идемпотентен)
obs3 = _fake_obs(0, opp_discard=[1086, 1079], opp_active_id=741)
revealed |= _extract_revealed_ids(obs3)
check("ход3: архетип остаётся alakazam после новых карт того же архетипа",
      identify_archetype(revealed) == "alakazam")

# Проверка изоляции сторон: свои карты (player 0, если yourIndex=0) НЕ должны попадать
# в revealed оппонента даже если ID совпадает с сигнатурой.
obs_self_only = {
    "current": {
        "yourIndex": 0,
        "players": [
            {"discard": [{"id": 741}], "active": [None], "bench": [], "prize": [None] * 6},  # моя сторона
            {"discard": [], "active": [None], "bench": [], "prize": [None] * 6},               # оппонент чист
        ],
    }
}
own_side_revealed = _extract_revealed_ids(obs_self_only)
check("свои карты (741 в МОЁМ discard) НЕ попадают в revealed оппонента",
      741 not in own_side_revealed, detail=str(own_side_revealed))


# ---------------------------------------------------------------------------
# Итог
# ---------------------------------------------------------------------------
print(f"\n{'СМОК ПРОЙДЕН' if n_fail == 0 else 'СМОК НЕ ПРОЙДЕН'}: fails={n_fail}")
sys.exit(0 if n_fail == 0 else 1)
