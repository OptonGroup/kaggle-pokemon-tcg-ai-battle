"""repro_v064_bug9.py — юнит-репро БАГ-9 (hand-scaling атаки, напр. Alakazam Powerful Hand).

Самодостаточный тест БЕЗ движка: подсовывает минимальные заглушки под `cg.api` (только
имена, которые `scoring_grimmsnarl.py` импортирует на уровне модуля — AreaType/LogType/
OptionType/SelectContext/all_card_data/all_attack/to_observation_class), затем импортирует
submissions/v064_starmie_scaling/scoring_grimmsnarl.py напрямую по пути через importlib и
вызывает opp_max_attack_damage(obs, target) на мок-obs.

Проверяет ровно то, что описано в ТЗ v064 БАГ-9:
  (а) Alakazam(743) с рукой 15 карт -> threat >= 300 (20 HP/карта * 15 = 300 ровно).
  (б) та же атака с рукой 0 карт -> threat == listed damage (0 в моках -- Powerful Hand
      сам по себе listed=0/"n/a", это и есть баг: без scaling-фикса threat всегда 0).
  (в) opp active НЕ 743 (обычная листед-damage атака) -> scaling не влияет, результат
      равен обычному weakness-adjusted listed damage (SCALING_ATTACKERS не должен
      трогать любую другую атаку).

Запуск:
    .venv/bin/python scripts/repro_v064_bug9.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORER_PATH = os.path.join(ROOT, "submissions", "v064_starmie_scaling", "scoring_grimmsnarl.py")


# ── Minimal cg.api stub (только то, что scoring_grimmsnarl.py трогает на уровне модуля) ──

class _AttrSentinel:
    """Любой доступ к атрибуту возвращает имя атрибута -- достаточно, чтобы AreaType.DECK,
    LogType.TURN_END и т.п. не падали при импорте (они не вызываются в этом тесте, только
    существуют как объекты в момент `from cg.api import ...`)."""

    def __getattr__(self, name):
        return name


class _CardData:
    def __init__(self, cardId, attacks=(), weakness=None, energyType=None):
        self.cardId = cardId
        self.attacks = list(attacks)
        self.weakness = weakness
        self.energyType = energyType


class _Attack:
    def __init__(self, attackId, damage, energies):
        self.attackId = attackId
        self.damage = damage
        self.energies = list(energies)


ALAKAZAM_ID = 743
POWERFUL_HAND_ATTACK_ID = 90001
OTHER_ACTIVE_ID = 999          # непричастная к scaling атака, для проверки (в)
OTHER_ATTACK_ID = 90002
OTHER_ATTACK_DAMAGE = 50

_FAKE_CARD_DB = [
    _CardData(ALAKAZAM_ID, attacks=[POWERFUL_HAND_ATTACK_ID]),   # Powerful Hand: listed=0/"n/a"
    _CardData(OTHER_ACTIVE_ID, attacks=[OTHER_ATTACK_ID]),        # обычная листед-damage атака
]
_FAKE_ATTACKS = [
    _Attack(POWERFUL_HAND_ATTACK_ID, damage=0, energies=["P"]),   # cost {P}=1, listed dmg=0
    _Attack(OTHER_ATTACK_ID, damage=OTHER_ATTACK_DAMAGE, energies=[]),  # cost {0}, dmg=50
]


def _install_cg_stub() -> None:
    cg_pkg = types.ModuleType("cg")
    cg_api = types.ModuleType("cg.api")
    cg_api.AreaType = _AttrSentinel()
    cg_api.LogType = _AttrSentinel()
    cg_api.OptionType = _AttrSentinel()
    cg_api.SelectContext = _AttrSentinel()
    cg_api.all_card_data = lambda: list(_FAKE_CARD_DB)
    cg_api.all_attack = lambda: list(_FAKE_ATTACKS)
    cg_api.to_observation_class = lambda obs_dict: obs_dict  # не вызывается в этом тесте
    cg_pkg.api = cg_api
    sys.modules["cg"] = cg_pkg
    sys.modules["cg.api"] = cg_api


def _load_scorer():
    spec = importlib.util.spec_from_file_location("_repro_v064_bug9_scorer", SCORER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Мок-объекты obs (только то, что opp_max_attack_damage/opp_state/opp_active_pokemon трогают) ──

class _MockPokemon:
    def __init__(self, id_, hp=999, energies=None):
        self.id = id_
        self.hp = hp
        self.energies = energies if energies is not None else []


class _MockPlayerState:
    def __init__(self, active=None, handCount=0):
        self.active = [active] if active is not None else []
        self.bench = []
        self.hand = []
        self.discard = []
        self.prize = []
        self.handCount = handCount


class _MockCurrent:
    def __init__(self, yourIndex, players):
        self.yourIndex = yourIndex
        self.players = players


class _MockObs:
    def __init__(self, current):
        self.current = current


def _make_obs(opp_active_id: int, opp_energies: list, opp_hand: int):
    """yourIndex=0 -- наш игрок; players[1] = оппонент (opp_state(obs) читает 1-yourIndex)."""
    my_pl = _MockPlayerState(active=None, handCount=0)
    opp_pl = _MockPlayerState(
        active=_MockPokemon(opp_active_id, hp=999, energies=opp_energies),
        handCount=opp_hand,
    )
    return _MockObs(_MockCurrent(yourIndex=0, players=[my_pl, opp_pl]))


def main() -> int:
    _install_cg_stub()
    scorer = _load_scorer()

    assert scorer.SCALING_ATTACKERS.get(ALAKAZAM_ID) == (20, 1), (
        f"SCALING_ATTACKERS[743] неожиданный: {scorer.SCALING_ATTACKERS.get(ALAKAZAM_ID)}"
    )

    target = _MockPokemon(id_=648, hp=320)  # наш Grimmsnarl ex, 320 HP, не в CARD_DB мока
    # (нет записи в CARD_DB под 648 -> weakness lookup вернёт None -> без *2, что и нужно
    #  для чистой проверки scaling-величины без лишней переменной).

    results = []

    # (а) Alakazam(743), рука=15 -> threat >= 300 (20*15=300 ровно).
    obs_a = _make_obs(ALAKAZAM_ID, opp_energies=[1], opp_hand=15)
    threat_a = scorer.opp_max_attack_damage(obs_a, target)
    ok_a = threat_a >= 300
    results.append(("(a) Alakazam hand=15 -> threat>=300", ok_a, f"got {threat_a}"))

    # (б) Alakazam(743), рука=0 -> threat == listed (0 в моках: Powerful Hand listed dmg=0).
    obs_b = _make_obs(ALAKAZAM_ID, opp_energies=[1], opp_hand=0)
    threat_b = scorer.opp_max_attack_damage(obs_b, target)
    ok_b = threat_b == 0
    results.append(("(b) Alakazam hand=0 -> threat==listed(0)", ok_b, f"got {threat_b}"))

    # (в) opp active НЕ 743 -> scaling не влияет, результат = обычный listed damage (50).
    obs_c = _make_obs(OTHER_ACTIVE_ID, opp_energies=[], opp_hand=31)  # большая рука -- не должна влиять
    threat_c = scorer.opp_max_attack_damage(obs_c, target)
    ok_c = threat_c == OTHER_ATTACK_DAMAGE
    results.append(("(c) opp active!=743 -> scaling не влияет, threat==listed(50)", ok_c, f"got {threat_c}"))

    all_ok = True
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name} ({detail})")
        all_ok = all_ok and ok

    print("\n=== " + ("ALL PASS" if all_ok else "SOME FAILED") + " ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
