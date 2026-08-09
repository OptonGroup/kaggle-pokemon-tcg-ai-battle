#!/usr/bin/env python3
"""verify_r1a_endturn_unit.py — детерминистическое доказательство п.3 ТЗ R1(a),
БЕЗ движка cg_native/libcg.so (тот вносит собственный, неуправляемый Python'ом
RNG на search_step — см. project_native_engine / scripts/gate_scout_guard.py:487-500
и контрольный эксперимент scripts/verify_r1a_endturn_determinism.py, который
показал MISMATCH даже между двумя вызовами ОДНОГО И ТОГО ЖЕ кода (флаг OFF дважды)
из-за общего C-стороннего состояния между последовательными search_begin/search_step
в одном процессе — то есть "живой" стенд НЕ МОЖЕТ дать чистое побитовое сравнение).

Метод: подменяем api/heuristic_choice/evaluate внутри загруженного main.py на
детерминированные заглушки (playout() обращается к ним по именам в своём модуле —
monkey-patch атрибутов модуля работает, т.к. функции читают глобалы модуля в
момент вызова) и скармливаем playout() СКРИПТОВАННУЮ последовательность состояний,
которая один-в-один воспроизводит сценарий из карты поиска (форс-выбор нового
активного соперником ПОСЛЕ нокаута нашей атакой — yourIndex транзитно флипается
на соперника и обратно ВНУТРИ одного и того же state.turn):

    S0 turn=5 yourIndex=0  (после нашего корневого действия, мы продолжаем ход)
    S1 turn=5 yourIndex=1  (соперник вынужден выбрать нового активного — ТРАНЗИТНЫЙ флип)
    S2 turn=5 yourIndex=0  (снова мы, тот же ход 5 — доигрываем)
    S3 turn=6 yourIndex=1  (наш ход РЕАЛЬНО закончился — turn 5->6)

Проверяется (при depth=1):
  A) endturn=False в submissions/scout_r1a_endturn совпадает БУКВАЛЬНО (та же
     evaluate()-точка) с оригиналом submissions/scout_deeper_p0p1.playout() — обе
     ложно останавливаются на S1 (транзитный флип), это и есть horizon effect.
  B) endturn=True останавливается на S3 (настоящий конец хода) — R1(a) исправляет (A).

Запуск:
    .venv/bin/python scripts/verify_r1a_endturn_unit.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(ROOT, "submissions", "scout_deeper_p0p1")
CAND_DIR = os.path.join(ROOT, "submissions", "scout_r1a_endturn")


def _load_module(sub_dir: str, tag: str):
    main_path = os.path.join(sub_dir, "main.py")
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    mod_name = f"_unit_{tag}_{id(object())}"
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


class FakeState:
    """Мимикрирует объект, который возвращает cg.api.search_step: атрибут
    .observation, .searchId."""
    def __init__(self, obs: dict, search_id=1):
        self.observation = obs
        self.searchId = search_id


def make_script():
    """S0..S3, как в докстринге. 'select' с 1 фиктивной опцией — playout() не
    смотрит на содержимое опций, только heuristic_choice (застаблен)."""
    def state(turn, your_index, marker, result=-1):
        return {
            "current": {"turn": turn, "yourIndex": your_index, "result": result,
                        "_marker": marker},
            "select": {"context": 0, "option": [{"type": 7}]},
        }

    s0 = state(5, 0, "S0")
    s1 = state(5, 1, "S1")   # транзитный флип: соперник форс-выбирает нового активного
    s2 = state(5, 0, "S2")   # снова мы, тот же ход
    s3 = state(6, 1, "S3")   # реальный конец нашего хода: turn 5->6
    return [s0, s1, s2, s3]


class FakeAPI:
    """search_step придаёт СЛЕДУЮЩЕЕ состояние скрипта, игнорируя searchId/choice —
    ровно то поведение, которое нужно playout(), чтобы пройти по S1->S2->S3."""
    def __init__(self, script: list):
        self.script = script
        self.pos = 0  # script[0] уже "текущее" (передаётся как starting state)

    def search_step(self, search_id, choice):
        self.pos += 1
        return FakeState(self.script[self.pos], search_id)


def stub_heuristic_choice(obs):
    return [0]


def make_stub_evaluate(calls: list):
    def _evaluate(cur, me):
        calls.append(cur.get("_marker"))
        return {"S0": 0.0, "S1": 0.11, "S2": 0.22, "S3": 0.33}.get(cur.get("_marker"), -1.0)
    return _evaluate


def run_playout(mod, endturn_kw: dict) -> tuple:
    """Возвращает (marker_на_котором_остановились, score)."""
    script = make_script()
    api = FakeAPI(script)
    calls: list = []
    mod.heuristic_choice = stub_heuristic_choice
    mod.evaluate = make_stub_evaluate(calls)
    initial_state = FakeState(script[0], search_id=1)
    score = mod.playout(api, initial_state, 0, 1, **endturn_kw)  # me=0, depth=1
    assert len(calls) == 1, f"evaluate() должен вызваться РОВНО ОДИН РАЗ, вызван {calls}"
    return calls[0], score


def main() -> int:
    ref_mod = _load_module(REF_DIR, "ref")
    cand_mod = _load_module(CAND_DIR, "cand")

    # (A) ref (нет endturn-параметра вовсе) vs cand endturn=False
    ref_marker, ref_score = run_playout(ref_mod, {})
    off_marker, off_score = run_playout(cand_mod, {"endturn": False})
    off_default_marker, off_default_score = run_playout(cand_mod, {})  # default тоже False

    print(f"ref  (эталон, нет endturn-параметра): остановился на {ref_marker}, score={ref_score}")
    print(f"cand endturn=False (явно):            остановился на {off_marker}, score={off_score}")
    print(f"cand endturn=<default>:                остановился на {off_default_marker}, score={off_default_score}")

    ok_a = (ref_marker == off_marker == off_default_marker == "S1"
            and ref_score == off_score == off_default_score)
    print(f"  -> {'PASS' if ok_a else 'FAIL'}: все три останавливаются на S1 (транзитный "
          f"yourIndex-флип), значения совпадают побитово" if ok_a else
          f"  -> FAIL: ожидали S1/S1/S1 с равными score, получили "
          f"{ref_marker}/{off_marker}/{off_default_marker}")

    # (B) cand endturn=True — должен пройти S1 (транзитный) и остановиться на S3 (реальный конец хода)
    on_marker, on_score = run_playout(cand_mod, {"endturn": True})
    print(f"cand endturn=True:                     остановился на {on_marker}, score={on_score}")
    ok_b = (on_marker == "S3")
    print(f"  -> {'PASS' if ok_b else 'FAIL'}: доигрывает мимо транзитного флипа S1 и "
          f"останавливается на настоящей границе хода S3" if ok_b else
          f"  -> FAIL: ожидали S3, получили {on_marker}")

    print()
    if ok_a and ok_b:
        print("PASS ВСЕГО: при endturn=False (в т.ч. default) поведение submissions/"
              "scout_r1a_endturn ПОБИТОВО идентично эталону submissions/scout_deeper_p0p1 "
              "(оба ложно стопятся на транзитном флипе S1); при endturn=True флаг "
              "делает именно то, что заявлено в R1(a) — доигрывает до настоящей границы хода.")
        return 0
    print("FAIL — см. выше.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
