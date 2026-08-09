#!/usr/bin/env python3
"""verify_r2_rootpar_unit.py — детерминистическое доказательство R2 (root-parallel +
бюджет, submissions/scout_r2_rootpar) БЕЗ движка cg_native/libcg.so.

Почему без движка: см. scripts/verify_r1a_endturn_determinism.py — cg_native хранит
собственное, не управляемое Python'ом RNG-состояние между последовательными
search_begin/search_step в одном процессе, поэтому "живое" побитовое сравнение по
снимкам ненадёжно. Здесь монки-патчим сам module-level search_choice() (то же имя,
что playout() читает как глобал модуля в момент вызова — тот же трюк, что и
verify_r1a_endturn_unit.py с heuristic_choice/evaluate) и проверяем ЧИСТУЮ логику
диспетчера/голосования/бюджета в изоляции.

Проверяется:
  A) SCOUT_R2_ROOTPAR не задан / "" / "0" / "1" / "abc" (мусор) => search_choice_dispatch
     вызывает search_choice() РОВНО ОДИН РАЗ и возвращает его результат БЕЗ изменений
     (т.е. при выключенном флаге дальнейший путь неотличим от эталона, где
     search_choice() — вообще единственная точка входа).
  B) SCOUT_R2_ROOTPAR=K (K>=2) => search_choice_dispatch вызывает search_choice() РОВНО
     K раз, голосует за argmax по возвращённым индексам; большинство побеждает;
     ничья разрешается в пользу МЕНЬШЕГО индекса (тот же тай-брейк, что и в самом
     search_choice(): max(scored) по (score, -i)).
  C) SEARCH_CFG['budget_ms'] на время root-parallel обёртки равен ИСХОДНОМУ, делённому
     на K (округление вниз, минимум 1), и восстанавливается в исходное значение сразу
     после возврата search_choice_dispatch (побочных эффектов на дальнейшие вызовы нет).
  D) ENV-оверрайды (B) из ТЗ (SCOUT_R2_SAMPLES/SCOUT_R2_BUDGET_MS/SCOUT_R2_EP_SOFT/
     SCOUT_R2_EP_HARD), заданные ДО импорта модуля, читаются в SEARCH_CFG/EPISODE_*_S;
     при отсутствии ENV — совпадают со значениями policy.json/дефолтами кода (те же,
     что у эталона submissions/scout_deeper_p0p1, который эти ENV-имена вообще не
     читает — сверяем оба модуля рядом).

Запуск:
    .venv/bin/python scripts/verify_r2_rootpar_unit.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(ROOT, "submissions", "scout_deeper_p0p1")
CAND_DIR = os.path.join(ROOT, "submissions", "scout_r2_rootpar")


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


def _clear_env():
    for k in ("SCOUT_R2_ROOTPAR", "SCOUT_R2_SAMPLES", "SCOUT_R2_BUDGET_MS",
              "SCOUT_R2_EP_SOFT", "SCOUT_R2_EP_HARD"):
        os.environ.pop(k, None)


FAKE_OBS = {"select": {"context": 0, "option": [{"type": 7}, {"type": 14}]},
            "current": {"turn": 1, "yourIndex": 0, "result": -1}}


def make_counting_stub(fixed_pick):
    calls = []

    def _stub(obs):
        calls.append(obs)
        return list(fixed_pick)
    return _stub, calls


def make_sequence_stub(picks):
    calls = {"n": 0}

    def _stub(obs):
        i = calls["n"]
        calls["n"] += 1
        return list(picks[i])
    return _stub, calls


def test_a_disabled(mod) -> bool:
    print("--- (A) флаг выключен: dispatch == прямой вызов search_choice, 1 раз ---")
    ok = True
    for env_val in (None, "", "0", "1", "abc", "-3"):
        _clear_env()
        if env_val is not None:
            os.environ["SCOUT_R2_ROOTPAR"] = env_val
        stub, calls = make_counting_stub([1])
        mod.search_choice = stub
        result = mod.search_choice_dispatch(FAKE_OBS)
        n_calls = len(calls)
        good = (n_calls == 1 and result == [1])
        ok = ok and good
        print(f"  ROOTPAR={env_val!r:>6}: calls={n_calls} result={result} "
              f"{'OK' if good else 'FAIL'}")
    return ok


def test_b_voting(mod) -> bool:
    print("--- (B) K>=2: голосование за argmax, тай-брейк по меньшему индексу ---")
    ok = True

    # 1) явное большинство: [2],[2],[0] -> winner=2
    _clear_env()
    os.environ["SCOUT_R2_ROOTPAR"] = "3"
    stub, calls = make_sequence_stub([[2], [2], [0]])
    mod.search_choice = stub
    result = mod.search_choice_dispatch(FAKE_OBS)
    good = (calls["n"] == 3 and result == [2])
    ok = ok and good
    print(f"  K=3 picks=[2,2,0] -> calls={calls['n']} result={result} "
          f"(expect [2]) {'OK' if good else 'FAIL'}")

    # 2) ничья 2:2 между индексами 5 и 2 -> winner=min(5,2)=2
    _clear_env()
    os.environ["SCOUT_R2_ROOTPAR"] = "4"
    stub, calls = make_sequence_stub([[5], [2], [5], [2]])
    mod.search_choice = stub
    result = mod.search_choice_dispatch(FAKE_OBS)
    good = (calls["n"] == 4 and result == [2])
    ok = ok and good
    print(f"  K=4 picks=[5,2,5,2] (tie) -> calls={calls['n']} result={result} "
          f"(expect [2], меньший индекс) {'OK' if good else 'FAIL'}")

    # 3) одно из деревьев вернуло None (сбой search_choice внутри) -> голос не считается
    _clear_env()
    os.environ["SCOUT_R2_ROOTPAR"] = "3"

    def _stub_with_none(obs, _n=[0]):
        _n[0] += 1
        return [7] if _n[0] != 2 else None
    calls_n = []
    orig = _stub_with_none

    def _wrapped(obs):
        calls_n.append(1)
        return orig(obs)
    mod.search_choice = _wrapped
    result = mod.search_choice_dispatch(FAKE_OBS)
    good = (len(calls_n) == 3 and result == [7])
    ok = ok and good
    print(f"  K=3 picks=[7,None,7] -> calls={len(calls_n)} result={result} "
          f"(expect [7], None-голос игнорируется) {'OK' if good else 'FAIL'}")

    return ok


def test_c_budget(mod) -> bool:
    print("--- (C) budget_ms делится на K на время обёртки, восстанавливается после ---")
    _clear_env()
    os.environ["SCOUT_R2_ROOTPAR"] = "3"
    mod.SEARCH_CFG["budget_ms"] = 9000
    probe = []

    def _stub(obs):
        probe.append(mod.SEARCH_CFG.get("budget_ms"))
        return [0]
    mod.search_choice = _stub
    mod.search_choice_dispatch(FAKE_OBS)
    after = mod.SEARCH_CFG.get("budget_ms")
    seen_during = set(probe)
    good = (seen_during == {3000} and after == 9000)
    print(f"  budget_ms=9000, K=3 -> видели во время вызовов={sorted(seen_during)} "
          f"(expect [3000]), после восстановлен={after} (expect 9000) "
          f"{'OK' if good else 'FAIL'}")
    return good


def test_d_env_overrides() -> bool:
    print("--- (D) SCOUT_R2_SAMPLES/BUDGET_MS/EP_SOFT/EP_HARD оверрайдят на импорте ---")
    _clear_env()
    ref_mod = _load_module(REF_DIR, "ref_noenv")
    ref_samples = ref_mod.SEARCH_CFG.get("samples")
    ref_budget = ref_mod.SEARCH_CFG.get("budget_ms")
    ref_soft = ref_mod.EPISODE_SOFT_S
    ref_hard = ref_mod.EPISODE_HARD_S

    cand_default = _load_module(CAND_DIR, "cand_noenv")
    d_ok = (cand_default.SEARCH_CFG.get("samples") == ref_samples
            and cand_default.SEARCH_CFG.get("budget_ms") == ref_budget
            and cand_default.EPISODE_SOFT_S == ref_soft
            and cand_default.EPISODE_HARD_S == ref_hard)
    print(f"  без ENV: cand samples={cand_default.SEARCH_CFG.get('samples')} "
          f"(ref={ref_samples}) budget_ms={cand_default.SEARCH_CFG.get('budget_ms')} "
          f"(ref={ref_budget}) soft={cand_default.EPISODE_SOFT_S} (ref={ref_soft}) "
          f"hard={cand_default.EPISODE_HARD_S} (ref={ref_hard}) "
          f"{'OK' if d_ok else 'FAIL'}")

    os.environ["SCOUT_R2_SAMPLES"] = "7"
    os.environ["SCOUT_R2_BUDGET_MS"] = "1234"
    os.environ["SCOUT_R2_EP_SOFT"] = "11"
    os.environ["SCOUT_R2_EP_HARD"] = "22"
    cand_env = _load_module(CAND_DIR, "cand_env")
    e_ok = (cand_env.SEARCH_CFG.get("samples") == 7
            and cand_env.SEARCH_CFG.get("budget_ms") == 1234
            and cand_env.EPISODE_SOFT_S == 11.0
            and cand_env.EPISODE_HARD_S == 22.0)
    print(f"  с ENV: samples={cand_env.SEARCH_CFG.get('samples')} (expect 7) "
          f"budget_ms={cand_env.SEARCH_CFG.get('budget_ms')} (expect 1234) "
          f"soft={cand_env.EPISODE_SOFT_S} (expect 11.0) hard={cand_env.EPISODE_HARD_S} "
          f"(expect 22.0) {'OK' if e_ok else 'FAIL'}")

    # эталон в том же процессе не должен был увидеть эти ENV вообще (он их не читает)
    ref_mod2 = _load_module(REF_DIR, "ref_after_env")
    ref_isolated = (ref_mod2.SEARCH_CFG.get("samples") == ref_samples
                    and ref_mod2.SEARCH_CFG.get("budget_ms") == ref_budget)
    print(f"  эталон после тех же ENV: samples={ref_mod2.SEARCH_CFG.get('samples')} "
          f"budget_ms={ref_mod2.SEARCH_CFG.get('budget_ms')} (не должны были измениться) "
          f"{'OK' if ref_isolated else 'FAIL'}")
    _clear_env()
    return d_ok and e_ok and ref_isolated


def main() -> int:
    _clear_env()
    mod = _load_module(CAND_DIR, "main")
    ok_a = test_a_disabled(mod)
    print()
    ok_b = test_b_voting(mod)
    print()
    ok_c = test_c_budget(mod)
    print()
    ok_d = test_d_env_overrides()
    print()
    all_ok = ok_a and ok_b and ok_c and ok_d
    print(f"ИТОГ: A(disabled)={'PASS' if ok_a else 'FAIL'} "
          f"B(voting)={'PASS' if ok_b else 'FAIL'} "
          f"C(budget)={'PASS' if ok_c else 'FAIL'} "
          f"D(env-overrides)={'PASS' if ok_d else 'FAIL'}")
    print("PASS ВСЕГО" if all_ok else "FAIL — см. выше")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
