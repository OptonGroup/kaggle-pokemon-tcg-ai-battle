#!/usr/bin/env python3
"""live_replay_check_v089.py — быстрая предсабмитная проверка агента на ЗАПИСАННЫХ решениях.

ЗАЧЕМ НЕ ПОЛНЫЕ ИГРЫ: локальный Docker гоняет x86-образ под Rosetta-эмуляцией; 6 партий
двух BC-агентов сожгли час CPU и не доиграли. Здесь тот же код агента (main.py сабмита)
прогоняется по РЕАЛЬНЫМ obs из корпуса — это ровно тот путь, что исполняется в проде на
каждом решении, но без движкового цикла.

Что проверяется (то, что реально ломает live — см. project_kaggle_live_safety):
  1. агент не кидает исключений (в проде исключение = greedy-fallback или поражение);
  2. КАЖДЫЙ ответ легален: индексы в [0, len(option)), длина в [minCount, maxCount],
     без дубликатов (нелегальный индекс = мгновенное поражение);
  3. время на решение. ВАЖНО: замер идёт под эмуляцией, поэтому абсолютные секунды
     ЗАВЫШЕНЫ; сравниваем кандидата с УЖЕ РАБОТАЮЩИМ в проде v088 на тех же решениях —
     значим именно КОЭФФИЦИЕНТ (v089 крупнее: 1129 деревьев в main против 732);
  4. первый вызов (select is None) возвращает ровно 60 id деки.

Пример:
  python scripts/live_replay_check_v089.py --agent-dir submissions/v089_bc2b \
      --ref-dir submissions/v088_bc2b --n 150
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_agent(sub_dir: Path, tag: str):
    sys.path.insert(0, str(sub_dir))
    try:
        spec = importlib.util.spec_from_file_location(f"_lrc_{tag}", str(sub_dir / "main.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"_lrc_{tag}"] = mod
        spec.loader.exec_module(mod)
        return mod.agent
    finally:
        sys.path.remove(str(sub_dir))


def check_legal(select, obs: dict, where: str) -> str | None:
    sel = obs.get("select") or {}
    options = sel.get("option") or []
    lo, hi = sel.get("minCount", 1), sel.get("maxCount", 1)
    if not isinstance(select, list):
        return f"{where}: тип {type(select).__name__}, ожидался list"
    if len(select) != len(set(select)):
        return f"{where}: дубликаты {select}"
    if not (lo <= len(select) <= hi):
        return f"{where}: длина {len(select)} вне [{lo},{hi}]"
    for i in select:
        if not isinstance(i, int) or not (0 <= i < len(options)):
            return f"{where}: индекс {i} вне [0,{len(options)})"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-dir", required=True)
    ap.add_argument("--ref-dir", default=None, help="эталон (уже в проде) для сравнения времени")
    ap.add_argument("--obs-dir",
                    default="out/corpus_2b/versions/v088_dragapult/merged/obs_shards")
    ap.add_argument("--decisions",
                    default="out/corpus_2b/versions/v088_dragapult/merged/decisions.jsonl")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    a_dir = REPO / args.agent_dir
    deck = [int(x) for x in (a_dir / "deck.csv").read_text().split() if x.strip()]
    print(f"[replay] deck.csv кандидата: {len(deck)} карт")
    if len(deck) != 60:
        print("[replay] ПРОВАЛ: дека не 60 карт")
        return 1

    # набираем решения из РАЗНЫХ эпизодов
    rng = random.Random(42)
    eps: dict[int, list[int]] = {}
    with (REPO / args.decisions).open() as fh:
        for line in fh:
            d = json.loads(line)
            eps.setdefault(d["episode_id"], []).append(d["obs_ref"])
    ep_ids = rng.sample(sorted(eps), min(60, len(eps)))
    picks = []
    for ep in ep_ids:
        for ref in rng.sample(eps[ep], min(4, len(eps[ep]))):
            picks.append((ep, ref))
    rng.shuffle(picks)
    picks = picks[: args.n]

    obs_root = REPO / args.obs_dir
    obs_list = []
    by_ep: dict[int, list[int]] = {}
    for ep, ref in picks:
        by_ep.setdefault(ep, []).append(ref)
    for ep, refs in by_ep.items():
        shard = obs_root / f"{ep}.jsonl"
        if not shard.exists():
            continue
        need = set(refs)
        with shard.open() as fh:
            for i, line in enumerate(fh):
                if i in need:
                    obs_list.append((f"{ep}:{i}", json.loads(line)))
                    need.discard(i)
                    if not need:
                        break
    print(f"[replay] решений собрано: {len(obs_list)}")

    rep = {"n": len(obs_list)}
    for label, sub in [("v089", a_dir)] + ([("ref", REPO / args.ref_dir)] if args.ref_dir else []):
        agent = load_agent(sub, label)
        # первый вызов: движок просит деку (select is None)
        t0 = time.perf_counter()
        deck_ans = agent({"select": None})
        t_deck = time.perf_counter() - t0
        deck_ok = isinstance(deck_ans, list) and len(deck_ans) == 60
        errs, times = [], []
        for uid, obs in obs_list:
            try:
                t0 = time.perf_counter()
                ans = agent(obs)
                times.append(time.perf_counter() - t0)
            except Exception as e:  # noqa: BLE001
                errs.append(f"{uid}: {type(e).__name__}: {e}")
                continue
            bad = check_legal(ans, obs, uid)
            if bad:
                errs.append(bad)
        rep[label] = {"deck_call_ok": deck_ok, "deck_call_s": round(t_deck, 3),
                      "n_errors": len(errs), "errors": errs[:5],
                      "mean_s": round(statistics.mean(times), 4) if times else None,
                      "p95_s": round(sorted(times)[int(len(times) * 0.95)], 4) if times else None,
                      "max_s": round(max(times), 4) if times else None}
        print(f"[replay] {label}: дека_ок={deck_ok} ошибок={len(errs)} "
              f"среднее={rep[label]['mean_s']}с p95={rep[label]['p95_s']}с "
              f"макс={rep[label]['max_s']}с")
        for e in errs[:5]:
            print(f"    ! {e}")

    ok = rep["v089"]["n_errors"] == 0 and rep["v089"]["deck_call_ok"]
    if args.ref_dir and rep.get("ref", {}).get("mean_s"):
        ratio = rep["v089"]["mean_s"] / rep["ref"]["mean_s"]
        rep["time_ratio_vs_ref"] = round(ratio, 2)
        # live-бюджет: 600с на ЭПИЗОД (~150-250 решений). Оцениваем по эмулированному
        # времени кандидата, приведённому к эталону, который в проде укладывается.
        print(f"[replay] время кандидата / эталона: ×{ratio:.2f} "
              f"(эталон v088 в проде укладывается в 600с/эпизод)")
        if ratio > 3.0:
            print("[replay] ВНИМАНИЕ: кандидат более чем втрое медленнее прод-версии")
            ok = False
    rep["verdict"] = "OK" if ok else "FAIL"
    print("\nИТОГ:", "OK — исключений нет, все ответы легальны, время сопоставимо" if ok
          else "ПРОВАЛ — сабмитить нельзя")
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
