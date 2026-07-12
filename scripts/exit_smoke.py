#!/usr/bin/env python3
"""exit_smoke.py — первый-когда-либо smoke src/exit/exit_loop.py (PLAN.md F3-3).

Каркас (WP1-WP5, написан 01 июля) ни разу не прогонялся. Он был написан в
PIMC/Iono-эру и жёстко хардкодит feature_module="features_iono.py" в местах
экспорта (см. фикс в exit_loop.py: параметр feature_module теперь
прокидывается через run_iteration/run_exit_loop). Этот скрипт адаптирует
smoke под ТЕКУЩЕГО чемпиона — Crustle IL (v009a_crustle) — вместо Iono, и
запускает ОДНУ итерацию ExIt на маленьком объёме партий (не production).

Движок: cg_native (macOS arm64) — harness.py пропатчен алиасить 'cg' на
cg_native внутри worker-процессов (см. src/exit/harness.py:_worker_fn).

PIMC-эксперт (Шаг 1 run_iteration) — АРХИТЕКТУРНО обязателен в текущем
каркасе (Step 1 всегда оборачивает pilot в PIMCAgent), даже для
filter_mode="win". Поскольку PIMC официально закрыт (PLAN.md §H3-4-re KILL,
проигрыш random = сигнатура бага), здесь bank урезан до долей секунды через
PIMC_BANK_* env — это НЕ полноценный PIMC-поиск, а способ быстро прогнать
пайплайн и почти всегда падать на дешёвый prior_agent (= pilot) путь.

Запуск:
    .venv/bin/python scripts/exit_smoke.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# --- Урезаем PIMC bank до smoke-размера ДО любого импорта exit.pimc_agent
#     (константы BANK_*_DEFAULT читаются из os.environ на уровне модуля,
#     внутри worker-процессов os.environ наследуется от родителя при spawn).
os.environ.setdefault("PIMC_BANK_TOTAL_SEC", "6.0")
os.environ.setdefault("PIMC_BANK_SAFETY", "0.6")
os.environ.setdefault("PIMC_EST_TOTAL_DECISIONS", "30")
os.environ.setdefault("PIMC_MIN_DECISION_BUDGET", "0.05")
os.environ.setdefault("PIMC_MAX_DECISION_BUDGET", "0.4")

SCRATCH = (
    "/private/tmp/claude-501/-Users-max-kaggle-pokemon-tcg-ai-battle/"
    "0c183027-29df-4955-b9de-46be4295c2b6/scratchpad/exit_smoke"
)
os.makedirs(SCRATCH, exist_ok=True)


def main() -> None:
    from exit.opponents import list_opponents, _load_deck_csv
    from exit.pilot import Pilot
    from exit.exit_loop import run_iteration, IterationResult

    # --- Crustle deck/model/features (v009a — жив, μ≈946, наш чемпион) ------
    v009a_dir = os.path.join(ROOT, "submissions", "v009a_crustle")
    deck = _load_deck_csv(os.path.join(v009a_dir, "deck.csv"))
    assert len(deck) == 60

    feat_path = os.path.join(ROOT, "src", "il", "features_v009a.py")
    spec = importlib.util.spec_from_file_location("features_v009a", feat_path)
    feat_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feat_mod)
    build_feature_matrix = feat_mod.build_feature_matrix
    print(f"[smoke] features_v009a FEATURE_DIM={feat_mod.FEATURE_DIM}")

    model_path = os.path.join(ROOT, "models", "il_crustle_v009a.txt")
    assert os.path.exists(model_path), model_path
    pilot = Pilot(feature_fn=build_feature_matrix, deck=deck)
    pilot.load(model_path)
    print(f"[smoke] Загружен pilot: deck={len(pilot.deck)} карт, model={model_path}")

    # --- Малый замороженный пул ------------------------------------------
    pool = list_opponents(names=["random", "lucario_heur", "crustle_il"], skip_errors=True)
    print(f"[smoke] Пул: {[(a.name, a.mu) for a in pool]}")
    if len(pool) < 2:
        print("ОШИБКА: пул слишком мал (ожидалось >=2 loaded)", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.join(SCRATCH, "models_smoke")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    new_pilot, result = run_iteration(
        pilot=pilot,
        pool=pool,
        iteration=0,
        games_per_opp=4,       # МАЛО — smoke, не production (нужно >=300 для гейта)
        workers=2,
        pimc_budget=0.3,        # ceiling; реальный per-decision budget рулится PIMC_BANK_* env
        min_reward=1.0,
        out_dir=out_dir,
        filter_mode="win",
        feature_module="features_v009a.py",
    )
    dt = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print(f"[smoke] run_iteration() завершилась за {dt:.1f}s")
    print("=" * 70)
    result_dict = {
        "iteration": result.iteration,
        "winrate_before": result.winrate_before,
        "winrate_after": result.winrate,
        "wilson_lo": result.wilson_lo,
        "n_trajectories": result.n_trajectories,
        "promoted": result.promoted,
        "pilot_path": result.pilot_path,
        "per_opponent": result.per_opponent,
        "train_metrics": result.train_metrics,
        "wall_time_s": round(dt, 1),
        "new_pilot_is_same_object_as_input": new_pilot is pilot,
    }
    print(json.dumps(result_dict, ensure_ascii=False, indent=2))

    with open(os.path.join(SCRATCH, "smoke_result.json"), "w") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    print(f"\n[smoke] Результат сохранён: {os.path.join(SCRATCH, 'smoke_result.json')}")


if __name__ == "__main__":
    main()
