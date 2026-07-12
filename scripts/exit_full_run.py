"""
FULL ExIt RL RUN — iterative orchestrator.

Builds on the GO/NO-GO probe (scratchpad/gold/rl_govono/step1..step3*.py,
which showed 1 ExIt iteration lifting H2H vs v009a_crustle from 13.3%
(frozen BC prior) to 36.7% without non-mirror collapse). This script drives
FURTHER iterations starting from that iter1 checkpoint
(scratchpad/gold/rl_govono/trained_model.pth), each iteration:

  1. SELF-PLAY collection: current model vs a MIXED regime
       - ~40% mirror (self-play, both sides = current model, MCTS(10)+NN)
       - ~60% vs an EXPANDED, FIXED 6-archetype league (3 real submission
         agents: v009a_crustle/v017_alakazam/v018_grimmsnarl, + 3 frozen-NN
         "greedy" meta-deck opponents: FestivalLead/MegaStarmie/Zoroark)
     split roughly evenly across the league (non-mirror-collapse guard —
     project's known failure mode, PLAN.md search-pivot v3 postmortem).
     Memory-leak-safe: mp.Pool respawned every <=40 games/worker.

  2. TRAIN: 2-3 epochs, value+policy loss, KL-anchor (0.05) to the FROZEN
     BC prior (scratchpad/gold/cardemb_out_v2/out/model_best.pth) — NEVER to
     the previous iteration's weights, to prevent unbounded self-play drift.

  3. EVAL: H2H vs v009a_crustle (MAIN gate metric) + v017_alakazam +
     v018_grimmsnarl, N per matchup from --n-eval (default 90), seat-swap,
     Wilson 95% CI. Plus a cheaper non-mirror-health check vs the 3 frozen
     greedy meta-deck opponents (smaller N).

  4. CHECKPOINT: model_iter{N}.pth + a metrics.json snapshot under
     scratchpad/gold/rl_full/iter{N}/. Every iteration's results are also
     appended to scratchpad/gold/rl_full/trend.json so progress survives a
     session interruption.

STOPPING RULES (checked after each iteration, using H2H vs v009a_crustle):
  - SUCCESS: winrate > 0.52 AND wilson_lo > 0.48 (N >= 80)  -> beat champion.
  - PLATEAU: 2 consecutive iterations with |delta winrate| <= 0.03 AND
             winrate < 0.50 -> stop, report honestly.
  - MAX_ITERS reached -> stop with current best.

Usage:
  .venv/bin/python3 scripts/exit_full_run.py \
      --max-iters 4 --n-collect 1000 --n-eval 90 --workers 8 \
      --start-model scratchpad/gold/rl_govono/trained_model.pth
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

REPO = "/Users/max/kaggle-pokemon-tcg-ai-battle"
LIB_DIR = os.path.join(REPO, "scratchpad", "gold", "rl_govono")
sys.path.insert(0, LIB_DIR)

import rl_full_lib as L  # noqa: E402
from mcts_rl_lib import FROZEN_MODEL_PATH  # noqa: E402

OUT_ROOT = os.path.join(REPO, "scratchpad", "gold", "rl_full")
TREND_PATH = os.path.join(OUT_ROOT, "trend.json")

SUCCESS_WR = 0.52
SUCCESS_WILSON_LO = 0.48
PLATEAU_DELTA = 0.03
PLATEAU_PATIENCE = 2

# Known baseline trend from GO/NO-GO (not re-run here; carried forward so the
# final report shows the full 13% -> 37% -> ... arc).
KNOWN_TREND = [
    {"iter": 0, "label": "frozen BC prior", "model": FROZEN_MODEL_PATH,
     "vs_v009a_crustle": {"winrate": 0.1333, "wilson_lo": 0.0531, "wilson_hi": 0.2968, "n": 30}},
    {"iter": 1, "label": "GO/NO-GO (320 self-play games)",
     "model": os.path.join(LIB_DIR, "trained_model.pth"),
     "vs_v009a_crustle": {"winrate": 0.3667, "wilson_lo": 0.2187, "wilson_hi": 0.5449, "n": 30},
     "vs_v018_grimmsnarl": {"winrate": 0.9667, "wilson_lo": 0.8333, "wilson_hi": 0.9941, "n": 30}},
]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def collect_iteration(iter_num: int, current_model_path: str, n_total: int,
                       workers: int, out_dir: str, mirror_frac: float = 0.4):
    n_mirror = int(round(n_total * mirror_frac))
    n_league_total = n_total - n_mirror
    n_per_opp = max(1, n_league_total // len(L.LEAGUE))

    stats = {}
    all_samples = []

    log(f"  [collect] mirror: {n_mirror} games")
    t0 = time.perf_counter()
    samples, n_ok = L.run_chunked(
        L._mirror_worker, n_mirror, workers,
        extra_args=(current_model_path, L.OUR_DECK_PATH),
        seed_salt=iter_num * 1000 + 1,
    )
    dt = time.perf_counter() - t0
    all_samples.extend(samples)
    stats["mirror"] = {"n_games_ok": n_ok, "n_samples": len(samples), "wall_s": dt}
    log(f"    n_ok={n_ok} n_samples={len(samples)} wall={dt:.1f}s "
        f"({n_ok/dt if dt>0 else 0:.2f} games/s)")

    for opp_idx, (name, kind, main_py, deck_csv) in enumerate(L.LEAGUE):
        log(f"  [collect] vs {name} ({kind}): {n_per_opp} games")
        t0 = time.perf_counter()
        extra = (current_model_path, L.OUR_DECK_PATH, kind, main_py, deck_csv, FROZEN_MODEL_PATH)
        samples, n_ok = L.run_chunked(
            L._league_worker, n_per_opp, workers, extra_args=extra,
            seed_salt=iter_num * 1000 + 100 + opp_idx * 10,
        )
        dt = time.perf_counter() - t0
        all_samples.extend(samples)
        stats[name] = {"n_games_ok": n_ok, "n_samples": len(samples), "wall_s": dt}
        log(f"    n_ok={n_ok} n_samples={len(samples)} wall={dt:.1f}s "
            f"({n_ok/dt if dt>0 else 0:.2f} games/s)")

    pkl_path = os.path.join(out_dir, "selfplay_samples.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(all_samples, f)
    stats["total_samples"] = len(all_samples)
    stats["pkl_path"] = pkl_path
    return all_samples, stats


def train_iteration(samples: list, init_model_path: str, out_model_path: str, epochs: int = 2):
    log(f"  [train] {len(samples)} samples, epochs={epochs}, init={os.path.basename(init_model_path)}")
    t0 = time.perf_counter()
    history = L.train_on_samples(samples, init_model_path=init_model_path,
                                  out_model_path=out_model_path, epochs=epochs)
    dt = time.perf_counter() - t0
    log(f"    done in {dt:.1f}s -> {out_model_path}")
    return history, dt


def eval_iteration(model_path: str, n_eval: int, n_health: int, workers: int, seed_salt: int):
    results = {}
    log(f"  [eval] main gate matchups, N={n_eval} each: {L.EVAL_OPPONENTS}")
    for opp in L.EVAL_OPPONENTS:
        r = L.eval_h2h(model_path, opp, n=n_eval, workers=workers, seed_salt=seed_salt)
        results[opp] = r
        log(f"    vs {opp:20s} WR={r['winrate']:.3f} wilson=[{r['wilson_lo']:.3f},{r['wilson_hi']:.3f}] "
            f"({r['wins']}-{r['losses']}-{r['draws']}, n={r['n']}, {r['wall_s']:.1f}s)")

    log(f"  [eval] non-mirror health, N={n_health} each: {L.HEALTH_OPPONENTS}")
    for opp in L.HEALTH_OPPONENTS:
        r = L.eval_h2h(model_path, opp, n=n_health, workers=workers, seed_salt=seed_salt + 500)
        results[opp] = r
        log(f"    vs {opp:20s} WR={r['winrate']:.3f} wilson=[{r['wilson_lo']:.3f},{r['wilson_hi']:.3f}] "
            f"({r['wins']}-{r['losses']}-{r['draws']}, n={r['n']}, {r['wall_s']:.1f}s)")

    return results


def append_trend(entry: dict):
    os.makedirs(OUT_ROOT, exist_ok=True)
    trend = []
    if os.path.exists(TREND_PATH):
        with open(TREND_PATH) as f:
            trend = json.load(f)
    trend.append(entry)
    with open(TREND_PATH, "w") as f:
        json.dump(trend, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=4, help="max NEW iterations to run this session")
    ap.add_argument("--start-iter-num", type=int, default=2, help="absolute iteration number to label the first new iteration")
    ap.add_argument("--n-collect", type=int, default=1000, help="self-play games per iteration")
    ap.add_argument("--n-eval", type=int, default=90, help="games per main gate matchup (v009a/v017/v018)")
    ap.add_argument("--n-health", type=int, default=40, help="games per non-mirror-health matchup (meta greedy)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2, help="training epochs per iteration")
    ap.add_argument("--start-model", default=os.path.join(LIB_DIR, "trained_model.pth"))
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    if not os.path.exists(TREND_PATH):
        with open(TREND_PATH, "w") as f:
            json.dump(KNOWN_TREND, f, indent=2)

    current_model = args.start_model
    # best-so-far vs v009a (from KNOWN_TREND iter1)
    best_wr_v009a = 0.3667
    prev_wr_v009a = 0.3667
    plateau_count = 0
    verdict = "MAX_ITERS"
    best_model_path = current_model
    best_model_result = None

    log("=" * 70)
    log("FULL ExIt RL RUN")
    log(f"  start_model   = {current_model}")
    log(f"  max_iters     = {args.max_iters} (labeled iter{args.start_iter_num}..iter{args.start_iter_num+args.max_iters-1})")
    log(f"  n_collect     = {args.n_collect} self-play games/iter")
    log(f"  n_eval        = {args.n_eval}/matchup (main gate: {L.EVAL_OPPONENTS})")
    log(f"  n_health      = {args.n_health}/matchup (health: {L.HEALTH_OPPONENTS})")
    log(f"  workers       = {args.workers}")
    log(f"  known trend so far: iter0=13.3%, iter1=36.7% vs v009a_crustle")
    log("=" * 70)

    for i in range(args.max_iters):
        iter_num = args.start_iter_num + i
        iter_dir = os.path.join(OUT_ROOT, f"iter{iter_num}")
        os.makedirs(iter_dir, exist_ok=True)
        t_iter0 = time.perf_counter()
        log(f"\n{'#'*70}\n# ITERATION {iter_num} (session iter {i+1}/{args.max_iters})\n{'#'*70}")

        # 1) collect
        samples, collect_stats = collect_iteration(
            iter_num, current_model, args.n_collect, args.workers, iter_dir,
        )

        # 2) train
        out_model_path = os.path.join(iter_dir, f"model_iter{iter_num}.pth")
        history, train_wall = train_iteration(samples, current_model, out_model_path, epochs=args.epochs)

        # free samples memory + delete the (large) pickle to save disk once trained
        del samples
        try:
            os.remove(collect_stats["pkl_path"])
            collect_stats["pkl_path"] = "(deleted after training)"
        except Exception:
            pass

        # 3) eval
        eval_results = eval_iteration(out_model_path, args.n_eval, args.n_health, args.workers, seed_salt=iter_num)

        t_iter_total = time.perf_counter() - t_iter0

        # 4) checkpoint + trend log
        wr_v009a = eval_results["v009a_crustle"]["winrate"]
        wilson_lo_v009a = eval_results["v009a_crustle"]["wilson_lo"]

        entry = {
            "iter": iter_num,
            "label": f"exit_full_run session iter {i+1}",
            "model": out_model_path,
            "collect_stats": collect_stats,
            "train_history": history,
            "eval_results": eval_results,
            "wall_s_total": t_iter_total,
        }
        with open(os.path.join(iter_dir, "metrics.json"), "w") as f:
            json.dump(entry, f, indent=2)
        append_trend(entry)

        log(f"\n[iter{iter_num}] DONE in {t_iter_total:.0f}s. "
            f"vs v009a_crustle: WR={wr_v009a:.3f} wilson_lo={wilson_lo_v009a:.3f} "
            f"(prev={prev_wr_v009a:.3f})")

        if wr_v009a > best_wr_v009a:
            best_wr_v009a = wr_v009a
            best_model_path = out_model_path
            best_model_result = eval_results["v009a_crustle"]

        # ---- stopping rules ----
        if wr_v009a > SUCCESS_WR and wilson_lo_v009a > SUCCESS_WILSON_LO:
            log(f"[STOP] SUCCESS: WR={wr_v009a:.3f} > {SUCCESS_WR}, "
                f"wilson_lo={wilson_lo_v009a:.3f} > {SUCCESS_WILSON_LO}. BEAT CHAMPION.")
            verdict = "BEAT_CHAMPION"
            current_model = out_model_path
            break

        delta = wr_v009a - prev_wr_v009a
        if abs(delta) <= PLATEAU_DELTA and wr_v009a < 0.50:
            plateau_count += 1
            log(f"[plateau-watch] delta={delta:+.3f} <= {PLATEAU_DELTA}, WR<0.50 "
                f"-> plateau_count={plateau_count}/{PLATEAU_PATIENCE}")
        else:
            plateau_count = 0

        if plateau_count >= PLATEAU_PATIENCE:
            log(f"[STOP] PLATEAU: {plateau_count} consecutive iterations without "
                f"meaningful improvement, still below champion (WR={wr_v009a:.3f} < 0.50).")
            verdict = "PLATEAU"
            current_model = out_model_path
            break

        prev_wr_v009a = wr_v009a
        current_model = out_model_path  # carry forward to next iteration regardless

    else:
        log(f"[STOP] MAX_ITERS reached ({args.max_iters} new iterations).")

    # Final honest classification from the observed trend (overrides the loop's
    # break-reason label -- e.g. MAX_ITERS reached but best_wr already > 0.50).
    if verdict != "BEAT_CHAMPION":
        if best_wr_v009a > SUCCESS_WR and best_model_result and best_model_result.get("wilson_lo", 0) > SUCCESS_WILSON_LO:
            verdict = "BEAT_CHAMPION"
        elif best_wr_v009a < 0.20:
            verdict = "DIVERGED"
        elif best_wr_v009a > 0.3667:
            verdict = "IMPROVED_NOT_ENOUGH"
        else:
            verdict = "PLATEAU"

    summary = {
        "verdict": verdict,
        "best_model_path": best_model_path,
        "best_wr_v009a": best_wr_v009a,
        "best_model_result": best_model_result,
        "final_model_path": current_model,
    }
    with open(os.path.join(OUT_ROOT, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("\n" + "=" * 70)
    log(f"FINAL VERDICT: {verdict}")
    log(f"BEST MODEL: {best_model_path} (WR vs v009a_crustle = {best_wr_v009a:.3f})")
    log("=" * 70)


if __name__ == "__main__":
    main()
