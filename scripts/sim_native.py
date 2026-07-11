"""
sim_native.py — нативный (macOS arm64) smoke-тест + бенчмарк движка cabt.

Использует src/cg_native/ (обёртка организатора от 2026-07-01 с автодетектом
платформы: на macOS грузится libcg.dylib). НЕ трогает src/cg/ (Linux x86_64).

Usage:
    .venv/bin/python scripts/sim_native.py [n_games] [n_rollouts]

Методология идентична scripts/bench_engine.py (Docker-эмуляция), чтобы числа
были сравнимы 1:1.
"""
import sys, os, time, random, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from cg_native.game import battle_start, battle_finish, battle_select
from cg_native.api import to_observation_class, search_begin, search_step, search_end
from cg_native import sim as _sim

N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 60
N_ROLLOUTS = int(sys.argv[2]) if len(sys.argv) > 2 else 50

DECK_CSV = os.path.join(ROOT, 'submissions', 'v001_random', 'deck.csv')
with open(DECK_CSV) as f:
    DECK = [int(x.strip()) for x in f if x.strip()]
assert len(DECK) == 60, f"deck len={len(DECK)}"

print(f"native lib: {_sim.lib_path}")
print(f"python    : {sys.version.split()[0]}  platform: {sys.platform}")
print()


def random_action(obs):
    """Случайное легальное действие (obs — уже dataclass)."""
    opts = obs.select.option
    lo, hi = obs.select.minCount, obs.select.maxCount
    k = random.randint(lo, hi) if lo < hi else lo
    k = min(k, len(opts))
    k = max(k, lo)
    return random.sample(range(len(opts)), k)


# ── SMOKE + ШАГ A: полные партии random vs random ──────────────────────────
def run_one_game(seed=None):
    if seed is not None:
        random.seed(seed)
    obs_dict, start = battle_start(DECK[:], DECK[:])
    if obs_dict is None:
        raise RuntimeError(f"battle_start failed: errorType={start.errorType}")
    steps = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                return steps, obs.current.result
            sel = random_action(obs)
            obs_dict = battle_select(sel)
            steps += 1
    finally:
        battle_finish()


print("=== SMOKE: одна партия random vs random (native) ===")
steps, winner = run_one_game(seed=0)
assert winner in (0, 1, 2), f"insane result {winner}"
print(f"  OK: steps={steps}, result={winner} (0/1=победитель, 2=ничья)")
print()

print(f"=== ШАГ A: {N_GAMES} партий random vs random ===")
step_counts, results = [], []
t0 = time.perf_counter()
for i in range(N_GAMES):
    steps, winner = run_one_game(seed=i)
    step_counts.append(steps)
    results.append(winner)
t_total = time.perf_counter() - t0
total_steps = sum(step_counts)
w0 = results.count(0); w1 = results.count(1); dr = len(results) - w0 - w1
print(f"  Итого {N_GAMES} партий за {t_total:.2f}s   (P0 {w0} / P1 {w1} / draw {dr})")
print(f"  battle_select шагов/сек : {total_steps/t_total:.0f}")
print(f"  партий/сек              : {N_GAMES/t_total:.2f}")
print(f"  партий/мин              : {60*N_GAMES/t_total:.0f}")
print(f"  медиана шагов/партию    : {statistics.median(step_counts):.0f}")
print(f"  среднее шагов           : {statistics.mean(step_counts):.1f}")
print(f"  мс/ход (battle_select)  : {1000*t_total/total_steps:.4f}")
median_steps = statistics.median(step_counts)
print()


# ── ШАГ B: search_begin/search_step rollout до терминала ────────────────────
print(f"=== ШАГ B: {N_ROLLOUTS} search-rollout до терминала (fresh battle каждый — как bench_engine) ===")

def advance_to_main_select(seed):
    random.seed(seed)
    obs_dict, start = battle_start(DECK[:], DECK[:])
    if obs_dict is None:
        raise RuntimeError("battle_start failed")
    for _ in range(20):
        obs = to_observation_class(obs_dict)
        if obs.current and obs.current.result != -1:
            return None
        if obs.select is not None:
            return obs
        obs_dict = battle_select(DECK[:])
    return None


def do_search_rollout(obs, max_steps=3000):
    """Один rollout: search_begin + random search_step до конца. Возвращает (steps, result)."""
    your_index = obs.current.yourIndex
    opp = 1 - your_index
    n_opp_deck  = obs.current.players[opp].deckCount
    n_opp_prize = len(obs.current.players[opp].prize)
    n_opp_hand  = obs.current.players[opp].handCount
    n_your_prize= len(obs.current.players[your_index].prize)
    n_your_deck = obs.current.players[your_index].deckCount

    state = search_begin(
        obs,
        your_deck=(DECK*4)[:n_your_deck],
        your_prize=(DECK*4)[:n_your_prize],
        opponent_deck=(DECK*4)[:n_opp_deck],
        opponent_prize=(DECK*4)[:n_opp_prize],
        opponent_hand=(DECK*4)[:n_opp_hand],
        opponent_active=[],
    )
    steps = 0
    while True:
        s_obs = state.observation
        if s_obs.current and s_obs.current.result != -1:
            search_end()
            return steps, s_obs.current.result
        if steps >= max_steps:
            search_end()
            return steps, -2
        chosen = random_action(s_obs)
        state = search_step(state.searchId, chosen)
        steps += 1


rollout_steps, rollout_times = [], []
for i in range(N_ROLLOUTS):
    obs = advance_to_main_select(seed=200 + i)
    if obs is None:
        battle_finish()
        continue
    t1 = time.perf_counter()
    steps, res = do_search_rollout(obs)
    t2 = time.perf_counter()
    battle_finish()
    rollout_steps.append(steps)
    rollout_times.append(t2 - t1)

total_s, total_t = sum(rollout_steps), sum(rollout_times)
print(f"  rollouts: {len(rollout_times)}, суммарно {total_s} search_step за {total_t:.2f}s")
print(f"  search_step шагов/сек   : {total_s/total_t:.0f}")
print(f"  rollout/сек (1 процесс) : {len(rollout_times)/total_t:.1f}")
print(f"  среднее шагов/rollout   : {statistics.mean(rollout_steps):.1f}")
print(f"  среднее время rollout   : {1000*statistics.mean(rollout_times):.2f} ms")
budget = 600.0 / median_steps
print(f"  бюджет на 1 ход (600s/медиана шагов) : {budget:.1f}s → rollout'ов в бюджет: {budget/statistics.mean(rollout_times):.0f}")
print()


# ── ШАГ B2: PIMC-паттерн — много rollout с ОДНОГО obs (одна партия) ─────────
print("=== ШАГ B2: 100 rollout с одного корневого obs (PIMC-паттерн) ===")
obs = advance_to_main_select(seed=999)
if obs is not None:
    t1 = time.perf_counter()
    n2_steps = 0
    N2 = 100
    for _ in range(N2):
        steps, res = do_search_rollout(obs)
        n2_steps += steps
    t2 = time.perf_counter()
    battle_finish()
    print(f"  {N2} rollout за {t2-t1:.2f}s → {N2/(t2-t1):.1f} rollout/сек, {n2_steps/(t2-t1):.0f} search_step/сек")
else:
    print("  skip (не дошли до main select)")
print()


# ── ШАГ E: детерминизм ──────────────────────────────────────────────────────
print("=== ШАГ E: детерминизм (одинаковый seed → одинаковая партия?) ===")
r1 = run_one_game(seed=777)
r2 = run_one_game(seed=777)
print(f"  партия1: steps={r1[0]}, result={r1[1]}")
print(f"  партия2: steps={r2[0]}, result={r2[1]}")
print(f"  вердикт: {'ДЕТЕРМИНИРОВАНО' if r1 == r2 else 'НЕ детерминировано (RNG движка не зависит от python seed)'}")
print()
print("=== ГОТОВО ===")
