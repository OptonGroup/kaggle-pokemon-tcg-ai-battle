"""
bench_engine.py — замер скорости симуляции cabt движка.
Запускать внутри Docker (linux/amd64).

Usage:
    python scripts/bench_engine.py
"""
import sys, os, time, random, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_observation_class, search_begin, search_step, search_end

DECK_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'submissions', 'v001_random', 'deck.csv')
with open(DECK_CSV) as f:
    DECK = [int(x.strip()) for x in f if x.strip()]
assert len(DECK) == 60, f"deck len={len(DECK)}"


def random_action(obs_dict):
    """Случайное легальное действие. Если select is None — возвращаем колоду."""
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return DECK[:]
    opts = obs.select.option
    lo, hi = obs.select.minCount, obs.select.maxCount
    k = random.randint(lo, hi) if lo < hi else lo
    k = min(k, len(opts))
    k = max(k, lo)
    return random.sample(range(len(opts)), k)


# ── ШАГ A: полная партия random vs random ───────────────────────────────────
def run_one_game(seed=None):
    if seed is not None:
        random.seed(seed)
    obs_dict, start = battle_start(DECK[:], DECK[:])
    if obs_dict is None:
        raise RuntimeError(f"battle_start failed: {start.errorType}")
    steps = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            if obs.current and obs.current.result != -1:
                return steps, obs.current.result
            sel = random_action(obs_dict)
            obs_dict = battle_select(sel)
            steps += 1
    finally:
        battle_finish()


print("=== ШАГ A: random vs random full game ===")
N_GAMES = 20
step_counts = []
t0 = time.perf_counter()
for i in range(N_GAMES):
    steps, winner = run_one_game(seed=i)
    step_counts.append(steps)
    print(f"  game {i:2d}: steps={steps}, winner={winner}", flush=True)
t_total = time.perf_counter() - t0
total_steps = sum(step_counts)
print(f"\n  Итого {N_GAMES} партий за {t_total:.2f}s")
print(f"  battle_select шагов/сек : {total_steps/t_total:.0f}")
print(f"  партий/сек              : {N_GAMES/t_total:.2f}")
print(f"  медиана шагов/партию    : {statistics.median(step_counts):.0f}")
print(f"  мин/макс шагов          : {min(step_counts)} / {max(step_counts)}")
print(f"  среднее шагов           : {statistics.mean(step_counts):.1f}")
median_steps = statistics.median(step_counts)
print()


# ── ШАГ B: search_step скорость (rollout до конца) ──────────────────────────
print("=== ШАГ B: search_step rollout ===")

def advance_to_main_select(seed):
    """Запустить партию и пройти начальные deck-select, вернуть obs_dict на первом реальном выборе."""
    random.seed(seed)
    obs_dict, start = battle_start(DECK[:], DECK[:])
    if obs_dict is None:
        raise RuntimeError("battle_start failed")
    # Пройдём начальные выборы колоды (select is None) и мультиганы
    for _ in range(20):
        obs = to_observation_class(obs_dict)
        if obs.current and obs.current.result != -1:
            return None, obs_dict  # партия уже кончилась (маловероятно)
        if obs.select is not None:
            return obs, obs_dict
        obs_dict = battle_select(DECK[:])
    return None, obs_dict


def do_search_rollout(initial_obs, initial_obs_dict, max_steps=3000):
    """Один search rollout до конца партии."""
    obs = initial_obs
    your_index = obs.current.yourIndex
    opp_index = 1 - your_index

    n_opp_deck  = obs.current.players[opp_index].deckCount
    n_opp_prize = len(obs.current.players[opp_index].prize)
    n_opp_hand  = obs.current.players[opp_index].handCount
    n_your_prize= len(obs.current.players[your_index].prize)
    n_your_deck = obs.current.players[your_index].deckCount

    opp_deck_guess  = (DECK * 4)[:n_opp_deck]
    opp_prize_guess = (DECK * 4)[:n_opp_prize]
    opp_hand_guess  = (DECK * 4)[:n_opp_hand]
    your_prize_guess= (DECK * 4)[:n_your_prize]
    your_deck_guess = (DECK * 4)[:n_your_deck]

    state = search_begin(
        obs,
        your_deck=your_deck_guess,
        your_prize=your_prize_guess,
        opponent_deck=opp_deck_guess,
        opponent_prize=opp_prize_guess,
        opponent_hand=opp_hand_guess,
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
            return steps, -2  # обрыв по лимиту

        opts = s_obs.select.option
        lo, hi = s_obs.select.minCount, s_obs.select.maxCount
        k = random.randint(lo, hi) if lo < hi else lo
        k = min(k, len(opts))
        k = max(k, lo)
        chosen = random.sample(range(len(opts)), k)

        state = search_step(state.searchId, chosen)
        steps += 1


N_ROLLOUTS = 10
rollout_steps_list = []
rollout_times_list = []

for i in range(N_ROLLOUTS):
    obs, obs_dict = advance_to_main_select(seed=200+i)
    if obs is None:
        print(f"  rollout {i}: не дошли до выбора, skip")
        battle_finish()
        continue
    t1 = time.perf_counter()
    try:
        steps, winner = do_search_rollout(obs, obs_dict)
    except Exception as e:
        print(f"  rollout {i}: ERROR {e}")
        battle_finish()
        continue
    t2 = time.perf_counter()
    battle_finish()
    rollout_steps_list.append(steps)
    rollout_times_list.append(t2 - t1)
    print(f"  rollout {i:2d}: search_steps={steps}, winner={winner}, time={t2-t1:.3f}s", flush=True)

if rollout_steps_list:
    mean_t  = statistics.mean(rollout_times_list)
    total_s = sum(rollout_steps_list)
    total_t = sum(rollout_times_list)
    budget  = 600.0 / median_steps  # 10 мин / медиана шагов партии
    print(f"\n  search_step шагов/сек   : {total_s/total_t:.0f}")
    print(f"  rollout/сек             : {len(rollout_times_list)/total_t:.3f}")
    print(f"  среднее время rollout   : {mean_t:.2f}s")
    print(f"  rollout за 1s           : {1/mean_t:.2f}")
    print(f"  rollout за 60s          : {60/mean_t:.1f}")
    print(f"  бюджет на 1 ход (10min/медиана_шагов) : {budget:.2f}s")
    print(f"  rollout в бюджет на ход : {budget/mean_t:.1f}")
print()


# ── ШАГ D: thread-safety ─────────────────────────────────────────────────────
print("=== ШАГ D: threading (2 параллельных search rollout) ===")
import threading

td_results = {}
td_errors = {}

def thread_fn(idx, seed):
    try:
        obs, obs_dict = advance_to_main_select(seed)
        if obs is None:
            td_results[idx] = "no initial obs"
            battle_finish()
            return
        steps, winner = do_search_rollout(obs, obs_dict, max_steps=200)
        battle_finish()
        td_results[idx] = f"ok steps={steps} winner={winner}"
    except Exception as e:
        td_errors[idx] = str(e)
        try: battle_finish()
        except: pass

threads = [threading.Thread(target=thread_fn, args=(i, 300+i*7)) for i in range(2)]
for t in threads: t.start()
for t in threads: t.join()
for i in range(2):
    if i in td_errors:
        print(f"  thread {i}: ERROR: {td_errors[i]}")
    else:
        print(f"  thread {i}: {td_results.get(i, 'no result')}")
print()


# ── ШАГ E: детерминизм ──────────────────────────────────────────────────────
print("=== ШАГ E: детерминизм ===")
def run_fixed(seed):
    obs, obs_dict = advance_to_main_select(seed=seed)
    if obs is None:
        battle_finish()
        return None
    steps, winner = do_search_rollout(obs, obs_dict, max_steps=300)
    battle_finish()
    return (steps, winner)

r1 = run_fixed(777)
r2 = run_fixed(777)
print(f"  rollout1: {r1}")
print(f"  rollout2: {r2}")
print(f"  детерминизм: {'ДА (одинаково)' if r1 == r2 else 'НЕТ (стохастика/разные результаты)'}")
print()

print("=== ГОТОВО ===")
