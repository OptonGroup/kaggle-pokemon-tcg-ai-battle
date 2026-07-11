"""
thread_check_native.py — проверка thread-safety нативного libcg.dylib.

На Linux x86_64 (libcg.so) 2 параллельных потока с search-rollout давали
SIGSEGV → допускался только multiprocessing. Проверяем, не изменилось ли это
в нативной сборке. ЗАПУСКАТЬ ТОЛЬКО через subprocess/отдельный процесс —
SIGSEGV убивает процесс целиком.

Usage:
    .venv/bin/python scripts/thread_check_native.py   # exit 0 = потоки выжили
"""
import sys, os, time, random, threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from cg_native.game import battle_start, battle_finish, battle_select
from cg_native.api import to_observation_class, search_begin, search_step, search_end

DECK_CSV = os.path.join(ROOT, 'submissions', 'v001_random', 'deck.csv')
with open(DECK_CSV) as f:
    DECK = [int(x.strip()) for x in f if x.strip()]


def random_action(obs):
    opts = obs.select.option
    lo, hi = obs.select.minCount, obs.select.maxCount
    k = random.randint(lo, hi) if lo < hi else lo
    k = max(min(k, len(opts)), lo)
    return random.sample(range(len(opts)), k)


def worker(idx, seed, out):
    try:
        rnd = random.Random(seed)
        random.seed(seed)
        obs_dict, start = battle_start(DECK[:], DECK[:])
        if obs_dict is None:
            out[idx] = "battle_start failed"
            return
        obs = to_observation_class(obs_dict)
        # дойти до первого select
        for _ in range(20):
            if obs.select is not None:
                break
            obs_dict = battle_select(DECK[:])
            obs = to_observation_class(obs_dict)
        yi = obs.current.yourIndex
        opp = 1 - yi
        state = search_begin(
            obs,
            your_deck=(DECK*4)[:obs.current.players[yi].deckCount],
            your_prize=(DECK*4)[:len(obs.current.players[yi].prize)],
            opponent_deck=(DECK*4)[:obs.current.players[opp].deckCount],
            opponent_prize=(DECK*4)[:len(obs.current.players[opp].prize)],
            opponent_hand=(DECK*4)[:obs.current.players[opp].handCount],
            opponent_active=[],
        )
        steps = 0
        while steps < 300:
            s_obs = state.observation
            if s_obs.current and s_obs.current.result != -1:
                break
            state = search_step(state.searchId, random_action(s_obs))
            steps += 1
        search_end()
        battle_finish()
        out[idx] = f"ok steps={steps}"
    except Exception as e:
        out[idx] = f"EXC {type(e).__name__}: {e}"


out = {}
threads = [threading.Thread(target=worker, args=(i, 300 + i * 7, out)) for i in range(2)]
t0 = time.perf_counter()
for t in threads: t.start()
for t in threads: t.join(timeout=25)
alive = [t.is_alive() for t in threads]
print(f"elapsed {time.perf_counter()-t0:.1f}s, threads alive after join: {alive}")
for i in range(2):
    print(f"thread {i}: {out.get(i, 'NO RESULT (hung?)')}")
ok = (not any(alive)) and all(str(out.get(i, '')).startswith('ok') for i in range(2))
print("THREADING:", "SURVIVED" if ok else "PROBLEM")
sys.exit(0 if ok else 1)
