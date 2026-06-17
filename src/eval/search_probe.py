"""Q-SEARCH probe — measure the real cost of an engine search cycle.

Runs INSIDE Docker (libcg.so is Linux x86_64 only).

What it does:
  1. battle_start(lucario, lucario), play forward a few turns with the heuristic
     pilot until we reach a mid-game MAIN decision with a non-trivial select.
  2. At that obs, build a determinization. Because this is a PROBE (timing only),
     we cheat and use the TRUE decks for both sides (we know both are Lucario).
  3. Time search_begin. Then run a full rollout: loop search_step choosing via the
     heuristic policy (applied to the search-state observation) until result != -1.
     Time it; count steps.
  4. Print: search_begin time, mean search_step time, mean steps-to-terminal,
     rollouts/sec, and an estimate of how many rollouts fit a per-decision budget
     that keeps the 600s game bank safe.

Usage (inside Docker):
    python src/eval/search_probe.py [n_rollouts] [seed]
"""
import os
import sys
import time
import random
import statistics

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC_DIR)

from cg.game import battle_start, battle_finish, battle_select
from cg.api import (
    to_observation_class, search_begin, search_step, search_end,
)

# Heuristic pilot (importable copy of v002). It loads its own deck.csv.
from agents.heuristic_agent import (
    agent as heuristic_agent, decide_on_obs, my_deck as LUCARIO_DECK,
)


def _full_deck_multiset() -> list[int]:
    """The 60-card Lucario decklist (both players use it in this probe)."""
    return list(LUCARIO_DECK)


def _visible_our_cards(obs) -> list[int]:
    """All card IDs currently visible to us (our zones). Used to compute the
    hidden pool = full_deck - visible. For the probe we instead just feed the
    TRUE deckCount/prize-count split, so we only need counts here, but we keep
    the function for parity with the real agent's determinizer."""
    st = obs.current
    me = st.players[st.yourIndex]
    seen = []
    # hand
    if me.hand:
        for c in me.hand:
            seen.append(c.id)
    # active + bench (+ attached energy/tools/pre-evos)
    for pk in list(me.active) + list(me.bench):
        if pk is None:
            continue
        seen.append(pk.id)
        for c in pk.energyCards:
            seen.append(c.id)
        for c in pk.tools:
            seen.append(c.id)
        for c in pk.preEvolution:
            seen.append(c.id)
    # discard
    for c in me.discard:
        seen.append(c.id)
    # face-up prizes (None = face-down)
    for c in me.prize:
        if c is not None:
            seen.append(c.id)
    # our stadium in play
    for c in st.stadium:
        if c.playerIndex == st.yourIndex:
            seen.append(c.id)
    return seen


def _subtract(pool: list[int], remove: list[int]) -> list[int]:
    """Multiset subtraction: pool - remove (best effort, ignores missing)."""
    out = list(pool)
    for r in remove:
        if r in out:
            out.remove(r)
    return out


def build_determinization_true(obs, rng):
    """Build a (your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active)
    tuple using the TRUE full deck on BOTH sides (probe only).

    We know the full 60-card list for us. Subtract our visible cards -> our hidden
    pool, then split into deck (deckCount) and prize (prize count) sizes.
    For the opponent we do the same using their full Lucario list and THEIR visible
    cards (since this is self-play we genuinely know their list)."""
    st = obs.current
    yi = st.yourIndex
    me = st.players[yi]
    opp = st.players[1 - yi]

    # --- our hidden pool ---
    our_hidden = _subtract(_full_deck_multiset(), _visible_our_cards(obs))
    rng.shuffle(our_hidden)
    need_my_prize = len(me.prize)
    need_my_deck = me.deckCount
    your_prize = our_hidden[:need_my_prize]
    your_deck = our_hidden[need_my_prize:need_my_prize + need_my_deck]

    # --- opponent hidden pool (we cheat: use their true full list) ---
    opp_seen = []
    for pk in list(opp.active) + list(opp.bench):
        if pk is None:
            continue
        opp_seen.append(pk.id)
        for c in pk.energyCards:
            opp_seen.append(c.id)
        for c in pk.tools:
            opp_seen.append(c.id)
        for c in pk.preEvolution:
            opp_seen.append(c.id)
    for c in opp.discard:
        opp_seen.append(c.id)
    for c in opp.prize:
        if c is not None:
            opp_seen.append(c.id)
    for c in st.stadium:
        if c.playerIndex == (1 - yi):
            opp_seen.append(c.id)

    opp_hidden = _subtract(_full_deck_multiset(), opp_seen)
    rng.shuffle(opp_hidden)
    need_opp_prize = len(opp.prize)
    need_opp_deck = opp.deckCount
    need_opp_hand = opp.handCount

    opp_prize = opp_hidden[:need_opp_prize]
    rest = opp_hidden[need_opp_prize:]
    opp_deck = rest[:need_opp_deck]
    rest2 = rest[need_opp_deck:]
    opp_hand = rest2[:need_opp_hand]

    # face-down active?
    opp_active = []
    if len(opp.active) > 0 and opp.active[0] is None:
        # pick a basic from the pool (Riolu/Solrock/Lunatone/Makuhita are basics)
        for basic in (677, 676, 675, 673):
            opp_active = [basic]
            break

    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active


def reach_midgame_decision(target_turn=6, max_moves=400):
    """Play forward with the heuristic on both sides until we hit a MAIN decision
    at turn >= target_turn with > 1 option. Return the live obs_dict (with
    search_begin_input present)."""
    obs_dict, start = battle_start(LUCARIO_DECK, LUCARIO_DECK)
    if obs_dict is None:
        raise RuntimeError(f"battle_start failed: errorType={start.errorType}")
    moves = 0
    while moves < max_moves:
        obs = to_observation_class(obs_dict)
        if obs.current and obs.current.result != -1:
            # Game ended before we found a good decision; restart not handled here
            raise RuntimeError("Game ended before reaching a mid-game decision.")
        sel = obs.select
        st = obs.current
        # Good probe point: a real decision mid-game with branching.
        if (sel is not None and st is not None and st.turn >= target_turn
                and len(sel.option) >= 2 and obs_dict.get("search_begin_input")):
            return obs_dict
        # else step forward with the heuristic
        choice = heuristic_agent(obs_dict)
        obs_dict = battle_select(choice)
        moves += 1
    raise RuntimeError("Did not reach a mid-game decision within max_moves.")


def rollout(root_state, rng, depth_cap=None):
    """From a SearchState, roll out to terminal (or depth_cap) choosing each
    decision via the heuristic policy applied to the search-state observation.
    Returns (result, n_steps, total_step_time)."""
    state = root_state
    steps = 0
    total_t = 0.0
    while True:
        obs = state.observation
        if obs.current is not None and obs.current.result != -1:
            return obs.current.result, steps, total_t
        sel = obs.select
        if sel is None:
            # terminal-ish / no decision
            return (obs.current.result if obs.current else -1), steps, total_t
        if depth_cap is not None and steps >= depth_cap:
            return -1, steps, total_t  # depth cut, undecided
        # Decide via the FULL heuristic on the search-state Observation dataclass.
        # This gives a realistic steps-to-terminal (the heuristic plays a real game,
        # not a race-to-attack). Falls back to a safe legal move on any error.
        try:
            choice = decide_on_obs(obs)
            # sanity: enforce legality bounds
            n = len(sel.option)
            choice = [c for c in dict.fromkeys(choice) if 0 <= c < n]
            if len(choice) < sel.minCount:
                choice = _heuristic_on_state(obs, rng)
            elif len(choice) > sel.maxCount:
                choice = choice[:sel.maxCount]
        except Exception:
            choice = _heuristic_on_state(obs, rng)
        t0 = time.perf_counter()
        state = search_step(state.searchId, choice)
        total_t += time.perf_counter() - t0
        steps += 1


def _heuristic_on_state(obs, rng):
    """Choose a move for a search-state Observation (dataclass, no dict).

    The heuristic agent operates on an obs_dict. Rather than re-serialize, we use
    a lightweight legal-move policy here for the PROBE: greedily prefer ATTACK/END
    to make rollouts terminate, else random legal. This is enough to MEASURE engine
    cost (search_step dominates); the real agent uses the full heuristic on dicts."""
    sel = obs.select
    n = len(sel.option)
    mn = max(sel.minCount, 1) if sel.minCount > 0 else (1 if sel.maxCount >= 1 else 0)
    mx = sel.maxCount
    # Prefer a single ATTACK or END option when available (drives games to terminal).
    from cg.api import OptionType
    attack_idx = [i for i, o in enumerate(sel.option) if o.type == OptionType.ATTACK]
    end_idx = [i for i, o in enumerate(sel.option) if o.type == OptionType.END]
    if mn <= 1 <= mx:
        if attack_idx:
            return [attack_idx[0]]
        if end_idx:
            return [end_idx[0]]
        return [rng.randrange(n)]
    # multi-select: pick the first mn..mx indices
    k = mn if mn > 0 else min(1, n)
    k = max(k, sel.minCount)
    if k == 0:
        return []
    idxs = list(range(n))
    rng.shuffle(idxs)
    return sorted(idxs[:max(k, sel.minCount)])


def main():
    n_rollouts = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    rng = random.Random(seed)

    print(f"=== Q-SEARCH probe (n_rollouts={n_rollouts}, seed={seed}) ===")
    print("Reaching a mid-game MAIN decision via heuristic self-play...")
    obs_dict = reach_midgame_decision(target_turn=6)
    obs = to_observation_class(obs_dict)
    st = obs.current
    print(f"  Reached turn={st.turn}, yourIndex={st.yourIndex}, "
          f"select.context={obs.select.context}, "
          f"n_options={len(obs.select.option)}, "
          f"my_deckCount={st.players[st.yourIndex].deckCount}, "
          f"opp_deckCount={st.players[1-st.yourIndex].deckCount}, "
          f"opp_handCount={st.players[1-st.yourIndex].handCount}")

    # --- time search_begin over several determinizations ---
    begin_times = []
    rollout_results = []  # (n_steps, total_step_time)
    n_root_options = len(obs.select.option)

    for r in range(n_rollouts):
        det = build_determinization_true(obs, rng)
        your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active = det
        t0 = time.perf_counter()
        try:
            root = search_begin(
                obs, your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active
            )
        except Exception as e:
            print(f"  [rollout {r}] search_begin FAILED: {type(e).__name__}: {e}")
            search_end()
            continue
        begin_times.append(time.perf_counter() - t0)
        # full rollout to terminal
        result, steps, step_t = rollout(root, rng)
        rollout_results.append((steps, step_t, result))
        search_end()

    battle_finish()

    if not begin_times:
        print("No successful search_begin calls — cannot report timings.")
        return

    mean_begin = statistics.mean(begin_times)
    all_steps = [s for s, _, _ in rollout_results]
    all_step_t = [t for _, t, _ in rollout_results]
    step_per_call = []
    for s, t, _ in rollout_results:
        if s > 0:
            step_per_call.append(t / s)
    total_rollout_t = [b + t for b, (_, t, _) in zip(begin_times, rollout_results)]

    mean_steps = statistics.mean(all_steps) if all_steps else 0
    mean_step_time = statistics.mean(step_per_call) if step_per_call else 0
    mean_rollout_t = statistics.mean(total_rollout_t) if total_rollout_t else 0
    rollouts_per_sec = 1.0 / mean_rollout_t if mean_rollout_t > 0 else 0

    finished = sum(1 for _, _, res in rollout_results if res != -1)

    print()
    print("=== Q-SEARCH RESULTS (ARM emulation via Docker) ===")
    print(f"  successful rollouts        : {len(rollout_results)}/{n_rollouts} "
          f"(reached terminal: {finished})")
    print(f"  root option count          : {n_root_options}")
    print(f"  search_begin time (mean)   : {mean_begin*1000:.3f} ms")
    print(f"  search_step time (mean)    : {mean_step_time*1000:.4f} ms/step")
    print(f"  steps-to-terminal (mean)   : {mean_steps:.1f}")
    print(f"  steps-to-terminal (max)    : {max(all_steps) if all_steps else 0}")
    print(f"  full rollout time (mean)   : {mean_rollout_t*1000:.2f} ms "
          f"(begin + ~{mean_steps:.0f} steps)")
    print(f"  rollouts / sec             : {rollouts_per_sec:.1f}")
    print()

    # --- budget estimate ---
    # 600s bank for the WHOLE game. A long game ~ 100+ decisions for us. Be safe.
    BANK = 600.0
    SAFETY = 0.5          # use at most half the bank for search; keep margin
    EST_DECISIONS = 120   # conservative upper bound on our decisions in a long game
    per_decision_budget = (BANK * SAFETY) / EST_DECISIONS
    rollouts_per_decision = per_decision_budget * rollouts_per_sec
    print("=== BUDGET ESTIMATE ===")
    print(f"  bank                       : {BANK:.0f} s/game")
    print(f"  safety fraction            : {SAFETY}")
    print(f"  est. decisions / game      : {EST_DECISIONS}")
    print(f"  per-decision budget        : {per_decision_budget*1000:.1f} ms")
    print(f"  rollouts that fit / decision: {rollouts_per_decision:.1f}")
    print()
    # K determinizations x O root options x 1 rollout each = K*O rollouts at root.
    if rollouts_per_decision > 0:
        feasible_KO = rollouts_per_decision
        print(f"  => At root with {n_root_options} options, a budget of "
              f"{rollouts_per_decision:.0f} rollouts supports "
              f"K = {max(1, int(feasible_KO / max(1, n_root_options)))} "
              f"determinizations (1 rollout per option per determinization).")
    print()
    print("NOTE: This is ARM-emulated Docker; native Kaggle x86_64 is typically "
          "2-5x faster, so real budgets are more generous.")


if __name__ == "__main__":
    main()
