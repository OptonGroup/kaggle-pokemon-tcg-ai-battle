"""v003 — determinized engine-search agent (dev copy).

Strategy:
  - At MAIN decisions (the impactful action choices) with >=2 options, run a
    determinized 1-ply + rollout search:
      * sample plausible hidden states (determinizations),
      * for each determinization, pick a root option via round-robin (UCB1-guided
        selection distributes rollouts towards promising options), roll out to
        terminal with the heuristic pilot (or to a depth cap, then score the leaf),
      * aggregate scores across all determinizations and pick the best root option.
  - Budget-driven: keep running rollouts until the per-decision deadline.
    Distribute rollouts round-robin across options initially, then focus via UCB1.
    Fresh determinization every few rollouts to average over hidden-state uncertainty.
  - At every OTHER decision (energy attach, card pick, yes/no, ...) just use the
    heuristic directly — search there is low-value and risks budget/complexity.
  - A self-imposed time budget keeps the ~600s/game bank safe: total time used is
    tracked; per-decision budget = remaining / est. remaining decisions, with a
    hard floor below which we skip search and play the heuristic move.
  - NEVER throws: the public agent() wraps everything and falls back to the
    heuristic move, then to a trivially-legal move.

Environment variables (read at import, override defaults):
  V003_USABLE_FRACTION   float 0-1, fraction of 600s bank usable for search (def 0.55)
  V003_MAX_DECISION_S    float, hard per-decision cap in seconds (def 6.0)
  V003_MIN_SIMS          int, min simulations before trusting result (def 10)
  V003_DET_BATCH         int, rollouts between fresh determinizations (def 5)

This dev copy imports the heuristic from agents.heuristic_agent. The packaged
submission (submissions/v003_search/main.py) VENDORS all of this self-contained.
"""
import os
import sys
import time
import math
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# Ensure src/ root (parent of agents/) on path for `import cg`
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from cg.api import (
    AreaType, CardType, OptionType, SelectContext, Observation, Pokemon,
    to_observation_class, search_begin, search_step, search_end,
)

# Reuse the heuristic pilot as fallback + rollout policy.
import agents.heuristic_agent as H
from agents.heuristic_agent import my_deck, decide_on_obs, card_table

# ---------------------------------------------------------------------------
# Config (env-var overridable for local gate vs Kaggle)
# ---------------------------------------------------------------------------
TOTAL_BANK_S = 600.0
USABLE_FRACTION = float(os.environ.get("V003_USABLE_FRACTION", "0.55"))
EST_DECISIONS = 120            # conservative est. of our decisions in a long game
MIN_BUDGET_S = 0.10            # below this per-decision budget, skip search
HARD_PER_DECISION_CAP_S = float(os.environ.get("V003_MAX_DECISION_S", "6.0"))
MIN_SIMS = int(os.environ.get("V003_MIN_SIMS", "10"))  # min rollouts per option before trusting
DET_BATCH = int(os.environ.get("V003_DET_BATCH", "5"))  # rollouts between fresh determinizations
ROLLOUT_DEPTH_CAP = 140        # max search_steps per rollout before leaf eval

# UCB1 exploration constant (higher = more exploration across options)
UCB_C = 1.414

# A safe Basic Pokemon id from the Lucario prior, for padding / face-down active.
_SAFE_BASIC = 677  # Riolu (Basic)
_PRIOR_BASICS = (677, 676, 675, 673)  # Riolu, Solrock, Lunatone, Makuhita

# ---------------------------------------------------------------------------
# Running time accounting (module-global; reset at deck-selection / new game)
# ---------------------------------------------------------------------------
_time_spent = 0.0
_decisions_made = 0
_game_started = False


def _reset_game_state():
    global _time_spent, _decisions_made, _game_started
    _time_spent = 0.0
    _decisions_made = 0
    _game_started = True


# ---------------------------------------------------------------------------
# Determinization
# ---------------------------------------------------------------------------

def _full_deck() -> list[int]:
    return list(my_deck)


def _visible_card_ids(player_state, stadium, player_index) -> list[int]:
    seen = []
    if player_state.hand:
        for c in player_state.hand:
            seen.append(c.id)
    for pk in list(player_state.active) + list(player_state.bench):
        if pk is None:
            continue
        seen.append(pk.id)
        for c in pk.energyCards:
            seen.append(c.id)
        for c in pk.tools:
            seen.append(c.id)
        for c in pk.preEvolution:
            seen.append(c.id)
    for c in player_state.discard:
        seen.append(c.id)
    for c in player_state.prize:
        if c is not None:
            seen.append(c.id)
    for c in stadium:
        if c.playerIndex == player_index:
            seen.append(c.id)
    return seen


def _multiset_subtract(pool, remove):
    out = list(pool)
    for r in remove:
        if r in out:
            out.remove(r)
    return out


def _pad_to(lst, n, fill):
    out = list(lst)
    while len(out) < n:
        out.append(fill)
    return out[:n]


def build_determinization(obs: Observation, rng: random.Random):
    """Return (your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active)
    sized to match the engine's requirements, sampled from priors.

    Our hidden pool = our full 60-card deck minus everything visible to us.
    Opponent prior = the SAME Lucario list (v003 prior). Opp hidden pool = prior
    minus their visible cards; sample deck/prize/hand from it, pad if short.
    """
    st = obs.current
    yi = st.yourIndex
    me = st.players[yi]
    opp = st.players[1 - yi]

    # --- our hidden cards ---
    our_hidden = _multiset_subtract(_full_deck(), _visible_card_ids(me, st.stadium, yi))
    rng.shuffle(our_hidden)
    need_my_prize = len(me.prize)
    need_my_deck = me.deckCount
    # If our hidden pool is short (shouldn't happen), pad with a safe basic.
    our_hidden = _pad_to(our_hidden, need_my_prize + need_my_deck, _SAFE_BASIC)
    your_prize = our_hidden[:need_my_prize]
    your_deck = our_hidden[need_my_prize:need_my_prize + need_my_deck]

    # --- opponent hidden cards (prior = Lucario list) ---
    opp_visible = _visible_card_ids(opp, st.stadium, 1 - yi)
    opp_hidden = _multiset_subtract(_full_deck(), opp_visible)
    rng.shuffle(opp_hidden)
    need_opp_prize = len(opp.prize)
    need_opp_deck = opp.deckCount
    need_opp_hand = opp.handCount
    total_opp = need_opp_prize + need_opp_deck + need_opp_hand
    opp_hidden = _pad_to(opp_hidden, total_opp, _SAFE_BASIC)
    opp_prize = opp_hidden[:need_opp_prize]
    rest = opp_hidden[need_opp_prize:]
    opp_deck = rest[:need_opp_deck]
    rest2 = rest[need_opp_deck:]
    opp_hand = rest2[:need_opp_hand]

    # --- face-down opponent active ---
    opp_active = []
    if len(opp.active) > 0 and opp.active[0] is None:
        opp_active = [_PRIOR_BASICS[0]]

    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active


# ---------------------------------------------------------------------------
# Rollout policy + leaf eval (isolated heuristic state)
# ---------------------------------------------------------------------------

def _legalize(choice, sel):
    """Coerce a heuristic choice into a legal selection for `sel`."""
    n = len(sel.option)
    seen = []
    for c in choice:
        if isinstance(c, int) and 0 <= c < n and c not in seen:
            seen.append(c)
    if len(seen) > sel.maxCount:
        seen = seen[:sel.maxCount]
    if len(seen) < sel.minCount:
        # pad with first unused legal indices
        for i in range(n):
            if i not in seen:
                seen.append(i)
            if len(seen) >= sel.minCount:
                break
    return seen


def _rollout_choice(obs: Observation, rng: random.Random):
    """Heuristic move for a rollout decision; safe fallback to first legal."""
    sel = obs.select
    try:
        choice = decide_on_obs(obs)
        return _legalize(choice, sel)
    except Exception:
        return _legalize(list(range(sel.maxCount)), sel)


def _leaf_eval(obs: Observation, our_index: int) -> float:
    """Heuristic value of a non-terminal leaf, from our perspective, in [-1, 1].

    Primary signal: prize differential (fewer prizes remaining = closer to win).
    Secondary: board HP + energy advantage. Normalized to ~[-1, 1]."""
    st = obs.current
    me = st.players[our_index]
    opp = st.players[1 - our_index]
    # prizes remaining: lower is better for that player
    my_prize = len(me.prize)
    opp_prize = len(opp.prize)
    prize_diff = (opp_prize - my_prize)  # +ve = we are ahead, range ~[-6, 6]
    v = prize_diff / 6.0 * 0.8

    def board_strength(ps):
        s = 0
        for pk in list(ps.active) + list(ps.bench):
            if pk is None:
                continue
            s += pk.hp + len(pk.energies) * 20
        return s
    my_b = board_strength(me)
    opp_b = board_strength(opp)
    denom = (my_b + opp_b) or 1
    v += ((my_b - opp_b) / denom) * 0.2
    if v > 1.0:
        v = 1.0
    elif v < -1.0:
        v = -1.0
    return v


def _result_value(result: int, our_index: int) -> float:
    if result == our_index:
        return 1.0
    if result == (1 - our_index):
        return -1.0
    return 0.0  # draw (result == 2) or anything else


def _rollout_from(state, our_index: int, rng: random.Random, deadline: float):
    """Roll out a SearchState to terminal (or depth/time cap). Return value in
    [-1, 1] from our perspective."""
    steps = 0
    while True:
        obs = state.observation
        if obs.current is not None and obs.current.result != -1:
            return _result_value(obs.current.result, our_index)
        sel = obs.select
        if sel is None:
            if obs.current is not None and obs.current.result != -1:
                return _result_value(obs.current.result, our_index)
            return _leaf_eval(obs, our_index)
        if steps >= ROLLOUT_DEPTH_CAP or time.perf_counter() >= deadline:
            return _leaf_eval(obs, our_index)
        choice = _rollout_choice(obs, rng)
        state = search_step(state.searchId, choice)
        steps += 1


# ---------------------------------------------------------------------------
# Core decision — budget-driven UCB1 sampling
# ---------------------------------------------------------------------------

def _per_decision_budget() -> float:
    remaining = TOTAL_BANK_S * USABLE_FRACTION - _time_spent
    if remaining <= 0:
        return 0.0
    est_left = max(1, EST_DECISIONS - _decisions_made)
    budget = remaining / est_left
    return min(budget, HARD_PER_DECISION_CAP_S)


def _save_heuristic_globals():
    return (H.plan, H.pre_turn, H.ability_used)


def _restore_heuristic_globals(saved):
    H.plan, H.pre_turn, H.ability_used = saved


def _ucb1_select(scores, counts, total_n, n_opts):
    """Select the option index with highest UCB1 value.
    Options with 0 counts get infinite priority (ensure each visited at least once)."""
    best_i = -1
    best_v = -1e18
    for i in range(n_opts):
        if counts[i] == 0:
            return i  # visit unvisited options first
        mean = scores[i] / counts[i]
        ucb = mean + UCB_C * math.sqrt(math.log(total_n) / counts[i])
        if ucb > best_v:
            best_v = ucb
            best_i = i
    return best_i


def _get_candidate_options(obs: Observation, n_opts: int) -> list[int]:
    """Return a pruned list of candidate option indices to search.

    Uses the heuristic to score all options, then keeps only those within
    a threshold of the best score. This concentrates rollouts on plausible
    moves and avoids wasting budget on obviously inferior options.

    Returns all indices if the heuristic fails or threshold filtering is off.
    """
    try:
        saved = _save_heuristic_globals()
        try:
            h_choice = decide_on_obs(obs)
        finally:
            _restore_heuristic_globals(saved)

        if not h_choice:
            return list(range(n_opts))

        # The heuristic returns sorted indices (best first) for MAIN decisions.
        # Keep the top-K candidates, where K is chosen to balance rollout depth
        # vs breadth. Typically: top 4 candidates gets 5x more rollouts/candidate.
        MAX_CANDIDATES = 4
        # Always include the heuristic's top pick + a few alternatives
        candidates = list(dict.fromkeys(h_choice[:MAX_CANDIDATES]))
        # Fill to MAX_CANDIDATES from remaining options if needed
        for i in range(n_opts):
            if len(candidates) >= MAX_CANDIDATES:
                break
            if i not in candidates:
                candidates.append(i)
        return candidates
    except Exception:
        return list(range(n_opts))


def _search_decision(obs: Observation, budget: float, rng: random.Random):
    """Run budget-driven determinized rollout search for a MAIN single-select decision.

    Algorithm:
      1. Pre-filter: use the heuristic to identify top candidates (reduces options
         from ~10 to ~4, tripling rollouts per candidate).
      2. UCB1-guided sampling: keep running rollouts until the deadline, with
         fresh determinizations every DET_BATCH rollouts. UCB1 focuses rollouts
         on promising options after MIN_SIMS baseline coverage.
      3. Return the option with the highest mean rollout value. Fall back to
         heuristic if no search runs (caller handles the None return).

    Returns the chosen [index] or None if search couldn't run (caller falls
    back to the heuristic move)."""
    sel = obs.select
    n_opts = len(sel.option)
    if n_opts == 0:
        return None
    # MAIN actions are single-select; bail to heuristic otherwise.
    if sel.minCount != 1 or sel.maxCount != 1:
        return None

    # Pre-filter: only search the top candidates
    candidates = _get_candidate_options(obs, n_opts)
    n_cands = len(candidates)

    our_index = obs.current.yourIndex
    start = time.perf_counter()
    deadline = start + budget

    scores = [0.0] * n_cands
    counts = [0] * n_cands
    total_rollouts = 0

    # Keep a fresh determinization open for batches of rollouts.
    # Each entry in current_root is (root_state, det_remaining).
    current_root = None
    det_uses = 0

    def open_new_det():
        nonlocal current_root, det_uses
        if current_root is not None:
            # close previous
            try:
                search_end()
            except Exception:
                pass
            current_root = None

        if time.perf_counter() >= deadline:
            return False

        det = build_determinization(obs, rng)
        yd, yp, od, op, oh, oa = det
        try:
            root = search_begin(obs, yd, yp, od, op, oh, oa)
            current_root = root
            det_uses = 0
            return True
        except Exception:
            try:
                search_end()
            except Exception:
                pass
            return False

    def close_det():
        nonlocal current_root
        if current_root is not None:
            try:
                search_end()
            except Exception:
                pass
            current_root = None

    # The heuristic globals are mutated by rollouts; save & restore around search.
    saved = _save_heuristic_globals()
    try:
        while time.perf_counter() < deadline:
            # Open a new determinization if needed
            if current_root is None:
                if not open_new_det():
                    break

            # Pick candidate index (not raw option index) to sample.
            # Round-robin until all candidates have MIN_SIMS, then UCB1.
            all_have_min = all(c >= MIN_SIMS for c in counts)
            if not all_have_min:
                cand_i = min(range(n_cands), key=lambda i: counts[i])
            else:
                cand_i = _ucb1_select(scores, counts, total_rollouts, n_cands)

            opt_idx = candidates[cand_i]  # actual option index for search_step

            if time.perf_counter() >= deadline:
                break

            try:
                child = search_step(current_root.searchId, [opt_idx])
            except Exception:
                close_det()
                continue

            obs_c = child.observation
            if obs_c.current is not None and obs_c.current.result != -1:
                val = _result_value(obs_c.current.result, our_index)
            else:
                val = _rollout_from(child, our_index, rng, deadline)

            scores[cand_i] += val
            counts[cand_i] += 1
            total_rollouts += 1
            det_uses += 1

            # Rotate determinization every DET_BATCH rollouts
            if det_uses >= DET_BATCH:
                close_det()

    finally:
        close_det()
        _restore_heuristic_globals(saved)

    # Need at least one evaluated candidate to trust search.
    evaluated = [i for i in range(n_cands) if counts[i] > 0]
    if not evaluated:
        return None

    # Pick the candidate with the highest mean value.
    best_cand = max(evaluated, key=lambda i: scores[i] / counts[i])
    return [candidates[best_cand]]


def decide(obs_dict: dict) -> list[int]:
    """Core decision (may raise; wrapped by agent())."""
    global _time_spent, _decisions_made, _game_started

    obs = to_observation_class(obs_dict)

    # Deck selection — also marks a fresh game.
    if obs.select is None:
        _reset_game_state()
        return my_deck

    if not _game_started:
        _reset_game_state()

    sel = obs.select
    context = sel.context

    t0 = time.perf_counter()
    try:
        use_search = (
            context == SelectContext.MAIN
            and len(sel.option) >= 2
            and sel.minCount == 1 and sel.maxCount == 1
            and obs_dict.get("search_begin_input") is not None
        )
        budget = _per_decision_budget() if use_search else 0.0

        if use_search and budget >= MIN_BUDGET_S:
            rng = random.Random((_decisions_made * 1000003) ^ 0x9E3779B1)
            chosen = _search_decision(obs, budget, rng)
            if chosen is not None:
                return chosen
        # Fallback / non-MAIN: heuristic move (operate on live obs_dict so the
        # heuristic's own globals advance correctly for the real game).
        return H._agent_impl(obs_dict)
    finally:
        dt = time.perf_counter() - t0
        _time_spent += dt
        _decisions_made += 1


def agent(obs_dict: dict) -> list[int]:
    """Never-throw public wrapper."""
    try:
        return decide(obs_dict)
    except Exception:
        try:
            return H.agent(obs_dict)
        except Exception:
            obs = to_observation_class(obs_dict)
            if obs.select is None:
                return my_deck
            n = len(obs.select.option)
            k = max(1, obs.select.minCount)
            return list(range(min(k, n)))
