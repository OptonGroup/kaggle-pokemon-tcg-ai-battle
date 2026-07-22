#!/usr/bin/env python3
"""gate_behavioral_fidelity.py — Phase 4.2 behavioral-fidelity gate for a
Grimmsnarl/Munkidori (jiatu.l) candidate submission.

Runs the candidate (default: submissions/v062_grimmsnarl_playbook) with its
per-decision stderr trace enabled ([dbg]/[dec] lines written by main.py's
_dbg / scoring_grimmsnarl.py's _dlog — see those two files for the exact
format) against 5 opponent proxies on the native macOS engine (cg_native,
libcg.dylib — NOT the Linux libcg.so). The engine is single-threaded here;
do not call battle_start/battle_select from multiple threads concurrently
(known SIGSEGV risk on this engine, see scripts/thread_check_native.py).

Runs (N games each, seat-balanced, seed fixed):
  vs_crustle          submissions/v009a_crustle
  vs_alakazam         submissions/v017_alakazam_heur
  vs_mirror           submissions/v061_grimmsnarl_heuristic (same jiatu.l deck)
  vs_lucario_greedy   decks/meta/MegaLucario_live.csv, greedy-attack proxy pilot
  vs_trmewtwo_greedy  decks/meta/TRMewtwo_live.csv, greedy-attack proxy pilot

Attribution trick: instead of trying to split one merged stderr stream by
prefix (fragile, and BOTH sides can emit [dbg]/[dec] lines — v061 the mirror
opponent uses the identical logging format), each single agent(obs_dict)
call is wrapped individually in contextlib.redirect_stderr(io.StringIO()).
Since the call is synchronous and single-threaded, whatever text a call
produces belongs unambiguously to whichever side just moved — this is used
to keep ONLY the candidate's own captured lines, discarding the opponent's,
which sidesteps the interleaving problem entirely rather than heuristically
guessing at it.

Metrics are read two ways: (1) parsed straight off the candidate's captured
[dbg]/[dec] trace lines (per the task spec), and (2) cross-checked against
ground-truth public board state (active+bench card IDs/energies), which the
harness can read directly off obs_dict after every battle_select step
regardless of whose turn it is (active/bench are public info in TCG).

Usage:
    .venv/bin/python scripts/gate_behavioral_fidelity.py \
        --candidate v062_grimmsnarl_playbook --games 40

    # quick smoke (2 games/run)
    .venv/bin/python scripts/gate_behavioral_fidelity.py --games 2

KNOWN LIMITATION — engine RNG is NOT seedable from Python: cg_native's
BattleStart() ctypes binding takes no seed argument (see src/cg_native/sim.py
lib.BattleStart.argtypes — just the 120 deck-card ints), and the compiled
libcg.dylib evidently keeps its own internal RNG state for deck shuffle /
opening hand / coin flip that is untouched by `random.seed()` in this
process — verified empirically: calling play_one_game() 3x in a row with the
IDENTICAL seed/deck/opponent/side inside the SAME process produces 3
different game lengths and 3 different decision traces. This is a
pre-existing property of the cg_native binding (scripts/h3_gate.py's worker
does `random.seed(seed)` before battle_start() the same way, with the same
limitation) — not something this script can fix from the Python side. The
`--seed` flag therefore fixes the deterministic PARTS of the run (opponent
assignment, our_first seat sequence via gi % 2, which games belong to which
run) and is recorded in the output for audit-trail purposes, but does NOT
make individual games bit-reproducible. This is exactly why the gate reports
N=40-game AGGREGATES rather than judging any single game — the per-game RNG
noise is expected and is what N=40 is meant to average out.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Native engine (macOS arm64) — cg_native wraps libcg.dylib. Register it under
# the "cg" / "cg.api" / ... module names BEFORE loading any submission module,
# so every `from cg.api import ...` / `import cg` inside a submission resolves
# to this single native engine instance rather than whatever Linux-oriented
# cg/ package happens to be bundled alongside that submission's main.py.
# Mirrors scripts/h3_gate.py's established pattern in this repo.
# ---------------------------------------------------------------------------
import cg_native  # noqa: E402
import cg_native.api  # noqa: E402
import cg_native.game  # noqa: E402
import cg_native.sim  # noqa: E402
import cg_native.utils  # noqa: E402
from cg_native.game import battle_start, battle_select, battle_finish  # noqa: E402
from cg_native.api import all_attack  # noqa: E402

sys.modules.setdefault("cg", cg_native)
sys.modules.setdefault("cg.api", cg_native.api)
sys.modules.setdefault("cg.game", cg_native.game)
sys.modules.setdefault("cg.sim", cg_native.sim)
sys.modules.setdefault("cg.utils", cg_native.utils)

GATES_DIR = ROOT / "results" / "gates"
GATES_DIR.mkdir(parents=True, exist_ok=True)

MUNKIDORI_ID = 112
MARNIE_LINE_IDS = {646, 647, 648}  # Impidimp / Morgrem / Grimmsnarl ex
GRIMMSNARL_EX_ID = 648

DEFAULT_MAX_MOVES = 600

# ---------------------------------------------------------------------------
# [dec] trace-line parser — format written by scoring_grimmsnarl.py's
# choose_options()/_dlog():
#   t{turn} m={matchup} ctx{ctx} n{n} -> #{idx} '{reason}' s={score} [| 2nd ...] pick={sel}
# ---------------------------------------------------------------------------
DEC_RE = re.compile(
    r"^t(?P<turn>\S+)\s+m=(?P<matchup>\S+)\s+ctx(?P<ctx>-?\d+)\s+n(?P<n>\d+)\s+->\s+"
    r"#(?P<idx>-?\d+)\s+'(?P<reason>.*?)'\s+s=(?P<score>-?\d+)"
)


# ---------------------------------------------------------------------------
# Submission / meta-deck loading (fresh module instance per game — stateful
# agents keep module-level globals, e.g. v062's _adrena_state / counters;
# reloading fresh each game is the established stateful-audit convention in
# this repo, see scripts/h3_gate.py's _load_submission_module).
# ---------------------------------------------------------------------------
_LOAD_COUNTER = 0


def _load_submission_module(sub_dir_name: str):
    global _LOAD_COUNTER
    _LOAD_COUNTER += 1
    sub_dir = ROOT / "submissions" / sub_dir_name
    main_path = sub_dir / "main.py"
    assert main_path.exists(), f"не найден {main_path}"
    if str(sub_dir) not in sys.path:
        sys.path.insert(0, str(sub_dir))
    mod_name = f"_fidelity_gate_{sub_dir_name}_{_LOAD_COUNTER}"
    old_cwd = os.getcwd()
    try:
        os.chdir(sub_dir)
        spec = importlib.util.spec_from_file_location(mod_name, main_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = str(main_path)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(old_cwd)
    return mod


def _load_deck_csv(path: Path) -> list[int]:
    ids = [int(x.strip()) for x in path.read_text().splitlines() if x.strip()]
    assert len(ids) == 60, f"{path}: expected 60 cards, got {len(ids)}"
    return ids


def _load_meta_deck(deck_name: str) -> list[int]:
    return _load_deck_csv(ROOT / "decks" / "meta" / f"{deck_name}.csv")


# ---------------------------------------------------------------------------
# Generic greedy-attack proxy pilot for meta-deck opponents (no lgb, no
# top-level cg.api import) — ported verbatim from scripts/h3_gate.py's
# greedy_meta_agent, same OptionType/SelectType/SelectContext int constants.
# ---------------------------------------------------------------------------
_ATTACK_DAMAGE_MAP: dict[int, int] | None = None


def _get_attack_damage_map() -> dict[int, int]:
    global _ATTACK_DAMAGE_MAP
    if _ATTACK_DAMAGE_MAP is None:
        _ATTACK_DAMAGE_MAP = {a.attackId: (a.damage or 0) for a in all_attack()}
    return _ATTACK_DAMAGE_MAP


def _opt_get(o, key, default=None):
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


_OT_YES = 1
_OT_CARD = 3
_OT_PLAY = 7
_OT_ATTACH = 8
_OT_EVOLVE = 9
_OT_ABILITY = 10
_OT_RETREAT = 12
_OT_ATTACK = 13
_OT_END = 14
_ST_YES_NO = 9
_ST_COUNT = 8
_SC_MAIN = 0
_GREEDY_PRIORITY = (_OT_ATTACK, _OT_ABILITY, _OT_EVOLVE, _OT_PLAY, _OT_ATTACH, _OT_RETREAT, _OT_CARD, _OT_END)


def greedy_meta_agent(obs_dict: dict) -> list[int]:
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return []
    opts = sel.get("option") or []
    n = len(opts)
    if n == 0:
        return []
    sel_type = sel.get("type")
    if sel_type == _ST_YES_NO:
        for i in range(n):
            if _opt_get(opts[i], "type") == _OT_YES:
                return [i]
        return [0]
    if sel_type == _ST_COUNT:
        best = max(range(n), key=lambda i: (_opt_get(opts[i], "number", 0) or 0))
        return [best]
    min_c = int(sel.get("minCount") or 1)
    max_c = int(sel.get("maxCount") or 1)
    context = sel.get("context")
    if context == _SC_MAIN:
        atk_idxs = [i for i in range(n) if _opt_get(opts[i], "type") == _OT_ATTACK]
        if atk_idxs:
            dmg_map = _get_attack_damage_map()
            best = max(atk_idxs, key=lambda i: (dmg_map.get(_opt_get(opts[i], "attackId"), 0) or 0))
            return [best]
        for ot in _GREEDY_PRIORITY:
            idxs = [i for i in range(n) if _opt_get(opts[i], "type") == ot]
            if idxs:
                return [idxs[0]]
    k = max(1, min_c)
    k = min(k, max_c, n)
    return list(range(k))


# ---------------------------------------------------------------------------
# Stderr-capturing agent call wrapper
# ---------------------------------------------------------------------------

def _call_agent(fn, obs_dict):
    """Call fn(obs_dict), capturing everything it wrote to sys.stderr during
    THIS call only. Synchronous/single-threaded, so no cross-talk is possible
    between sides — solves the mirror-matchup attribution problem exactly."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        choice = fn(obs_dict)
    return choice, buf.getvalue()


def _split_trace(chunks: list[str]) -> tuple[list[str], list[str]]:
    dbg_lines, dec_lines = [], []
    for chunk in chunks:
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("[dbg] "):
                dbg_lines.append(line[len("[dbg] "):])
            elif line.startswith("[dec] "):
                dec_lines.append(line[len("[dec] "):])
    return dbg_lines, dec_lines


# ---------------------------------------------------------------------------
# Board-state scan (ground truth, public info — works regardless of whose
# turn it is): tracks every card id ever seen on the candidate's own
# active+bench, and whether Munkidori (112) was ever seen with >=1 energy.
# ---------------------------------------------------------------------------

def _scan_board(obs_dict: dict, our_idx: int, state: dict) -> None:
    cur = obs_dict.get("current") or {}
    players = cur.get("players") or []
    if our_idx >= len(players):
        return
    pl = players[our_idx] or {}
    mons = list(pl.get("active") or []) + list(pl.get("bench") or [])
    for pk in mons:
        if not isinstance(pk, dict):
            continue
        cid = pk.get("id")
        if cid is None:
            continue
        cid = int(cid)
        state["ids_seen"].add(cid)
        if cid == MUNKIDORI_ID and len(pk.get("energies") or []) >= 1:
            state["munkidori_energized"] = True


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------

def _matchup_stable(seq: list[str]) -> bool:
    """True unless the matchup label reverts to 'generic' after having
    already committed to a non-generic label earlier in the same game."""
    first_ng = next((i for i, m in enumerate(seq) if m != "generic"), None)
    if first_ng is None:
        return True
    return all(m != "generic" for m in seq[first_ng:])


def play_one_game(cand_dir: str, cand_deck: list[int], opp_kind: str, opp_spec: str,
                   our_first: bool, seed: int, max_moves: int) -> dict:
    import random
    random.seed(seed)

    cand_mod = _load_submission_module(cand_dir)
    cand_agent = cand_mod.agent

    if opp_kind == "submission":
        opp_mod = _load_submission_module(opp_spec)
        opp_agent = opp_mod.agent
        opp_deck = list(getattr(opp_mod, "my_deck", None) or _load_deck_csv(ROOT / "submissions" / opp_spec / "deck.csv"))
    elif opp_kind == "meta":
        opp_agent = greedy_meta_agent
        opp_deck = _load_meta_deck(opp_spec)
    else:
        raise ValueError(f"unknown opp_kind {opp_kind!r}")

    our_idx = 0 if our_first else 1
    if our_first:
        deck0, deck1 = list(cand_deck), list(opp_deck)
        agent0, agent1 = cand_agent, opp_agent
    else:
        deck0, deck1 = list(opp_deck), list(cand_deck)
        agent0, agent1 = opp_agent, cand_agent

    board_state = {"ids_seen": set(), "munkidori_energized": False}
    our_chunks: list[str] = []
    harness_exception = None
    move_count = 0
    outcome = "aborted"
    winner = None

    obs_dict, start = battle_start(deck0, deck1)
    if obs_dict is None:
        battle_finish()
        return dict(
            outcome="crash", error=f"battle_start failed: errorType={getattr(start, 'errorType', None)}",
            moves=0, seed=seed, our_first=our_first,
            dbg_lines=[], dec_lines=[], ids_seen=[], munkidori_energized=False,
            internal_exception_count=0, harness_exception=None,
        )

    try:
        while move_count < max_moves:
            cur = obs_dict.get("current") or {}
            result = cur.get("result", -1)
            if result is not None and result >= 0:
                winner = result
                break
            sel = obs_dict.get("select")
            if sel is None:
                break
            _scan_board(obs_dict, our_idx, board_state)

            your_index = cur.get("yourIndex", 0)
            is_our_turn = (your_index == our_idx)
            agent_fn = agent0 if your_index == 0 else agent1
            try:
                if is_our_turn:
                    choice, captured = _call_agent(agent_fn, obs_dict)
                    if captured:
                        our_chunks.append(captured)
                else:
                    choice, _discard = _call_agent(agent_fn, obs_dict)
            except Exception as e:  # never-throw is the contract; this would be a real bug
                harness_exception = f"{'our' if is_our_turn else 'opp'} side: {type(e).__name__}: {e}"
                break

            obs_dict = battle_select(choice)
            move_count += 1
        else:
            outcome = "move_limit"
        # final board scan (post-KO / end-of-game state)
        _scan_board(obs_dict, our_idx, board_state)
    finally:
        try:
            battle_finish()
        except Exception:
            pass

    if harness_exception is not None:
        outcome = "harness_exception"
    elif winner is not None:
        our_side = our_idx
        if winner == 2:
            outcome = "draw"
        elif winner == our_side:
            outcome = "our_win"
        else:
            outcome = "opp_win"
    elif move_count >= max_moves:
        outcome = "move_limit"

    dbg_lines, dec_lines = _split_trace(our_chunks)
    internal_exc = int(getattr(cand_mod, "_exception_count", 0) or 0)

    return dict(
        outcome=outcome, moves=move_count, seed=seed, our_first=our_first,
        dbg_lines=dbg_lines, dec_lines=dec_lines,
        ids_seen=sorted(board_state["ids_seen"]),
        munkidori_energized=board_state["munkidori_energized"],
        internal_exception_count=internal_exc, harness_exception=harness_exception,
    )


# ---------------------------------------------------------------------------
# Per-game metric extraction from a played-game record
# ---------------------------------------------------------------------------

def _game_metrics(game: dict) -> dict:
    dbg_lines = game["dbg_lines"]
    dec_lines = game["dec_lines"]

    n_adrena_trigger = sum(1 for l in dbg_lines if l.startswith("guard=adrena_trigger"))
    n_munkidori_energy_guard = sum(1 for l in dbg_lines if l.startswith("guard=munkidori_energy"))
    n_adrena_target_bench = sum(1 for l in dbg_lines if l.startswith("adrena target (value): bench"))
    n_adrena_target_fallback = sum(1 for l in dbg_lines if l.startswith("adrena target (value): no bench"))

    parsed = []
    for l in dec_lines:
        m = DEC_RE.match(l)
        if m:
            parsed.append(m.groupdict())

    n_munkidori_retreat = sum(1 for p in parsed if p["reason"] == "retreat useless active Munkidori")
    n_stadium_replace = sum(1 for p in parsed if p["reason"] == "play Spikemuth Gym: replace opponent's stadium")
    n_stadium_fresh = sum(1 for p in parsed if p["reason"] == "play Spikemuth Gym")
    n_evolve_grimmsnarl_dec = sum(1 for p in parsed if p["reason"].startswith("evolve into Grimmsnarl ex"))

    matchup_seq = [p["matchup"] for p in parsed]
    matchup_stable = _matchup_stable(matchup_seq)
    distinct_nongeneric = sorted({m for m in matchup_seq if m != "generic"})

    ids_seen = set(game["ids_seen"])
    grimmsnarl_ex_fielded = GRIMMSNARL_EX_ID in ids_seen
    marnie_line_present = bool(ids_seen & MARNIE_LINE_IDS)

    return dict(
        outcome=game["outcome"],
        n_adrena_trigger=n_adrena_trigger,
        n_munkidori_energy_guard=n_munkidori_energy_guard,
        n_adrena_target_bench=n_adrena_target_bench,
        n_adrena_target_fallback=n_adrena_target_fallback,
        n_munkidori_retreat=n_munkidori_retreat,
        n_stadium_replace=n_stadium_replace,
        n_stadium_fresh=n_stadium_fresh,
        n_evolve_grimmsnarl_dec=n_evolve_grimmsnarl_dec,
        matchup_stable=matchup_stable,
        distinct_nongeneric_matchups=distinct_nongeneric,
        grimmsnarl_ex_fielded=grimmsnarl_ex_fielded,
        marnie_line_present=marnie_line_present,
        munkidori_energized_board=game["munkidori_energized"],
        internal_exception_count=game["internal_exception_count"],
        harness_exception=game["harness_exception"],
        moves=game["moves"],
    )


def _mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def _frac(bools):
    return (sum(1 for b in bools if b) / len(bools)) if bools else 0.0


def _ratio(num_list, den_list):
    n = sum(num_list)
    d = sum(num_list) + sum(den_list)
    return (n / d) if d > 0 else None


def aggregate_run(games_metrics: list[dict]) -> dict:
    n = len(games_metrics)
    outcomes = [g["outcome"] for g in games_metrics]
    win_ct = outcomes.count("our_win")
    loss_ct = outcomes.count("opp_win")
    draw_ct = outcomes.count("draw")
    anomaly_ct = n - win_ct - loss_ct - draw_ct  # crash/harness_exception/move_limit

    adrena_counts = [g["n_adrena_trigger"] for g in games_metrics]
    munk_energy_guard_counts = [g["n_munkidori_energy_guard"] for g in games_metrics]
    munk_retreat_counts = [g["n_munkidori_retreat"] for g in games_metrics]

    harness_exc = sum(1 for g in games_metrics if g["harness_exception"])
    internal_exc_total = sum(g["internal_exception_count"] for g in games_metrics)

    return dict(
        n_games=n,
        win=win_ct, loss=loss_ct, draw=draw_ct, anomaly=anomaly_ct,
        win_rate=(win_ct / max(1, win_ct + loss_ct + draw_ct)),
        adrena_mean_per_game=_mean(adrena_counts),
        adrena_pct_games_ge1=_frac([c >= 1 for c in adrena_counts]),
        munkidori_energy_guard_mean_per_game=_mean(munk_energy_guard_counts),
        munkidori_energy_guard_pct_games_ge1=_frac([c >= 1 for c in munk_energy_guard_counts]),
        munkidori_ever_energized_pct_board=_frac([g["munkidori_energized_board"] for g in games_metrics]),
        adrena_value_bench_share=_ratio(
            [g["n_adrena_target_bench"] for g in games_metrics],
            [g["n_adrena_target_fallback"] for g in games_metrics],
        ),
        stadium_replace_share=_ratio(
            [g["n_stadium_replace"] for g in games_metrics],
            [g["n_stadium_fresh"] for g in games_metrics],
        ),
        munkidori_retreat_mean_per_game=_mean(munk_retreat_counts),
        grimmsnarl_ex_648_fielded_pct=_frac([g["grimmsnarl_ex_fielded"] for g in games_metrics]),
        evolve_grimmsnarl_dec_line_pct=_frac([g["n_evolve_grimmsnarl_dec"] >= 1 for g in games_metrics]),
        marnie_line_present_pct=_frac([g["marnie_line_present"] for g in games_metrics]),
        matchup_stable_pct=_frac([g["matchup_stable"] for g in games_metrics]),
        distinct_nongeneric_matchups_seen=sorted({m for g in games_metrics for m in g["distinct_nongeneric_matchups"]}),
        harness_exceptions=harness_exc,
        internal_exception_count_total=internal_exc_total,
        moves_median=statistics.median([g["moves"] for g in games_metrics]) if games_metrics else 0,
    )


# ---------------------------------------------------------------------------
# Run specs + verdict thresholds (per docs/gold/v062_expert_playbook_lines.md)
# ---------------------------------------------------------------------------

RUN_SPECS = [
    dict(name="vs_crustle", opp_kind="submission", opp_spec="v009a_crustle",
         adrena_mean_warn=7.0, adrena_pct_ge1_warn=0.90,
         expert_note="expert playbook: mean 10.68 (W) / 6.86 (L) activations/game; "
                      "pct>=1activation 99.0% (W) / 86.5% (L)"),
    dict(name="vs_alakazam", opp_kind="submission", opp_spec="v017_alakazam_heur",
         adrena_mean_warn=2.5, require_munkidori_energy_guard=True,
         expert_note="expert playbook: mean 4.21 (W) / 2.15 (L) activations/game -- "
                      "starkest per-matchup correlate in the whole dataset"),
    dict(name="vs_mirror", opp_kind="submission", opp_spec="v061_grimmsnarl_heuristic",
         adrena_mean_warn=4.0,
         expert_note="expert playbook: mean 6.92 (W) / 3.55 (L) activations/game"),
    dict(name="vs_lucario_greedy", opp_kind="meta", opp_spec="MegaLucario_live",
         adrena_mean_warn=None,
         expert_note="no per-matchup adrena threshold specified for this proxy in the "
                      "task spec (n мало in playbook too) -- universal checks only"),
    dict(name="vs_trmewtwo_greedy", opp_kind="meta", opp_spec="TRMewtwo_live",
         adrena_mean_warn=None,
         expert_note="no per-matchup adrena threshold specified for this proxy in the "
                      "task spec -- universal checks only (playbook tr_mewtwo bucket is "
                      "the real v009a-style archetype, this greedy proxy on the deck list "
                      "is a board/curve proxy, not a like-for-like AI comparison)"),
]

UNIVERSAL_GRIMMSNARL_EX_WARN = 0.85
UNIVERSAL_ADRENA_BENCH_SHARE_WARN = 0.50
UNIVERSAL_MARNIE_LINE_FAIL = 0.70
CRUSTLE_ADRENA_HARD_FAIL = 3.0

# Stable per-run seed ordinal, keyed by run NAME (not list position) so that
# filtering via --runs does not change any run's seed sequence. Python's
# built-in hash() is process-randomized (PYTHONHASHSEED) and must NOT be used
# here — the gate's whole point is a fixed, reproducible seed.
_RUN_SEED_ORDINAL = {spec["name"]: i for i, spec in enumerate(RUN_SPECS)}


def verdict_for_run(spec: dict, agg: dict) -> tuple[str, list[str], list[str]]:
    warns: list[str] = []
    fails: list[str] = []

    if agg["marnie_line_present_pct"] < UNIVERSAL_MARNIE_LINE_FAIL:
        fails.append(
            f"646/647/648 (Marnie line) present on board in only "
            f"{agg['marnie_line_present_pct']:.1%} of games (<{UNIVERSAL_MARNIE_LINE_FAIL:.0%} FAIL floor)"
        )
    if agg["harness_exceptions"] > 0 or agg["internal_exception_count_total"] > 0:
        fails.append(
            f"exceptions>0 (harness-level={agg['harness_exceptions']} games, "
            f"internal-caught total={agg['internal_exception_count_total']})"
        )
    if spec["name"] == "vs_crustle" and agg["adrena_mean_per_game"] < CRUSTLE_ADRENA_HARD_FAIL:
        fails.append(
            f"adrena_mean_per_game {agg['adrena_mean_per_game']:.2f} < {CRUSTLE_ADRENA_HARD_FAIL} "
            f"hard FAIL floor vs crustle"
        )

    if spec.get("adrena_mean_warn") is not None and agg["adrena_mean_per_game"] < spec["adrena_mean_warn"]:
        warns.append(
            f"adrena_mean_per_game {agg['adrena_mean_per_game']:.2f} < WARN threshold "
            f"{spec['adrena_mean_warn']} ({spec.get('expert_note', '')})"
        )
    if spec.get("adrena_pct_ge1_warn") is not None and agg["adrena_pct_games_ge1"] < spec["adrena_pct_ge1_warn"]:
        warns.append(
            f"adrena_pct_games_ge1 {agg['adrena_pct_games_ge1']:.1%} < WARN threshold "
            f"{spec['adrena_pct_ge1_warn']:.0%}"
        )
    if spec.get("require_munkidori_energy_guard") and agg["munkidori_energy_guard_mean_per_game"] <= 0:
        warns.append("munkidori_energy_guard never fired in this run (first-energy-to-Munkidori "
                      "routing not exercised)")
    if agg["grimmsnarl_ex_648_fielded_pct"] < UNIVERSAL_GRIMMSNARL_EX_WARN:
        warns.append(
            f"grimmsnarl_ex_648_fielded_pct {agg['grimmsnarl_ex_648_fielded_pct']:.1%} < "
            f"universal WARN floor {UNIVERSAL_GRIMMSNARL_EX_WARN:.0%}"
        )
    if agg["adrena_value_bench_share"] is not None and agg["adrena_value_bench_share"] < UNIVERSAL_ADRENA_BENCH_SHARE_WARN:
        warns.append(
            f"adrena_value_bench_share {agg['adrena_value_bench_share']:.1%} < "
            f"universal WARN floor {UNIVERSAL_ADRENA_BENCH_SHARE_WARN:.0%}"
        )

    if fails:
        return "FAIL", warns, fails
    if warns:
        return "WARN", warns, fails
    return "PASS", warns, fails


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", default="v062_grimmsnarl_playbook")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES)
    ap.add_argument("--out", default=str(GATES_DIR / "v062_fidelity_gate.json"))
    ap.add_argument("--runs", default=None,
                     help="comma-separated subset of run names to execute (default: all)")
    args = ap.parse_args()

    cand_dir = ROOT / "submissions" / args.candidate
    assert cand_dir.exists(), f"candidate dir not found: {cand_dir}"
    cand_deck = _load_deck_csv(cand_dir / "deck.csv")

    run_specs = RUN_SPECS
    if args.runs:
        wanted = set(args.runs.split(","))
        run_specs = [s for s in RUN_SPECS if s["name"] in wanted]
        assert run_specs, f"--runs matched nothing, known: {[s['name'] for s in RUN_SPECS]}"

    report: dict = dict(
        candidate=args.candidate, n_games_per_run=args.games, seed=args.seed,
        max_moves=args.max_moves, generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        engine="cg_native (libcg.dylib, macOS arm64)",
        runs={},
    )

    t_start = time.perf_counter()
    for spec in run_specs:
        t0 = time.perf_counter()
        print(f"=== {spec['name']} (opp={spec['opp_spec']}, kind={spec['opp_kind']}) "
              f"N={args.games} ===", flush=True)
        games_metrics = []
        for gi in range(args.games):
            our_first = (gi % 2 == 0)
            seed = args.seed * 10_000 + _RUN_SEED_ORDINAL[spec["name"]] * 1000 + gi
            game = play_one_game(
                args.candidate, cand_deck, spec["opp_kind"], spec["opp_spec"],
                our_first, seed, args.max_moves,
            )
            gm = _game_metrics(game)
            games_metrics.append(gm)
            if gm["outcome"] not in ("our_win", "opp_win", "draw"):
                print(f"  [!] game {gi}: anomalous outcome={gm['outcome']!r} "
                      f"harness_exc={game['harness_exception']!r} moves={gm['moves']}", flush=True)
        agg = aggregate_run(games_metrics)
        verdict, warns, fails = verdict_for_run(spec, agg)
        dt = time.perf_counter() - t0
        print(f"  -> {verdict}  win_rate={agg['win_rate']:.2f}  "
              f"adrena_mean={agg['adrena_mean_per_game']:.2f}  "
              f"grimmsnarl_ex_pct={agg['grimmsnarl_ex_648_fielded_pct']:.1%}  "
              f"({dt:.1f}s)", flush=True)
        for w in warns:
            print(f"    WARN: {w}", flush=True)
        for f in fails:
            print(f"    FAIL: {f}", flush=True)
        report["runs"][spec["name"]] = dict(
            spec={k: v for k, v in spec.items()},
            aggregate=agg,
            verdict=verdict, warns=warns, fails=fails,
            wall_sec=dt,
            games=games_metrics,
        )

    verdicts = [r["verdict"] for r in report["runs"].values()]
    if "FAIL" in verdicts:
        overall = "FAIL"
    elif "WARN" in verdicts:
        overall = "WARN"
    else:
        overall = "PASS"
    report["overall_verdict"] = overall
    report["wall_sec_total"] = time.perf_counter() - t_start

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print()
    print(f"=== OVERALL: {overall} ({report['wall_sec_total']:.1f}s total) ===")
    print(f"written: {args.out}")
    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
