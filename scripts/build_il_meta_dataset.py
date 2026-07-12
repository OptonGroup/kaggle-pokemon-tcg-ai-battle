#!/usr/bin/env python3
"""
build_il_meta_dataset.py — Build a FRESH, multi-archetype IL dataset reflecting the
CURRENT metagame (2026-07-12 leaderboard snapshot), for card-embedding NN training
(PLAN_TOP20.md Phase 0.5) and as a drop-in refresh for the existing LightGBM IL pipeline.

Rationale (see docs/gold/PLAN_TOP20.md, PLAN_GOLD.md, and this task's report):
  - The existing data/il/train.npz (built 2026-06-17) reflects the Crustle-mirror era
    meta and is stale; the meta has rotated twice since (07-03, 07-11).
  - "1 strong expert > crowd" is a rule calibrated for LightGBM with hand-engineered
    features on ONE archetype. For a per-card-id embedding encoder (archetype-agnostic
    by construction), diversity across many CURRENTLY strong experts/archetypes is the
    better prior — reinforced empirically here: aidy (Alakazam expert, recon'd 07-11 at
    mu=1127) had fallen to mu=959/rank179 by 07-12; keidroid (Starmie #1 in the 07-11
    recon) had fallen to mu=725/rank1895. Single-pilot targeting is fragile to exactly
    this kind of day-to-day top-of-ladder volatility documented in PLAN_GOLD/PLAN_TOP20.
  - So: build ONE multi-archetype dataset filtered to (winner-side AND acting player's
    CURRENT team mu >= EXPERT_MU_THRESHOLD), using the existing generic (archetype-
    agnostic) feature builder in src/il/features.py, across the full local replay corpus
    (data/episodes/ + data/episodes/all_teams/, ~2200 files after this task's harvest).

Outputs (does NOT touch the old data/il/train.npz / val.npz):
  data/il/train_meta_2026-07-12.npz   (X, y, groups, weights, ctx, mu, archetype_id)
  data/il/val_meta_2026-07-12.npz     (same schema)
  data/il/meta_2026-07-12_manifest.json  — per-qualifying-(episode,player) record:
      {episodeId, player_index, player_name, teamName, mu, rank, archetype,
       opp_archetype, n_pairs_main, n_pairs_total, is_val}
    This manifest + the raw replay JSONs themselves are the "raw" artifact a future
    card-embedding trainer should use directly (via src.replays.parse.parse_replay),
    since features.py collapses card identity down to a couple of Crustle-only flags
    and is NOT sufficient input for a per-card-id encoder.

Usage:
  .venv/bin/python scripts/build_il_meta_dataset.py [--mu-threshold 1000] [--val-frac 0.15]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

from replays.parse import parse_replay, walk_replays  # noqa: E402
from il.features import build_feature_matrix, FEATURE_DIM  # noqa: E402

LEADERBOARD_CSV = REPO / "docs" / "gold" / "leaderboard_2026-07-12.csv"
OUT_DIR = REPO / "data" / "il"
REPLAY_DIRS = [REPO / "data" / "episodes" / "all_teams", REPO / "data" / "episodes"]

SKIP_CONTEXTS = {41, 42}  # IS_FIRST, MULLIGAN
WINNER_WEIGHT = 2.0
VAL_FRAC = 0.15

# Archetype signatures (Pokemon-only card-id sets; see task investigation for how
# these were derived from decks/meta/*.csv + data/EN_Card_Data.csv).
ARCHETYPE_SIGS = {
    "Crustle": {344, 345},
    "Alakazam": {741, 742, 743},
    "Starmie": {1030, 1031},
    "Lucario": {673, 674, 675, 676, 677, 678},
    "Munkidori/Grimmsnarl": {112},
    "Cubchoo": {506},
    "N_Zoroark": {292, 293},
    "Ogerpon": {96, 272},
    "Garchomp": {379, 380, 341, 342},
    "FestivalGrookey": {89, 90, 92, 93},
}


def load_uname2team(path: Path) -> dict[str, tuple[str, float, int]]:
    """
    Map replay info.Agents[i].Name -> (teamName, score, rank).

    IMPORTANT: replay Agents[].Name is the Kaggle TEAM NAME, not the individual
    Kaggle username (verified empirically: e.g. team "Yushin Ito" / member
    username "yushin505" appears in replays as "Yushin Ito"). TeamName is the
    primary key; individual usernames are also indexed as a fallback for the
    rarer case where a solo player's team name equals their username (which is
    common enough it worked "by luck" for names like tonakaiiii/iwashi/aidy).
    """
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                score = float(row["Score"])
                rank = int(row["Rank"])
            except (ValueError, KeyError):
                continue
            team_name = row["TeamName"]
            out.setdefault(team_name, (team_name, score, rank))
            for uname in row["TeamMemberUserNames"].split(","):
                uname = uname.strip()
                if uname:
                    out.setdefault(uname, (team_name, score, rank))
    return out


def guess_archetype(deck_ids: list[int]) -> str:
    s = set(deck_ids)
    hits = [name for name, sig in ARCHETYPE_SIGS.items() if sig.issubset(s)]
    return "+".join(hits) if hits else "unknown"


def extract_decks(data: dict) -> dict[int, list[int]]:
    steps = data.get("steps", [])
    if len(steps) < 2:
        return {}
    decks = {}
    for pi in range(min(2, len(steps[1]))):
        action = steps[1][pi].get("action", [])
        if len(action) == 60 and all(isinstance(x, int) for x in action):
            decks[pi] = action
    return decks


def episode_to_val_flag(episode_id, val_frac: float = VAL_FRAC) -> bool:
    h = int(hashlib.md5(str(episode_id).encode()).hexdigest(), 16)
    return (h % 100) < int(val_frac * 100)


def get_all_unique_replays() -> list[Path]:
    raw = walk_replays(REPLAY_DIRS)
    seen: set[str] = set()
    unique: list[Path] = []
    for p in raw:
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p)
    return unique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu-threshold", type=float, default=1000.0)
    ap.add_argument(
        "--both-sides-mu-threshold",
        type=float,
        default=None,
        help=(
            "If set, ALSO include the LOSING side's decisions in a game when BOTH "
            "players' mu >= this threshold (losing to another elite player is still "
            "expert-quality play). Default None = winner-side only (original behavior)."
        ),
    )
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    ap.add_argument("--tag", default="2026-07-12")
    args = ap.parse_args()

    print("=" * 70)
    print(f"  Building multi-archetype meta IL dataset (mu >= {args.mu_threshold})")
    print("=" * 70)

    uname2team = load_uname2team(LEADERBOARD_CSV)
    print(f"[1/4] Leaderboard usernames loaded: {len(uname2team)}")

    paths = get_all_unique_replays()
    print(f"[2/4] Unique replay files found: {len(paths)}")

    train_X, train_y, train_g, train_w, train_ctx, train_mu = [], [], [], [], [], []
    val_X, val_y, val_g, val_w, val_ctx, val_mu = [], [], [], [], [], []
    train_arch, val_arch = [], []

    archetypes_all = sorted(ARCHETYPE_SIGS.keys()) + ["unknown"]
    arch2id = {a: i for i, a in enumerate(archetypes_all)}

    manifest = []
    seen_episode_player = set()

    n_files = 0
    n_parse_err = 0
    n_qualifying_games = 0
    n_qualifying_winner = 0
    n_qualifying_loser_strong_opp = 0
    total_pairs = 0
    total_pairs_main = 0
    skipped_shape = 0
    mu_distribution_by_cut = {c: 0 for c in (1150, 1100, 1050, 1000, 950, 900, 850, 800)}
    winner_expert_counter: dict[tuple, int] = {}
    opp_archetype_counter: dict[str, int] = {}

    for path in paths:
        n_files += 1
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            n_parse_err += 1
            continue

        agents = data.get("info", {}).get("Agents", [])
        rewards = data.get("rewards", [])
        ep_id = data.get("id") or data.get("info", {}).get("EpisodeId", 0)
        decks = extract_decks(data)

        # Which (player_index) in this game qualifies as "expert" play for this dataset?
        #   - WINNER side qualifies whenever their own mu >= mu_threshold (original rule).
        #   - LOSER side ALSO qualifies (relaxed 2026-07-12 for card-embedding data volume)
        #     when --both-sides-mu-threshold is set and BOTH players' mu clear it — losing
        #     to another elite player is still expert-quality play, just not the win.
        pi_info = {}
        for pi in range(min(2, len(agents))):
            name = agents[pi].get("Name")
            info = uname2team.get(name)
            if info is None:
                continue
            teamName, mu, rank = info
            pi_info[pi] = (name, teamName, mu, rank)
            for cut in mu_distribution_by_cut:
                if mu >= cut:
                    mu_distribution_by_cut[cut] += 1

        qualifying_pis = {}
        for pi, (name, teamName, mu, rank) in pi_info.items():
            if mu < args.mu_threshold:
                continue
            r = rewards[pi] if pi < len(rewards) else None
            is_winner = r is not None and r > 0
            if is_winner:
                qualifying_pis[pi] = (name, teamName, mu, rank, "winner")
                continue
            if args.both_sides_mu_threshold is not None:
                opp_mu = pi_info.get(1 - pi, (None, None, None, None))[2]
                if (
                    opp_mu is not None
                    and mu >= args.both_sides_mu_threshold
                    and opp_mu >= args.both_sides_mu_threshold
                ):
                    qualifying_pis[pi] = (name, teamName, mu, rank, "loser_strong_opp")

        if not qualifying_pis:
            continue

        try:
            pairs = parse_replay(path)
        except Exception:
            n_parse_err += 1
            continue

        is_val = episode_to_val_flag(ep_id, args.val_frac)

        for pi, (name, teamName, mu, rank, reason) in qualifying_pis.items():
            key = (ep_id, pi)
            if key in seen_episode_player:
                continue
            seen_episode_player.add(key)
            n_qualifying_games += 1
            if reason == "winner":
                n_qualifying_winner += 1
            else:
                n_qualifying_loser_strong_opp += 1

            own_deck = decks.get(pi, [])
            opp_deck = decks.get(1 - pi, [])
            arch = guess_archetype(own_deck)
            opp_arch = guess_archetype(opp_deck)
            arch_id = arch2id.get(arch.split("+")[0], arch2id["unknown"])
            opp_archetype_counter[opp_arch] = opp_archetype_counter.get(opp_arch, 0) + 1
            winner_expert_counter[(name, teamName, mu, rank)] = (
                winner_expert_counter.get((name, teamName, mu, rank), 0) + 1
            )

            n_pairs_this = 0
            n_pairs_main_this = 0
            for pair in pairs:
                if pair["player_index"] != pi:
                    continue
                ctx = pair["select_context"]
                if ctx in SKIP_CONTEXTS:
                    continue
                n_opts = pair["n_options"]
                if n_opts == 0:
                    continue
                obs = pair["obs_dict"]
                chosen = set(pair["chosen_option_indices"])
                try:
                    X_group = build_feature_matrix(obs)
                except Exception:
                    skipped_shape += 1
                    continue
                if X_group.shape[0] == 0 or X_group.shape[0] != n_opts:
                    skipped_shape += 1
                    continue

                y_group = np.zeros(n_opts, dtype=np.int8)
                for ci in chosen:
                    if ci < n_opts:
                        y_group[ci] = 1

                n_pairs_this += 1
                total_pairs += 1
                if ctx == 0:
                    n_pairs_main_this += 1
                    total_pairs_main += 1

                target_X = val_X if is_val else train_X
                target_y = val_y if is_val else train_y
                target_g = val_g if is_val else train_g
                target_w = val_w if is_val else train_w
                target_ctx = val_ctx if is_val else train_ctx
                target_mu = val_mu if is_val else train_mu
                target_arch = val_arch if is_val else train_arch

                target_X.append(X_group)
                target_y.append(y_group)
                target_g.append(n_opts)
                target_w.append(np.full(n_opts, WINNER_WEIGHT, dtype=np.float32))
                target_ctx.append(ctx)
                target_mu.append(mu)
                target_arch.append(arch_id)

            manifest.append({
                "episodeId": ep_id,
                "player_index": pi,
                "player_name": name,
                "teamName": teamName,
                "mu": mu,
                "rank": rank,
                "archetype": arch,
                "opp_archetype": opp_arch,
                "n_pairs_main": n_pairs_main_this,
                "n_pairs_total": n_pairs_this,
                "is_val": is_val,
                "replay_file": path.name,
                "is_winner": reason == "winner",
                "qualify_reason": reason,
            })

    print(f"[3/4] Scanned {n_files} files ({n_parse_err} parse errors)")
    print(
        f"  Qualifying (episode, player) @ mu>={args.mu_threshold}: {n_qualifying_games} total "
        f"(winner={n_qualifying_winner}, loser_strong_opp={n_qualifying_loser_strong_opp}, "
        f"both_sides_mu_threshold={args.both_sides_mu_threshold})"
    )
    print(f"  Total (obs,action) pairs collected: {total_pairs}  (MAIN-only: {total_pairs_main})")
    print(f"  Skipped (shape mismatch): {skipped_shape}")
    print(f"\n  Mu-threshold sensitivity (qualifying WINNER game-instances, cumulative):")
    for cut in sorted(mu_distribution_by_cut, reverse=True):
        print(f"    mu>={cut:5.0f}: {mu_distribution_by_cut[cut]:5d}")

    print(f"\n  Winner-expert breakdown (name/team/mu/rank -> qualifying games):")
    for k, c in sorted(winner_expert_counter.items(), key=lambda kv: -kv[0][2]):
        name, teamName, mu, rank = k
        print(f"    {name:20s} team={teamName:20s} mu={mu:7.1f} rank={rank:5d}  games={c}")

    print(f"\n  Opponent archetype distribution (of qualifying games):")
    for a, c in sorted(opp_archetype_counter.items(), key=lambda kv: -kv[1]):
        print(f"    {a:25s} {c}")

    def stack(Xs, ys, gs, ws, ctxs, mus, archs):
        X = np.vstack(Xs) if Xs else np.zeros((0, FEATURE_DIM), dtype=np.float32)
        y = np.concatenate(ys) if ys else np.zeros(0, dtype=np.int8)
        g = np.array(gs, dtype=np.int32)
        w = np.concatenate(ws) if ws else np.zeros(0, dtype=np.float32)
        c = np.array(ctxs, dtype=np.int32)
        m = np.array(mus, dtype=np.float32)  # per-GROUP mu (len == len(g))
        a = np.array(archs, dtype=np.int32)  # per-GROUP archetype id (len == len(g))
        return X, y, g, w, c, m, a

    Xtr, ytr, gtr, wtr, ctxtr, mutr, artr = stack(
        train_X, train_y, train_g, train_w, train_ctx, train_mu, train_arch)
    Xva, yva, gva, wva, ctxva, muva, arva = stack(
        val_X, val_y, val_g, val_w, val_ctx, val_mu, val_arch)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / f"train_meta_{args.tag}.npz"
    val_path = OUT_DIR / f"val_meta_{args.tag}.npz"
    manifest_path = OUT_DIR / f"meta_{args.tag}_manifest.json"

    np.savez_compressed(train_path, X=Xtr, y=ytr, groups=gtr, weights=wtr, ctx=ctxtr,
                         mu=mutr, archetype_id=artr)
    np.savez_compressed(val_path, X=Xva, y=yva, groups=gva, weights=wva, ctx=ctxva,
                         mu=muva, archetype_id=arva)
    with open(manifest_path, "w") as f:
        json.dump({
            "mu_threshold": args.mu_threshold,
            "both_sides_mu_threshold": args.both_sides_mu_threshold,
            "n_qualifying_winner": n_qualifying_winner,
            "n_qualifying_loser_strong_opp": n_qualifying_loser_strong_opp,
            "archetype_id_map": arch2id,
            "n_qualifying_games": n_qualifying_games,
            "records": manifest,
        }, f, indent=1)

    print(f"\n[4/4] Saved:")
    print(f"  {train_path}  (groups={len(gtr)}, rows={len(Xtr)})")
    print(f"  {val_path}  (groups={len(gva)}, rows={len(Xva)})")
    print(f"  {manifest_path}  ({len(manifest)} records)")


if __name__ == "__main__":
    main()
