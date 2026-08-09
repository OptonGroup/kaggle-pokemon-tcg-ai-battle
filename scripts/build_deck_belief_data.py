#!/usr/bin/env python3
"""
build_deck_belief_data.py — build a train/val dataset for DeckBeliefNet
(scripts/deckbelief_common.py): predicting an opponent's full 60-card deck
from currently-revealed cards.

Sources (see PLAN §Opponent-Deck-Belief for the full writeup):
  - data/episodes/official/{date}/*.json — organizer's daily episode dataset
    (kaggle/pokemon-tcg-ai-battle-episodes-{date}, CC0-1.0), downloaded via
    `kaggle datasets download`. Primary source — much larger/more diverse
    than our own submissions' games.
  - data/episodes/{all_teams,top_players} + loose (*-replay.json) — our own
    account's games, secondary/supplementary source.

For each replay file, for each player_index in {0,1}:
  - true_decks_from_data(data)[1 - player_index] is the ground-truth 60-card
    deck of that player's OPPONENT (steps[1][player].action — not redacted in
    a replay file, see src/replays/parse.py).
  - opponent_reveal_progression_from_data(data, milestones) gives milestone
    snapshots of the opponent's revealed card counts (deduped by (id, serial)
    — physical copies, not step-appearances).

Each (player_index, milestone) record becomes one training sample:
    revealed:   sparse dict {card_id: count}  (evidence, a hard lower bound)
    true_deck:  the opponent's real 60-card list (label)

Train/val split is by DECK SIGNATURE (the true 60-card multiset, order
independent) — not by episode or by sample — so the same physical deck list
never appears in both splits (a random split would leak: the same deck
recurs across many episodes/samples). We search over random signature
subsets for the one whose attached SAMPLE COUNT is closest to the target
val fraction (signatures don't all carry the same number of samples).

Output (data/belief/):
  train.pt, val.pt   — dict of tensors (revealed, true_deck: FloatTensor
                        (N, N_CARDS); milestone, episode_id, player_index:
                        LongTensor (N,)) — plain tensors, no custom classes,
                        so torch.load() works anywhere without this repo on
                        the path (same property as data/il/cardemb/*.pt).
  corpus_decks_train.pt, corpus_decks_val.pt
                     — LongTensor (K, 60): unique 60-card deck lists per
                        split, used by scripts/train_deckbelief.py to
                        generate synthetic partial-reveal samples (co-
                        occurrence pretraining stage).
  meta.json          — counts, milestones, ace_spec_ids (from EN_Card_Data.csv,
                        for reproducibility — the model itself derives this
                        fresh via deckbelief_common.build_card_caps_and_ace_mask).

Usage:
  .venv/bin/python scripts/build_deck_belief_data.py --limit 200   # sanity
  .venv/bin/python scripts/build_deck_belief_data.py               # full run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from replays.parse import (  # noqa: E402
    true_decks_from_data,
    opponent_reveal_progression_from_data,
    walk_replays,
)
from deckbelief_common import N_CARDS, build_card_caps_and_ace_mask  # noqa: E402

MILESTONES = (1, 2, 4, 8, 12, 20)


def gather_replay_paths(official_root: Path, local_dirs: list[Path]) -> list[Path]:
    official_paths = sorted(official_root.glob("*/*.json")) if official_root.is_dir() else []
    local_paths = walk_replays(local_dirs)
    # De-dup by filename stem in case the same episode was ever pulled into
    # both the official dump and our own account's cache.
    seen_stems = set()
    out = []
    for p in official_paths + local_paths:
        stem = p.stem.split("-")[0]  # "86586905" from either "86586905" or "episode-86586905-replay"
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        out.append(p)
    return out


def _process_one(path_str: str) -> tuple[list[dict], int, int]:
    """Worker: parse one replay file, return (samples, n_decks_ok, n_parse_errors)."""
    try:
        with open(path_str) as f:
            data = json.load(f)
    except Exception:
        return [], 0, 1

    decks = true_decks_from_data(data)
    if len(decks) < 2:
        return [], 0, 0

    # data["id"] is sometimes a UUID string (official daily dataset); prefer the
    # integer info.EpisodeId, falling back to data["id"] only if it parses as int.
    episode_id = data.get("info", {}).get("EpisodeId")
    if episode_id is None:
        raw_id = data.get("id")
        episode_id = raw_id if isinstance(raw_id, int) else 0
    try:
        records = opponent_reveal_progression_from_data(data, MILESTONES)
    except Exception:
        return [], 1, 1

    samples = []
    for rec in records:
        opp_idx = 1 - rec["player_index"]
        true_deck = decks.get(opp_idx)
        if true_deck is None:
            continue
        samples.append({
            "revealed": rec["revealed"],           # dict card_id -> count
            "true_deck": true_deck,                # list of 60 card ids
            "milestone": rec["milestone"],
            "episode_id": int(episode_id) if episode_id else 0,
            "player_index": rec["player_index"],
        })
    return samples, 1, 0


def deck_signature(deck: list[int]) -> tuple[int, ...]:
    return tuple(sorted(deck))


def split_by_signature(samples: list[dict], val_frac: float, seed: int):
    """Return (train_indices, val_indices) into `samples`, split by unique deck
    signature so no signature straddles both splits. Searches random subsets
    of signatures for the one whose SAMPLE COUNT (not signature count) is
    closest to val_frac of the total."""
    import random

    sig_of = [deck_signature(s["true_deck"]) for s in samples]
    by_sig: dict[tuple[int, ...], list[int]] = {}
    for i, sig in enumerate(sig_of):
        by_sig.setdefault(sig, []).append(i)

    sigs = list(by_sig.keys())
    rng = random.Random(seed)
    target = len(samples) * val_frac
    best_val_sigs, best_err = None, float("inf")
    for _ in range(2048):
        rng.shuffle(sigs)
        acc = 0
        chosen = []
        for sig in sigs:
            if acc >= target:
                break
            chosen.append(sig)
            acc += len(by_sig[sig])
        err = abs(acc - target)
        if err < best_err:
            best_err, best_val_sigs = err, set(chosen)
    val_sigs = best_val_sigs or set()

    train_idx, val_idx = [], []
    for i, sig in enumerate(sig_of):
        (val_idx if sig in val_sigs else train_idx).append(i)
    return train_idx, val_idx, by_sig, val_sigs


def to_tensor_bundle(samples: list[dict], indices: list[int]):
    import torch

    n = len(indices)
    revealed = torch.zeros(n, N_CARDS, dtype=torch.float32)
    true_deck = torch.zeros(n, N_CARDS, dtype=torch.float32)
    milestone = torch.zeros(n, dtype=torch.int64)
    episode_id = torch.zeros(n, dtype=torch.int64)
    player_index = torch.zeros(n, dtype=torch.int64)

    for row, i in enumerate(indices):
        s = samples[i]
        for cid, cnt in s["revealed"].items():
            if 0 <= cid < N_CARDS:
                revealed[row, cid] = cnt
        deck_counts = Counter(s["true_deck"])
        for cid, cnt in deck_counts.items():
            if 0 <= cid < N_CARDS:
                true_deck[row, cid] = cnt
        milestone[row] = s["milestone"]
        episode_id[row] = s["episode_id"]
        player_index[row] = s["player_index"]

    return {
        "revealed": revealed,
        "true_deck": true_deck,
        "milestone": milestone,
        "episode_id": episode_id,
        "player_index": player_index,
    }


def unique_decks_tensor(samples: list[dict], indices: list[int]):
    import torch

    seen = set()
    decks = []
    for i in indices:
        deck = samples[i]["true_deck"]
        sig = deck_signature(deck)
        if sig in seen:
            continue
        seen.add(sig)
        decks.append(sorted(deck))
    return torch.tensor(decks, dtype=torch.int64) if decks else torch.zeros(0, 60, dtype=torch.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--official-dir", default="data/episodes/official")
    ap.add_argument("--local-dirs", nargs="*", default=["data/episodes/all_teams", "data/episodes/top_players", "data/episodes"])
    ap.add_argument("--out-dir", default="data/belief")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260719)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N replay files (sanity run)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    official_root = REPO / args.official_dir
    local_dirs = [REPO / d for d in args.local_dirs]
    paths = gather_replay_paths(official_root, local_dirs)
    if args.limit:
        paths = paths[: args.limit]
    print(f"Found {len(paths)} unique replay files "
          f"(official={official_root}, local={[str(d) for d in local_dirs]})")

    all_samples: list[dict] = []
    n_files_ok = 0
    n_parse_errors = 0
    n_no_decks = 0
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_one, str(p)): p for p in paths}
        for fut in as_completed(futs):
            samples, decks_ok, errs = fut.result()
            n_files_ok += decks_ok
            n_parse_errors += errs
            if decks_ok and not samples:
                n_no_decks += 1
            all_samples.extend(samples)
            done += 1
            if done % 2000 == 0:
                print(f"  ...{done}/{len(paths)} files, {len(all_samples)} samples so far")

    print(f"Done parsing: {n_files_ok} files with valid decks, {n_parse_errors} parse errors, "
          f"{len(all_samples)} total samples")

    if not all_samples:
        print("No samples collected — aborting.")
        return

    train_idx, val_idx, by_sig, val_sigs = split_by_signature(all_samples, args.val_frac, args.seed)
    print(f"Split: {len(train_idx)} train samples, {len(val_idx)} val samples, "
          f"{len(by_sig)} distinct deck signatures ({len(val_sigs)} held out for val)")

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    train_bundle = to_tensor_bundle(all_samples, train_idx)
    val_bundle = to_tensor_bundle(all_samples, val_idx)
    torch.save(train_bundle, out_dir / "train.pt")
    torch.save(val_bundle, out_dir / "val.pt")

    train_corpus = unique_decks_tensor(all_samples, train_idx)
    val_corpus = unique_decks_tensor(all_samples, val_idx)
    torch.save(train_corpus, out_dir / "corpus_decks_train.pt")
    torch.save(val_corpus, out_dir / "corpus_decks_val.pt")

    _, ace_mask = build_card_caps_and_ace_mask(csv_path=str(REPO / "data" / "EN_Card_Data.csv"))
    ace_spec_ids = sorted(int(i) for i in torch.nonzero(ace_mask, as_tuple=True)[0].tolist())

    meta = {
        "built_at": __import__("datetime").datetime.now().isoformat(),
        "n_cards": N_CARDS,
        "milestones": list(MILESTONES),
        "n_files_scanned": len(paths),
        "n_files_with_decks": n_files_ok,
        "n_parse_errors": n_parse_errors,
        "n_files_no_samples": n_no_decks,
        "n_train_samples": len(train_idx),
        "n_val_samples": len(val_idx),
        "n_distinct_deck_signatures": len(by_sig),
        "n_val_deck_signatures": len(val_sigs),
        "n_train_corpus_decks": int(train_corpus.shape[0]),
        "n_val_corpus_decks": int(val_corpus.shape[0]),
        "ace_spec_ids": ace_spec_ids,
        "val_frac_target": args.val_frac,
        "seed": args.seed,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {out_dir}/{{train.pt, val.pt, corpus_decks_train.pt, corpus_decks_val.pt, meta.json}}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
