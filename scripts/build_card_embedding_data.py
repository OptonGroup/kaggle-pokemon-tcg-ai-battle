#!/usr/bin/env python3
"""
build_card_embedding_data.py — Build a raw, per-card-id-embedding-ready BC dataset
from top-expert replays, for the card-embedding IL pipeline (PLAN_GOLD bet: BC
feature ceiling is the LightGBM 61-feature encoder collapsing card identity down
to a few hand-picked flags; kiyotah's official EmbeddingBag+Transformer encoder/
decoder (scratchpad/gold/kiyotah_cell1.py, reused verbatim in
scripts/cardemb_common.py) learns per-card-id representations instead).

Input:
  data/il/meta_2026-07-12_manifest.json — winner-side (episodeId, player_index)
    records with mu >= mu_threshold (an "expert" pool), each already tagged
    is_val (split by EPISODE, done upstream in build_il_meta_dataset.py — reused
    here as-is to avoid re-deriving the split and to keep it consistent with the
    existing LightGBM val set for comparability).
  data/episodes/all_teams/<replay_file> — full JSON replay per record.

For each record, for each of the winner's own decisions in that replay where:
  - obs.select is not None
  - select.context not in {IS_FIRST(41), MULLIGAN(42)}   (trivial coin-flip / yes-no)
  - select.maxCount == 1                                  (single-choice; see NOTE)
  - 2 <= len(select.option) <= MAX_OPTIONS
  - the next step's action for this player is non-empty, in-range, and not a
    60-card deck submission
we build:
  - sv_enc = get_encoder_input(obs, your_deck_approx)     (kiyotah's encoder)
  - actions = [[0], [1], ..., [n_options-1]]  (classification over options)
  - sv_dec = get_decoder_input(obs, actions)              (kiyotah's decoder)
  - label = index of the expert's chosen option within `actions`

NOTE on scope: this first pass only covers maxCount == 1 decisions (per the
task brief, "for simplicity" — MAIN action choice, ATTACK choice, PLAY, RETREAT,
CARD-select-one, etc.). Sanity run on 10 replays showed this covers ~87% of all
non-trivial decisions; the remaining ~13% (multi-select, e.g. discard 2-of-5)
would need a combinatorial `actions` list like kiyotah's create_node() and are
left for a follow-up.

NOTE on your_deck: we do NOT have the winner's exact 60-card decklist. We
approximate it as the set of *distinct-by-serial* cards seen in the winner's
own active/bench/hand/discard/prize zones across the WHOLE replay (own-side
zones are always fully visible to their owner; using serial-not-cardId dedup
avoids re-counting the same physical card at every turn snapshot). This is an
approximation (may under-count cards never drawn/revealed) — flagged as a
limitation in the report, not a correctness bug in the encoder itself.

Output:
  data/il/cardemb/train.pt   — list[dict] (raw sparse samples, plain Python
                                lists/ints/floats — no custom classes, so it can
                                be torch.load()-ed anywhere without this repo on
                                the path)
  data/il/cardemb/val.pt     — same schema, val split
  data/il/cardemb/meta.json  — dataset stats + the constants needed to build
                                MyModel (encoder_size, decoder_size, ...)

Usage:
  .venv/bin/python scripts/build_card_embedding_data.py --limit 10   # sanity
  .venv/bin/python scripts/build_card_embedding_data.py              # full run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402

import cardemb_common as cc  # noqa: E402
from cg_native.api import to_observation_class  # noqa: E402

MANIFEST_PATH = REPO / "data" / "il" / "meta_2026-07-12_manifest.json"
REPLAY_DIR = REPO / "data" / "episodes" / "all_teams"
OUT_DIR = REPO / "data" / "il" / "cardemb"

_SKIP_CONTEXTS = {41, 42}  # IS_FIRST, MULLIGAN
MAX_OPTIONS = 48  # observed max on a 10-replay sanity sample was 31; headroom above that


def own_deck_approx(replay: dict, player_index: int) -> list[int]:
    """Approximate a player's decklist as distinct-by-serial cards seen across
    the whole replay in their own active/bench/hand/discard/prize zones."""
    seen: dict[int, int] = {}  # serial -> cardId

    def _note(card):
        if isinstance(card, dict) and "id" in card:
            seen[card.get("serial")] = card["id"]

    steps = replay.get("steps", [])
    for step in steps:
        if player_index >= len(step):
            continue
        obs = step[player_index].get("observation") or {}
        cur = obs.get("current")
        if not cur:
            continue
        players = cur.get("players", [])
        if player_index >= len(players):
            continue
        ps = players[player_index]
        for card in ps.get("active") or []:
            _note(card)
            if isinstance(card, dict):
                for ec in card.get("energyCards") or []:
                    _note(ec)
                for tl in card.get("tools") or []:
                    _note(tl)
        for slot in ps.get("bench") or []:
            _note(slot)
            if isinstance(slot, dict):
                for ec in slot.get("energyCards") or []:
                    _note(ec)
                for tl in slot.get("tools") or []:
                    _note(tl)
        for card in ps.get("hand") or []:
            _note(card)
        for card in ps.get("discard") or []:
            _note(card)
        for card in ps.get("prize") or []:
            _note(card)

    return list(seen.values())


def build_samples_for_record(replay: dict, record: dict, archetype_id_map: dict) -> tuple[list[dict], Counter, Counter]:
    """Return (samples, context_counter, error_counter) for one (episode, winner) record."""
    player_index = record["player_index"]
    your_deck = own_deck_approx(replay, player_index)
    archetype_id = archetype_id_map.get(record["archetype"], archetype_id_map.get("unknown", 10))

    steps = replay.get("steps", [])
    samples: list[dict] = []
    ctx_counter: Counter = Counter()
    err_counter: Counter = Counter()

    n_steps = len(steps)
    for i in range(n_steps - 1):
        step = steps[i]
        if player_index >= len(step):
            continue
        obs_dict = step[player_index].get("observation")
        if obs_dict is None:
            continue
        sel = obs_dict.get("select")
        if sel is None:
            continue
        ctx = sel.get("context", -1)
        if ctx in _SKIP_CONTEXTS:
            continue
        if sel.get("maxCount") != 1:
            continue
        opts = sel.get("option", [])
        n_opts = len(opts)
        if n_opts < 2 or n_opts > MAX_OPTIONS:
            continue

        next_step = steps[i + 1]
        if player_index >= len(next_step):
            continue
        action = next_step[player_index].get("action", [])
        if not action:
            continue
        if len(action) == 60 and all(isinstance(x, int) for x in action):
            continue
        if not all(isinstance(x, int) and 0 <= x < n_opts for x in action):
            continue
        if len(action) != 1:
            # maxCount == 1 should always yield a single chosen index; guard anyway
            continue
        label = action[0]

        try:
            obs = to_observation_class(obs_dict)
            sv_enc = cc.get_encoder_input(obs, your_deck)
            actions = [[k] for k in range(n_opts)]
            sv_dec = cc.get_decoder_input(obs, actions)
        except Exception as e:  # noqa: BLE001 — log and skip, keep building the rest
            err_counter[f"{type(e).__name__}: {str(e)[:100]}"] += 1
            continue

        ctx_counter[ctx] += 1
        samples.append(
            {
                "enc_index": sv_enc.index,
                "enc_value": sv_enc.value,
                "enc_offset": sv_enc.offset,
                "dec_index": sv_dec.index,
                "dec_value": sv_dec.value,
                "dec_offset": sv_dec.offset,
                "n_options": n_opts,
                "label": label,
                "mu": record["mu"],
                "archetype_id": archetype_id,
                "select_context": ctx,
                "episodeId": record["episodeId"],
                "player_index": player_index,
            }
        )

    return samples, ctx_counter, err_counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N manifest records (sanity mode)")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--replay-dir", type=Path, default=REPLAY_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    archetype_id_map = manifest["archetype_id_map"]
    records = manifest["records"]
    if args.limit:
        records = records[: args.limit]

    print(f"Manifest: {len(records)} records (mu_threshold={manifest['mu_threshold']})")

    train_samples: list[dict] = []
    val_samples: list[dict] = []
    all_ctx: Counter = Counter()
    all_err: Counter = Counter()
    n_episodes_ok = 0
    n_episodes_missing = 0

    t0 = time.time()
    for idx, record in enumerate(records):
        replay_path = args.replay_dir / record["replay_file"]
        if not replay_path.exists():
            n_episodes_missing += 1
            continue
        with open(replay_path) as f:
            replay = json.load(f)

        samples, ctx_counter, err_counter = build_samples_for_record(replay, record, archetype_id_map)
        all_ctx.update(ctx_counter)
        all_err.update(err_counter)
        n_episodes_ok += 1

        if record["is_val"]:
            val_samples.extend(samples)
        else:
            train_samples.extend(samples)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(records):
            elapsed = time.time() - t0
            print(
                f"  [{idx + 1}/{len(records)}] episodes processed, "
                f"train={len(train_samples)} val={len(val_samples)} samples so far "
                f"({elapsed:.1f}s)"
            )

    if n_episodes_missing:
        print(f"WARNING: {n_episodes_missing} replay files from the manifest were not found under {args.replay_dir}")
    if all_err:
        print(f"Encoder/decoder errors (sample skipped, non-fatal): {sum(all_err.values())} total")
        for k, v in all_err.most_common(10):
            print(f"    {v:5d}  {k}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_samples, args.out_dir / "train.pt")
    torch.save(val_samples, args.out_dir / "val.pt")

    def _random_baseline(samples: list[dict]) -> float:
        if not samples:
            return 0.0
        return sum(1.0 / s["n_options"] for s in samples) / len(samples)

    all_samples = train_samples + val_samples
    n_options_list = [s["n_options"] for s in all_samples]
    avg_n_options = sum(n_options_list) / len(n_options_list) if n_options_list else 0.0

    meta = {
        "built_at": "2026-07-12",
        "source_manifest": str(args.manifest),
        "mu_threshold": manifest["mu_threshold"],
        "archetype_id_map": archetype_id_map,
        "n_manifest_records": len(records),
        "n_episodes_processed": n_episodes_ok,
        "n_episodes_missing_replay": n_episodes_missing,
        "n_train_samples": len(train_samples),
        "n_val_samples": len(val_samples),
        "avg_options_per_decision": avg_n_options,
        "random_baseline_acc_mean_of_inverse": _random_baseline(all_samples),
        "random_baseline_acc_inverse_of_mean": (1.0 / avg_n_options) if avg_n_options else 0.0,
        "random_baseline_acc_train": _random_baseline(train_samples),
        "random_baseline_acc_val": _random_baseline(val_samples),
        "context_counts": {str(k): v for k, v in all_ctx.most_common()},
        "max_options_cap": MAX_OPTIONS,
        "model_constants": {
            "card_count": cc.CARD_COUNT,
            "attack_count": cc.ATTACK_COUNT,
            "num_words_encoder": cc.num_words_encoder,
            "encoder_size": cc.encoder_size,
            "decoder_size": cc.decoder_size,
        },
        "elapsed_sec": time.time() - t0,
    }
    with open(args.out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== DONE ===")
    print(f"  train samples: {len(train_samples)}   val samples: {len(val_samples)}")
    print(f"  avg options/decision: {avg_n_options:.2f}")
    print(f"  random baseline (mean of 1/n_options): {meta['random_baseline_acc_mean_of_inverse']:.4f}")
    print(f"  random baseline (1/mean n_options):    {meta['random_baseline_acc_inverse_of_mean']:.4f}")
    print(f"  wrote {args.out_dir / 'train.pt'}, val.pt, meta.json")


if __name__ == "__main__":
    main()
