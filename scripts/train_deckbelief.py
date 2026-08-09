#!/usr/bin/env python3
"""
train_deckbelief.py — two-stage training for DeckBeliefNet (scripts/deckbelief_common.py):

  1) Synthetic co-occurrence pretraining, generated on the fly from the REAL
     60-card decks in data/belief/corpus_decks_{train,val}.pt (singleton, pair,
     and random-subset partial reveals, mirroring the notebook this is adapted
     from) — teaches which cards tend to appear together, without depending on
     however many distinct real decks happen to exist locally.
  2) Real-snapshot fine-tuning on data/belief/{train,val}.pt (actual milestone
     snapshots from real games) — teaches the realistic, non-random ORDER in
     which cards become visible during a game.

Runs on CPU by default. The model is a ~715K-param MLP — deliberately small
enough that this doesn't need the GPU-rental workflow the project's heavier
BC/Transformer pipeline (scripts/train_card_embedding.py) required.

Usage:
  .venv/bin/python scripts/train_deckbelief.py
  .venv/bin/python scripts/train_deckbelief.py --epochs-synthetic 5 --epochs-finetune 10
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import torch
from torch.utils.data import Dataset, DataLoader

from deckbelief_common import DeckBeliefNet, N_CARDS, build_card_caps_and_ace_mask


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


class TensorBundleDataset(Dataset):
    """Wraps a {"revealed":..., "true_deck":...} tensor bundle (see
    build_deck_belief_data.py) as a torch Dataset of (revealed, true_deck) pairs."""

    def __init__(self, bundle: dict):
        self.revealed = bundle["revealed"]
        self.true_deck = bundle["true_deck"]

    def __len__(self):
        return self.revealed.shape[0]

    def __getitem__(self, i):
        return self.revealed[i], self.true_deck[i]


def make_synthetic_samples(decks: torch.Tensor, seed: int) -> list[tuple[dict, list[int]]]:
    """
    For each 60-card deck (row of `decks`), generate partial-reveal samples:
    every single card, every pair of distinct cards, and 2 random samples each
    at reveal sizes 4/8/12/20 — mirrors the source notebook's co-occurrence
    recipe. Returns list of (revealed_counts_dict, full_deck_list).
    """
    rng = random.Random(seed)
    rows = []
    for deck_row in decks.tolist():
        unique_ids = sorted(set(deck_row))
        for cid in unique_ids:
            rows.append(({cid: 1}, deck_row))
        for a, b in itertools.combinations(unique_ids, 2):
            rows.append(({a: 1, b: 1}, deck_row))
        for reveal_size in (4, 8, 12, 20):
            if reveal_size > len(deck_row):
                continue
            for _ in range(2):
                sample = rng.sample(deck_row, reveal_size)
                counts: dict = {}
                for cid in sample:
                    counts[cid] = counts.get(cid, 0) + 1
                rows.append((counts, deck_row))
    return rows


class SyntheticDataset(Dataset):
    def __init__(self, rows: list[tuple[dict, list[int]]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        counts, deck = self.rows[i]
        revealed = torch.zeros(N_CARDS)
        for cid, cnt in counts.items():
            revealed[cid] = cnt
        true_deck = torch.zeros(N_CARDS)
        for cid in deck:
            true_deck[cid] += 1
        return revealed, true_deck


def fit(model, opt, train_loader, val_loader, epochs, patience, label, device):
    best_loss, best_state, stale = float("inf"), None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = l1_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                total += l1_loss(pred, y).item() * len(x)
                n += len(x)
        val_loss = total / max(1, n)
        history.append(val_loss)
        print(f"  [{label}] epoch {epoch + 1:3d}/{epochs}: val_L1={val_loss:.4f}")

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            print(f"  [{label}] early stop at epoch {epoch + 1} (best val_L1={best_loss:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_loss


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/belief")
    ap.add_argument("--out-dir", default="out/deckbelief")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs-synthetic", type=int, default=5)
    ap.add_argument("--epochs-finetune", type=int, default=10)
    ap.add_argument("--lr-synthetic", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--synthetic-cap", type=int, default=40000,
                    help="cap on synthetic train samples per epoch (random subsample if the generated pool is larger)")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260719)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    data_dir = REPO / args.data_dir
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    card_caps, ace_mask = build_card_caps_and_ace_mask(csv_path=str(REPO / "data" / "EN_Card_Data.csv"))
    model = DeckBeliefNet(ace_mask, card_caps).to(device)

    # ---- Stage 1: synthetic co-occurrence pretraining ----
    train_corpus = torch.load(data_dir / "corpus_decks_train.pt", weights_only=True)
    val_corpus = torch.load(data_dir / "corpus_decks_val.pt", weights_only=True)
    print(f"Synthetic corpus: {train_corpus.shape[0]} train decks, {val_corpus.shape[0]} val decks")

    synth_train_rows = make_synthetic_samples(train_corpus, args.seed)
    synth_val_rows = make_synthetic_samples(val_corpus, args.seed + 1)
    print(f"Synthetic samples: {len(synth_train_rows)} train (pre-cap), {len(synth_val_rows)} val (pre-cap)")

    if args.synthetic_cap and len(synth_train_rows) > args.synthetic_cap:
        synth_train_rows = random.Random(args.seed).sample(synth_train_rows, args.synthetic_cap)
    val_cap = max(1, args.synthetic_cap // 4) if args.synthetic_cap else None
    if val_cap and len(synth_val_rows) > val_cap:
        synth_val_rows = random.Random(args.seed + 1).sample(synth_val_rows, val_cap)

    synth_train_loader = DataLoader(SyntheticDataset(synth_train_rows), batch_size=args.batch_size, shuffle=True)
    synth_val_loader = DataLoader(SyntheticDataset(synth_val_rows), batch_size=args.batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_synthetic, weight_decay=1e-4)
    t0 = time.time()
    synth_history, synth_best = fit(model, opt, synth_train_loader, synth_val_loader,
                                     args.epochs_synthetic, args.patience, "synthetic", device)
    print(f"Stage 1 (synthetic) done in {time.time() - t0:.1f}s, best val_L1={synth_best:.4f}")

    # ---- Stage 2: real-snapshot fine-tuning ----
    train_bundle = torch.load(data_dir / "train.pt", weights_only=True)
    val_bundle = torch.load(data_dir / "val.pt", weights_only=True)
    print(f"Real snapshots: {train_bundle['revealed'].shape[0]} train, {val_bundle['revealed'].shape[0]} val")

    real_train_loader = DataLoader(TensorBundleDataset(train_bundle), batch_size=args.batch_size, shuffle=True)
    real_val_loader = DataLoader(TensorBundleDataset(val_bundle), batch_size=args.batch_size)

    opt2 = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune, weight_decay=1e-4)
    t1 = time.time()
    finetune_history, finetune_best = fit(model, opt2, real_train_loader, real_val_loader,
                                           args.epochs_finetune, args.patience, "finetune", device)
    print(f"Stage 2 (finetune) done in {time.time() - t1:.1f}s, best val_L1={finetune_best:.4f}")

    torch.save(model.state_dict(), out_dir / "model.pth")
    meta = {
        "n_cards": N_CARDS,
        "synthetic_history": synth_history,
        "synthetic_best_val_l1": synth_best,
        "finetune_history": finetune_history,
        "finetune_best_val_l1": finetune_best,
        "args": vars(args),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {out_dir}/model.pth + meta.json")


if __name__ == "__main__":
    main()
