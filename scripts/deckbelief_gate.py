#!/usr/bin/env python3
"""
deckbelief_gate.py — offline numeric gate for a trained DeckBeliefNet
(out/deckbelief/model.pth by default), before it's ever wired into a live
agent. No game engine (libcg) required — pure tensor math on
data/belief/val.pt, matching this repo's value_probe.py "no engine needed"
convention.

Metrics (mirroring the source notebook's own evaluation), reported per
milestone (1/2/4/8/12/20 revealed opponent cards):
  - normalized unseen-card L1: L1 distance between predicted and true HIDDEN
    (not-yet-revealed) card counts, normalized by the true hidden total, for
    both the trained model and a "prior" baseline (population-average train
    deck, pushed through the SAME bounded-fill constraints given this
    sample's own revealed-card floor — see DeckBeliefNet.apply_constraints).
  - L1 ratio = model_L1 / prior_L1 (must be < 1, i.e. beat the naive prior,
    and should shrink as milestone grows — more evidence should help more).
  - hidden recall@10: of the truly-hidden distinct cards, what fraction of
    their probability mass falls in the model's top-10 predicted-count cards
    (excluding already-revealed cards).

Pass/fail bars (proposed, see PLAN — negotiable pending real numbers, the
STRUCTURAL requirement of "beats prior, improves with milestone" matters more
than hitting an exact number):
  - L1 ratio <= 0.85 at milestone=1, <= 0.55 at milestone>=8
  - recall@10 >= 0.40 at milestone>=8
  - L1 ratio must be non-increasing as milestone grows (model should exploit
    more evidence, not ignore it)

Usage:
  .venv/bin/python scripts/deckbelief_gate.py
  .venv/bin/python scripts/deckbelief_gate.py --model out/deckbelief/model.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import torch

from deckbelief_common import DeckBeliefNet, N_CARDS, build_card_caps_and_ace_mask

MILESTONES = (1, 2, 4, 8, 12, 20)

# Proposed starting bars (see module docstring) — not derived from any prior
# belief-net in this repo (there isn't one yet), flagged as negotiable.
L1_RATIO_BAR = {1: 0.85, 8: 0.55}
RECALL10_BAR = {8: 0.40}


def evaluate(model: DeckBeliefNet, prior_logits: torch.Tensor, bundle: dict) -> list[dict]:
    revealed_all = bundle["revealed"]
    true_all = bundle["true_deck"]
    milestone_all = bundle["milestone"]

    rows = []
    model.eval()
    with torch.no_grad():
        for m in MILESTONES:
            mask = milestone_all == m
            n = int(mask.sum())
            if n == 0:
                rows.append({"milestone": m, "n": 0})
                continue
            revealed = revealed_all[mask]
            true_deck = true_all[mask]

            pred = model(revealed)
            prior_pred = model.apply_constraints(prior_logits.expand(n, -1), revealed)

            hidden = (true_deck - revealed).clamp_min(0)
            hidden_total = hidden.sum(dim=-1, keepdim=True).clamp_min(1.0)
            model_hidden = (pred - revealed).clamp_min(0)
            prior_hidden = (prior_pred - revealed).clamp_min(0)

            model_l1 = ((model_hidden - hidden).abs().sum(dim=-1) / (2 * hidden_total.squeeze(-1))).mean().item()
            prior_l1 = ((prior_hidden - hidden).abs().sum(dim=-1) / (2 * hidden_total.squeeze(-1))).mean().item()

            top10 = torch.topk(model_hidden, k=min(10, N_CARDS), dim=-1).indices
            recall10 = (torch.gather(hidden, -1, top10).sum(dim=-1) / hidden_total.squeeze(-1)).mean().item()
            top10_prior = torch.topk(prior_hidden, k=min(10, N_CARDS), dim=-1).indices
            recall10_prior = (torch.gather(hidden, -1, top10_prior).sum(dim=-1) / hidden_total.squeeze(-1)).mean().item()

            rows.append({
                "milestone": m,
                "n": n,
                "model_l1": model_l1,
                "prior_l1": prior_l1,
                "l1_ratio": model_l1 / prior_l1 if prior_l1 > 1e-9 else float("nan"),
                "model_recall10": recall10,
                "prior_recall10": recall10_prior,
            })
    return rows


def check_bars(rows: list[dict]) -> tuple[bool, list[str]]:
    problems = []
    prev_ratio = None
    for row in rows:
        if row.get("n", 0) == 0:
            continue
        m = row["milestone"]
        ratio = row["l1_ratio"]
        if m in L1_RATIO_BAR and ratio > L1_RATIO_BAR[m]:
            problems.append(f"milestone={m}: L1 ratio {ratio:.3f} > bar {L1_RATIO_BAR[m]}")
        if m in RECALL10_BAR and row["model_recall10"] < RECALL10_BAR[m]:
            problems.append(f"milestone={m}: recall@10 {row['model_recall10']:.3f} < bar {RECALL10_BAR[m]}")
        if prev_ratio is not None and ratio > prev_ratio + 0.03:  # small slack for noise
            problems.append(f"milestone={m}: L1 ratio {ratio:.3f} increased vs previous milestone "
                             f"{prev_ratio:.3f} (evidence should help monotonically, not hurt)")
        prev_ratio = ratio
    return (len(problems) == 0), problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="out/deckbelief/model.pth")
    ap.add_argument("--data-dir", default="data/belief")
    args = ap.parse_args()

    data_dir = REPO / args.data_dir
    train_bundle = torch.load(data_dir / "train.pt", weights_only=True)
    val_bundle = torch.load(data_dir / "val.pt", weights_only=True)

    card_caps, ace_mask = build_card_caps_and_ace_mask(csv_path=str(REPO / "data" / "EN_Card_Data.csv"))
    model = DeckBeliefNet(ace_mask, card_caps)
    state = torch.load(REPO / args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Prior baseline: population-average TRAIN deck (log for use as logits —
    # bounded_fill's water_fill treats it via softmax, so any monotonic
    # transform of the mean works; log matches the source notebook's choice
    # and keeps zero-count cards from dominating via -inf).
    mean_train_deck = train_bundle["true_deck"].mean(dim=0)
    prior_logits = torch.log(mean_train_deck.clamp_min(1e-5)).unsqueeze(0)

    rows = evaluate(model, prior_logits, val_bundle)

    print(f"{'milestone':>9} {'n':>6} {'model_L1':>9} {'prior_L1':>9} {'ratio':>7} "
          f"{'recall@10':>10} {'prior_r@10':>11}")
    for row in rows:
        if row.get("n", 0) == 0:
            print(f"{row['milestone']:>9} {'0':>6}  (no val samples at this milestone)")
            continue
        print(f"{row['milestone']:>9} {row['n']:>6} {row['model_l1']:>9.4f} {row['prior_l1']:>9.4f} "
              f"{row['l1_ratio']:>7.3f} {row['model_recall10']:>10.3f} {row['prior_recall10']:>11.3f}")

    passed, problems = check_bars(rows)
    print()
    if passed:
        print("GATE: PASS")
    else:
        print("GATE: FAIL")
        for p in problems:
            print(f"  - {p}")

    report = {"rows": rows, "passed": passed, "problems": problems}
    out_path = REPO / "out" / "deckbelief" / "gate_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
