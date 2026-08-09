#!/usr/bin/env python3
"""
build_deckbelief_prototypes.py — build per-archetype prototype count-vectors
for the live detect_matchup() belief-similarity check (submissions/v047_deckbelief).

For each of the 8 EXISTING archetype line-sets already hardcoded in
_detect_matchup_board_only (CRUSTLE_LINE, HOP_LINE, STARMIE_LINE,
LUCARIO_LINE, ALAKAZAM_LINE, GRIMMSNARL_LINE, DIPPLIN_LINE, ABOMASNOW_LINE),
average the full 60-card count vectors of every real deck in our corpus
(data/belief/corpus_decks_{train,val}.pt) whose card set intersects that
line, giving a data-driven "typical decklist shape" for that archetype
(not just the 2-3 identifying Pokemon IDs, but everything that tends to run
alongside them — energy, trainers, etc.) that the belief-net's predicted
expected-count vector can be compared against via cosine similarity.

MVP SCOPE NOTE: this only builds prototypes for the 8 archetypes the agent's
existing numeric branches (opp_max_damage/Judge/Boss/...) already recognize
by name — it deliberately does NOT try to discover brand-new archetype
clusters from the corpus (that's the "generic-bucket" half of the fix,
explicitly deferred past this MVP, see PLAN §5.1). If an archetype has zero
matching decks in the local corpus, falls back to a synthetic vector (line
IDs at their per-card cap, everything else 0) — only the cosine DIRECTION
matters for similarity matching, so an approximate fallback is fine.

Output: submissions/v047_deckbelief/deckbelief_prototypes.json
  {"sim_threshold": 0.55, "ace_spec_ids": [...],
   "prototypes": {"crustle": {"344": 4.0, ...}, ...}}

Usage:
  .venv/bin/python scripts/build_deckbelief_prototypes.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import torch

from deckbelief_common import N_CARDS, build_card_caps_and_ace_mask

# Mirrors submissions/v044_archaludon_abomasnow/main.py's own LINE constants
# (kept as a literal copy here, not an import, so this offline build script
# has no dependency on the submission's main.py module).
ARCHETYPE_LINES = {
    "crustle": {344, 345, 532},
    "starmie": {1030, 1031},
    "lucario": {677, 678},
    "hop": {288, 289, 299, 304, 307, 308, 309, 310, 878, 879},
    "alakazam": {741, 742, 743},
    "grimmsnarl": {646, 647, 648},
    "dipplin": {89, 90, 92, 93, 247},
    "abomasnow": {722, 723},
    # Our OWN deck's line (Duraludon/Archaludon ex) -- added after a live-forensic
    # mirror-match diagnostic found the belief net, with no "self" prototype to
    # match against, confidently (mis)classified an Archaludon-mirror opponent as
    # "crustle" (generic trainer-card overlap outscored every other real
    # archetype) -- which then wrongly fired the crustle-only override block for
    # the rest of the game and lost 93% of mirror games in local gating. This
    # was exactly the "Archaludon-зеркало falls through to generic" gap flagged
    # in PLAN §5.1 as a known blind spot of the 8-archetype MVP scope; giving it
    # a real prototype closes the specific catastrophic failure mode without
    # requiring the full "detect any new archetype" generalization.
    "archaludon": {169, 190},
}

SIM_THRESHOLD = 0.55  # starting guess, see PLAN §5.1 — tune once live data exists
SIM_MARGIN = 0.08  # top-vs-second-best cosine sim margin required (rejects ambiguous reads)
MIN_EVIDENCE = 8  # minimum revealed opponent cards before trusting belief classification at all
# (see submissions/v047_deckbelief/main.py's _classify_belief docstring for why: the
# model has no trained behavior at 0 revealed cards and produces a confident-looking
# but purely-prior-driven guess there, diagnosed via a mirror-match live forensic.)


def deck_to_counts(deck: list[int]) -> torch.Tensor:
    v = torch.zeros(N_CARDS)
    for cid, cnt in Counter(deck).items():
        if 0 <= cid < N_CARDS:
            v[cid] = cnt
    return v


def main():
    data_dir = REPO / "data" / "belief"
    train_corpus = torch.load(data_dir / "corpus_decks_train.pt", weights_only=True)
    val_corpus = torch.load(data_dir / "corpus_decks_val.pt", weights_only=True)
    all_decks = torch.cat([train_corpus, val_corpus], dim=0).tolist()
    print(f"Corpus: {len(all_decks)} unique decks (train+val combined, prototypes are "
          f"fixed similarity anchors, not part of gradient training -- no leakage concern)")

    _, ace_mask = build_card_caps_and_ace_mask(csv_path=str(REPO / "data" / "EN_Card_Data.csv"))

    prototypes = {}
    for name, line_ids in ARCHETYPE_LINES.items():
        matching = [deck for deck in all_decks if line_ids & set(deck)]
        if matching:
            vecs = torch.stack([deck_to_counts(d) for d in matching])
            proto = vecs.mean(dim=0)
            source = f"{len(matching)} real decks"
        else:
            proto = torch.zeros(N_CARDS)
            for cid in line_ids:
                proto[cid] = 1.0 if (0 <= cid < N_CARDS and ace_mask[cid]) else 4.0
            source = "synthetic fallback (0 matching decks in corpus)"
        prototypes[name] = {str(i): float(v) for i, v in enumerate(proto.tolist()) if v > 1e-6}
        print(f"  {name:12s}: {source}, {len(prototypes[name])} nonzero cards in prototype")

    out = {
        "sim_threshold": SIM_THRESHOLD,
        "sim_margin": SIM_MARGIN,
        "min_evidence": MIN_EVIDENCE,
        "n_cards": N_CARDS,
        "prototypes": prototypes,
    }
    out_path = REPO / "submissions" / "v047_deckbelief" / "deckbelief_prototypes.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
