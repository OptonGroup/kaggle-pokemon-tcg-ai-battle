"""
deckbelief_common.py — DeckBeliefNet: predicts an opponent's full 60-card deck's
expected per-card count vector from currently-revealed cards.

Adapted from the Kaggle notebook
hitoshimochizuki/live-opponent-deck-prediction-from-revealed-cards (2026-07-19):
same MLP + "bounded fill" constrained-output idea (revealed counts are hard
lower bounds, per-card cap, ACE SPEC cards share one collective slot, output
sums to exactly 60), but:
  - ace_spec_mask / card_caps are supplied by the caller, derived from
    CARD_DB[cid].aceSpec (submissions' own `all_card_data()`-based dict) or
    from data/EN_Card_Data.csv's "Rule" column directly (see
    build_card_caps_and_ace_mask below) — never hardcoded in this module.
  - N_CARDS mirrors scripts/cardemb_common.py's existing CARD_COUNT convention
    (kept as its own literal — this module has zero import dependency on
    cardemb_common, a different pipeline).

This module has no game-engine dependency (no cg/cg_native import) so it can
be trained locally on any platform and bundled standalone inside a submission
(same convention as submissions/v024_cardemb_rl/cardemb_model.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_CARDS = 1268  # == cardemb_common.CARD_COUNT (max(all_card_data().cardId) + 1)
HIDDEN = 256
DECK_SIZE = 60.0
DEFAULT_CARD_CAP = 4
BASIC_ENERGY_CAP = 60  # decks can run e.g. 11+ copies of a single basic energy type
BASIC_ENERGY_IDS = tuple(range(1, 9))  # verified: card ids 1-8 == the 8 basic energy types


def water_fill(logits: torch.Tensor, floor: torch.Tensor, cap: torch.Tensor,
                total: torch.Tensor) -> torch.Tensor:
    """
    Distribute `total` (..., 1) units across the last dim of `logits`, respecting
    per-item floor <= result <= cap, using `logits` as relative (softmax) weights
    for the flexible remainder (total - floor.sum(-1)).

    Standard water-filling: each round, split the remaining budget proportionally
    (softmax) among still-active items; any item whose share would exceed its
    remaining headroom (cap - already allocated) is clamped to that headroom and
    removed from the active set; repeat with the shrunk budget until nothing new
    saturates. Returns floor + allocation (shape matches `logits`).
    """
    headroom = (cap - floor).clamp_min(0.0)
    remaining = (total - floor.sum(-1, keepdim=True)).clamp_min(0.0)
    allocation = torch.zeros_like(logits)
    active = headroom > 1e-9
    for _ in range(logits.shape[-1]):
        if not bool(active.any()):
            break
        masked = logits.masked_fill(~active, float("-inf"))
        weights = torch.softmax(masked, dim=-1)
        weights = torch.where(active, weights, torch.zeros_like(weights))
        proposal = weights * remaining
        room = headroom - allocation
        saturated = active & (proposal >= room - 1e-6)
        if not bool(saturated.any()):
            allocation = allocation + proposal
            break
        allocation = torch.where(saturated, room, allocation)
        remaining = remaining - torch.where(saturated, room, torch.zeros_like(room)).sum(-1, keepdim=True)
        active = active & ~saturated
    return floor + allocation


class DeckBeliefNet(nn.Module):
    """
    Predicts the opponent's full-deck expected card-count vector from a vector
    of currently-revealed card counts.

    ace_spec_mask: bool tensor (n_cards,) — True for cards limited to 1 TOTAL
        copy across the whole ACE SPEC category (the game rule is "at most 1
        ACE SPEC card in your deck, period" — not 1 copy each).
    card_caps: float tensor (n_cards,) — per-card individual cap; ignored for
        ACE SPEC cards, which go through the shared category cap instead.
    """

    def __init__(self, ace_spec_mask: torch.Tensor, card_caps: torch.Tensor,
                 n_cards: int = N_CARDS, hidden: int = HIDDEN):
        super().__init__()
        self.n_cards = n_cards
        self.net = nn.Sequential(
            nn.Linear(n_cards, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_cards),
        )
        ace_spec_mask = ace_spec_mask.bool()
        self.register_buffer("ace_spec_mask", ace_spec_mask)
        self.register_buffer("card_caps", card_caps.float())
        self.register_buffer("non_ace_ids", torch.nonzero(~ace_spec_mask, as_tuple=True)[0])
        self.register_buffer("ace_ids", torch.nonzero(ace_spec_mask, as_tuple=True)[0])

    def forward(self, revealed: torch.Tensor) -> torch.Tensor:
        """
        revealed: (..., n_cards) float, raw counts of cards seen so far (both
        ordinary and ACE SPEC reveals already merged in).
        Returns the expected full-deck count vector, shape (..., n_cards),
        satisfying: result >= revealed elementwise, sum(result) == DECK_SIZE,
        per-card cap respected, and at most 1 total across all ACE SPEC cards.
        """
        logits = self.net(revealed)
        return self.apply_constraints(logits, revealed)

    def apply_constraints(self, logits: torch.Tensor, revealed: torch.Tensor) -> torch.Tensor:
        """
        Project raw per-card `logits` (any source — the learned net, or a
        constant/prior vector for baseline comparisons, see
        scripts/deckbelief_gate.py) into a valid expected-count vector, given
        `revealed` as the evidence floor. Factored out of forward() so a
        baseline "prior" (population-average deck, ignoring this sample's own
        evidence beyond the floor) can be scored through the exact same
        constraint machinery as the learned model, for a fair comparison.
        """
        non_logits = logits.index_select(-1, self.non_ace_ids)
        ace_logits = logits.index_select(-1, self.ace_ids)
        # Collapse the whole ACE SPEC category into one "virtual card" whose
        # logit is the log-sum-exp of its members, so a single softmax over
        # [non-ace cards..., ace-category] is well-defined.
        ace_group_logit = torch.logsumexp(ace_logits, -1, keepdim=True)

        non_floor = revealed.index_select(-1, self.non_ace_ids)
        ace_floor = revealed.index_select(-1, self.ace_ids)
        ace_group_floor = ace_floor.sum(-1, keepdim=True)

        non_caps = self.card_caps.expand_as(revealed).index_select(-1, self.non_ace_ids)
        ace_group_cap = torch.ones_like(ace_group_floor)  # at most 1 ACE SPEC card, total

        category_logits = torch.cat([non_logits, ace_group_logit], dim=-1)
        category_floor = torch.cat([non_floor, ace_group_floor], dim=-1)
        category_cap = torch.cat([non_caps, ace_group_cap], dim=-1)
        total = torch.full_like(ace_group_floor, DECK_SIZE)

        filled = water_fill(category_logits, category_floor, category_cap, total)
        non_result = filled[..., :-1]
        ace_group_result = filled[..., -1:]

        # Expand the ACE SPEC category's allocated total back to individual
        # cards, proportional to their own logits within the group.
        extra_ace = (ace_group_result - ace_group_floor).clamp_min(0.0)
        ace_result = ace_floor + torch.softmax(ace_logits, -1) * extra_ace

        out = torch.zeros_like(revealed)
        out.index_copy_(-1, self.non_ace_ids, non_result)
        out.index_copy_(-1, self.ace_ids, ace_result)
        return out


def build_card_caps_and_ace_mask(
    card_db: dict | None = None,
    csv_path: str = "data/EN_Card_Data.csv",
    n_cards: int = N_CARDS,
    basic_energy_ids: tuple[int, ...] = BASIC_ENERGY_IDS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (card_caps, ace_spec_mask) tensors of shape (n_cards,).

    Prefers `card_db` (dict[cardId -> CardData] with a `.aceSpec` bool field,
    matching submissions' own `CARD_DB = {c.cardId: c for c in all_card_data()}`
    convention) when given; otherwise reads the "Rule" column of
    data/EN_Card_Data.csv directly (== "ACE SPEC") — zero game-engine
    dependency, works on any platform without cg/cg_native.
    """
    ace_spec_mask = torch.zeros(n_cards, dtype=torch.bool)
    if card_db is not None:
        for cid, card in card_db.items():
            if 0 <= cid < n_cards and getattr(card, "aceSpec", False):
                ace_spec_mask[cid] = True
    else:
        import csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    cid = int(row["Card ID"])
                except (KeyError, ValueError):
                    continue
                if 0 <= cid < n_cards and row.get("Rule", "").strip() == "ACE SPEC":
                    ace_spec_mask[cid] = True

    card_caps = torch.full((n_cards,), float(DEFAULT_CARD_CAP))
    for eid in basic_energy_ids:
        if 0 <= eid < n_cards:
            card_caps[eid] = float(BASIC_ENERGY_CAP)
    return card_caps, ace_spec_mask
