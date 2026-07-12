"""
cardemb_common.py — shared encoder/model code for the card-embedding IL pipeline.

This is (almost) a verbatim extraction of the encoder/model classes from the
organizer's official RL/MCTS sample (scratchpad/gold/kiyotah_cell1.py, kiyotah).
We reuse it AS-IS for two reasons:
  1. It already implements a per-card-id EmbeddingBag encoder/decoder + small
     Transformer (MyModel) that is exactly the "card embedding" architecture we
     want (vs. our LightGBM baseline that collapses card identity to a few
     hand-picked flags).
  2. Byte-for-byte reuse keeps the encoding logic identical to what a future
     MCTS/self-play pipeline (if we ever revisit it) would use, so a BC-trained
     MyModel checkpoint is a drop-in prior for that code path too.

What changed vs. kiyotah_cell1.py:
  - Removed the self-play/MCTS training loop, battle_start/battle_select calls,
    and the `/kaggle/input/**/cg-lib` sys.path hack — none of that is needed for
    supervised behavior cloning on precomputed replay data.
  - card_count / attack_count / decoder_size are HARDCODED (computed once,
    2026-07-12, from src/cg_native's all_card_data()/all_attack(), which reads
    the same competition card database as the Linux cg-lib). This lets this
    module be imported with ZERO game-engine dependency, on any platform
    (needed for the Kaggle GPU training kernel, which should not need to load
    libcg at all). See `recompute_constants()` below if the card DB ever grows
    (organizer's docs warn new cards/attacks/enum values may be appended).
  - SelectContext/OptionType/AreaType int values are inlined as plain ints
    (same enum, no import needed) — see the cg_native/cg `api.py` source for
    the authoritative definitions if these ever need to be re-derived.

Everything else (SparseVector, add_card/add_cards/add_pokemon/add_player,
get_encoder_input, get_card, decoder_main/decoder_card_id/decoder_card,
get_decoder_input, DecoderLayer, MyModel) is unchanged logic from kiyotah's code.
"""
from __future__ import annotations

import math

import torch
import torch.nn
import torch.nn.functional

# ---------------------------------------------------------------------------
# Constants (hardcoded — computed 2026-07-12 via src/cg_native, see docstring).
# ---------------------------------------------------------------------------
CARD_COUNT = 1268   # max(all_card_data().cardId) + 1
ATTACK_COUNT = 1557  # max(all_attack().attackId) + 1

# SelectContext.RECOVER_SPECIAL_CONDITION = 48 (last enum value used in decoder sizing)
_RECOVER_SPECIAL_CONDITION = 48

num_words_encoder = 24
encoder_size = 22000  # Encoder input size exceeding the vocabulary size (kiyotah's constant)

decoder_main_feature = 8  # Feature count of SelectContext.Main
decoder_attack_offset = 14  # First index of Attack feature
decoder_card_offset = decoder_attack_offset + ATTACK_COUNT  # First index of Card Feature
decoder_size = decoder_card_offset + (1 + decoder_main_feature + _RECOVER_SPECIAL_CONDITION) * CARD_COUNT

# Back-compat aliases matching kiyotah's original names (some code/comments use these).
card_count = CARD_COUNT
attack_count = ATTACK_COUNT


def recompute_constants():
    """
    Recompute CARD_COUNT/ATTACK_COUNT/decoder_size from a live cg/cg_native install.
    Only needed if the organizer appends new cards/attacks/enum values later in the
    competition and the hardcoded constants above go stale. Requires libcg to be
    importable (i.e. NOT usable standalone on the Kaggle GPU training kernel).
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from cg_native.api import all_attack, all_card_data  # noqa: E402

    cards = all_card_data()
    attacks = all_attack()
    cc = max(cards, key=lambda c: c.cardId).cardId + 1
    ac = max(attacks, key=lambda a: a.attackId).attackId + 1
    return cc, ac


# ---------------------------------------------------------------------------
# torch.nn.EmbeddingBag input (verbatim from kiyotah_cell1.py)
# ---------------------------------------------------------------------------
class SparseVector:
    index: list[int]
    value: list[float]
    offset: list[int]
    pos: int

    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []
        self.pos = 0

    def add(self, index: int, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_pos(self, pos: int):
        self.pos += pos

    def add_single(self, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos)
            self.value.append(value)
        self.pos += 1

    def word_start(self):
        self.offset.append(len(self.index))


# Add encoder card feature
def add_card(sv: SparseVector, card):
    if card is not None:
        sv.add(card.id, 1)
    sv.add_pos(CARD_COUNT)


# Add encoder cards feature
def add_cards(sv: SparseVector, cards, value: float):
    if cards is not None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(CARD_COUNT)


# Add encoder Pokemon feature
def add_pokemon(sv: SparseVector, poke):
    if poke is None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * CARD_COUNT)
    else:
        sv.add_single(0)
        sv.add_single(poke.hp / 400)
        add_card(sv, poke)
        add_cards(sv, poke.tools, 1.0)
        add_cards(sv, poke.energyCards, 0.5)


# Add encoder player feature
def add_player(sv: SparseVector, ps):
    sv.add_single(ps.deckCount / 60)
    sv.add_single(len(ps.discard) / 60)
    sv.add_single(ps.handCount / 8)
    sv.add_single(len(ps.bench) / 5)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)

    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)

    add_cards(sv, ps.discard, 0.25)


def get_encoder_input(obs, your_deck: list[int]) -> SparseVector:
    your_index = obs.current.yourIndex
    state = obs.current

    sv = SparseVector()
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8):  # For bench
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                add_pokemon(sv, ps.bench[j])
            else:
                add_pokemon(sv, None)
            if j != 7:  # Not last
                sv.pos = pos  # Return to the previous position

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        if 0 < len(ps.active):
            add_pokemon(sv, ps.active[0])
        else:
            add_pokemon(sv, None)

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        add_player(sv, ps)

    sv.word_start()
    add_cards(sv, state.players[your_index].hand, 0.25)

    sv.word_start()
    for id in your_deck:
        sv.add(id, 0.25)
    sv.add_pos(CARD_COUNT)

    sv.word_start()
    add_cards(sv, state.stadium, 1.0)

    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10)
    sv.add_single(state.firstPlayer == your_index)
    return sv


# AreaType int values (inlined; see cg_native/api.py AreaType for source of truth)
_AREA_DECK = 1
_AREA_HAND = 2
_AREA_DISCARD = 3
_AREA_ACTIVE = 4
_AREA_BENCH = 5
_AREA_PRIZE = 6
_AREA_STADIUM = 7
_AREA_LOOKING = 12


def get_card(obs, area: int, index: int, player_index: int):
    ps = obs.current.players[player_index]
    if area == _AREA_DECK:
        return obs.select.deck[index]
    elif area == _AREA_HAND:
        return ps.hand[index]
    elif area == _AREA_DISCARD:
        return ps.discard[index]
    elif area == _AREA_ACTIVE:
        return ps.active[index]
    elif area == _AREA_BENCH:
        return ps.bench[index]
    elif area == _AREA_PRIZE:
        return ps.prize[index]
    elif area == _AREA_STADIUM:
        return obs.current.stadium[index]
    elif area == _AREA_LOOKING:
        return obs.current.looking[index]
    else:
        return None


# Add decoder Main Select feature
def decoder_main(sv: SparseVector, feature_index: int, card):
    if card is not None:
        sv.add(decoder_card_offset + feature_index * CARD_COUNT + card.id, 1)


# Add decoder Card ID feature
def decoder_card_id(sv: SparseVector, context: int, card_id: int):
    sv.add(decoder_card_offset + (decoder_main_feature + context) * CARD_COUNT + card_id, 1)


# Add decoder Card feature
def decoder_card(sv: SparseVector, context: int, card):
    if card is not None:
        decoder_card_id(sv, context, card.id)


# OptionType int values (inlined; see cg_native/api.py OptionType for source of truth)
_OPT_NUMBER = 0
_OPT_YES = 1
_OPT_NO = 2
_OPT_CARD = 3
_OPT_TOOL_CARD = 4
_OPT_ENERGY_CARD = 5
_OPT_ENERGY = 6
_OPT_PLAY = 7
_OPT_ATTACH = 8
_OPT_EVOLVE = 9
_OPT_ABILITY = 10
_OPT_DISCARD = 11
_OPT_RETREAT = 12
_OPT_ATTACK = 13
_OPT_END = 14
_OPT_SKILL = 15
_OPT_SPECIAL_CONDITION = 16


def get_decoder_input(obs, actions: list[list[int]]) -> SparseVector:
    sv = SparseVector()
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    context = obs.select.context
    for action in actions:
        sv.word_start()

        if len(action) == 0:
            sv.add(0, 1)
            continue

        for i in action:
            o = obs.select.option[i]
            t = o.type
            if t == _OPT_END:
                sv.add(1, 1)
            elif t == _OPT_YES:
                sv.add(2, 1)
            elif t == _OPT_NO:
                sv.add(3, 1)
            elif t == _OPT_SPECIAL_CONDITION:
                sv.add(4 + o.specialConditionType, 1)
            elif t == _OPT_NUMBER:
                sv.add(9 + min(o.number, 4), 1)
            elif t == _OPT_ATTACK:
                sv.add(decoder_attack_offset + o.attackId, 1)
            elif t == _OPT_PLAY:
                decoder_main(sv, 0, ps.hand[o.index])
            elif t == _OPT_ATTACH:
                decoder_main(sv, 1, get_card(obs, o.area, o.index, your_index))
                decoder_main(sv, 2, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
            elif t == _OPT_EVOLVE:
                decoder_main(sv, 3, get_card(obs, o.area, o.index, your_index))
                decoder_main(sv, 4, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
            elif t == _OPT_ABILITY:
                decoder_main(sv, 5, get_card(obs, o.area, o.index, your_index))
            elif t == _OPT_DISCARD:
                decoder_main(sv, 6, get_card(obs, o.area, o.index, your_index))
            elif t == _OPT_RETREAT:
                decoder_main(sv, 7, ps.active[0])
            elif t == _OPT_CARD:
                decoder_card(sv, context, get_card(obs, o.area, o.index, o.playerIndex))
            elif t == _OPT_TOOL_CARD:
                card = get_card(obs, o.area, o.index, o.playerIndex)
                decoder_card(sv, context, card.tools[o.toolIndex])
            elif t == _OPT_ENERGY_CARD or t == _OPT_ENERGY:
                card = get_card(obs, o.area, o.index, o.playerIndex)
                decoder_card(sv, context, card.energyCards[o.energyIndex])
            elif t == _OPT_SKILL:
                decoder_card_id(sv, context, o.cardId)

    return sv


# ---------------------------------------------------------------------------
# Model (verbatim from kiyotah_cell1.py)
# ---------------------------------------------------------------------------
class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int, dropout: float = 0.0):
        super(DecoderLayer, self).__init__()

        self.attention = torch.nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = torch.nn.Linear(d_model, d_feedforward)
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        # Regularization (added 2026-07-12 for the ×5-data + regularization pass —
        # default 0.0 keeps this byte-for-byte equivalent to kiyotah's original
        # DecoderLayer unless a caller explicitly opts in via MyModel(dropout=...)).
        self.dropout_attn = torch.nn.Dropout(dropout)
        self.dropout_ff = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        y = self.dropout_attn(y)
        res = self.norm1(x + y)
        y = self.fc1(res)
        y = torch.nn.functional.relu(y)
        y = self.dropout_ff(y)
        y = self.fc2(y)
        return self.norm2(res + y)


class MyModel(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        num_layers_encoder: int,
        num_layers_decoder: int,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        """
        dropout: TransformerEncoderLayer internal dropout (attn+ff, kiyotah's original
            hardcoded this to 0) AND the DecoderLayer attn/ff dropout added above.
        emb_dropout: extra nn.Dropout applied to the raw encoder_bag/decoder_bag
            EmbeddingBag outputs, before the Transformer stack — regularizes the
            12M-param per-card embedding table directly (added 2026-07-12; not in
            kiyotah's original). Default 0.0 for both -> byte-for-byte original model.
        """
        super(MyModel, self).__init__()

        self.d_model = d_model

        self.encoder_bag = torch.nn.EmbeddingBag(encoder_size, d_model, mode="sum")
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, dropout)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers_encoder, enable_nested_tensor=False)
        self.encoder_fc = torch.nn.Linear(d_model, 1)
        self.decoder_bag = torch.nn.EmbeddingBag(decoder_size, d_model, mode="sum")
        self.decoder = torch.nn.ModuleList()
        for _ in range(num_layers_decoder):
            self.decoder.append(DecoderLayer(d_model, num_heads, d_feedforward, dropout))
        self.decoder_fc = torch.nn.Linear(d_model, 1)
        self.emb_dropout_enc = torch.nn.Dropout(emb_dropout)
        self.emb_dropout_dec = torch.nn.Dropout(emb_dropout)

    def forward(
        self,
        index_encoder: torch.Tensor,
        value_encoder: torch.Tensor,
        offset_encoder: torch.Tensor,
        index_decoder: torch.Tensor,
        value_decoder: torch.Tensor,
        offset_decoder: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        v = self.emb_dropout_enc(v)
        v = v.reshape(-1, num_words_encoder, self.d_model).transpose(0, 1)
        batch_size = v.size(1)
        encoder_out = self.encoder(v)
        v = self.encoder_fc(encoder_out)
        v = torch.tanh(v.mean(0))

        p = self.decoder_bag(index_decoder, offset_decoder, value_decoder)
        p = self.emb_dropout_dec(p)
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, encoder_out)
        p = self.decoder_fc(p)
        p = p.transpose(0, 1).view(batch_size, -1)
        p = torch.tanh(p)
        return (v, p)


# ---------------------------------------------------------------------------
# Batch helper (same idea as kiyotah's LearnInput, generalized to variable
# per-sample word counts padded to a fixed MAX_OPTIONS with a mask).
# ---------------------------------------------------------------------------
class LearnInput:
    """Concatenates several SparseVector-like (index, value, offset) triples into
    one EmbeddingBag-ready batch, rebasing offsets by the running index count."""

    def __init__(self):
        self.index: list[int] = []
        self.value: list[float] = []
        self.offset: list[int] = []

    def add(self, index: list[int], value: list[float], offset: list[int]):
        count = len(self.index)
        self.index.extend(index)
        self.value.extend(value)
        for o in offset:
            self.offset.append(o + count)
