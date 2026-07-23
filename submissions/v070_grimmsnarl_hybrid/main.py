"""v070_grimmsnarl_hybrid — external prvsiyan/ptcg-rmy-grimmsnarl-replay-hybrid-v1
attention-BC pilot (bc_agent.py, unmodified) re-decked onto OUR Grimmsnarl decklist
(submissions/v064_starmie_scaling/deck.csv — swaps external card 1137 Tool Scrapper
for our 1197 Xerosic's Machinations; all other 59 cards identical).

bc_agent.py / model.pt are byte-identical copies of
submissions/external_prvsiyan_grimmsnarl_hybrid/{bc_agent.py,model.pt} — do NOT edit
bc_agent.py, load_policy() verifies a static feature_hash against card_feat_table and
will refuse to load the checkpoint if the encoding changes. Deck-specific features are
computed at runtime from deck.csv via _DECK_COUNTER inside bc_agent.predict_action, so
the shared checkpoint adapts to our decklist without retraining.

Only change vs. the external main.py: this wrapper adds per-decision stderr logging
(DECISION_LOG, matching the format used across submissions/v0NN_*/main.py) so live
Kaggle episode logs can be replayed move-by-move. There is no human-readable "reason"
string here (no heuristic scorer) — the BC model's own per-option logits
(``score_values`` from bc_agent.predict_action's ``scores`` tensor) and the
``no_action_value`` threshold serve as the score/margin analogue, plus whether the
move came from one of the two deterministic replay-validated overrides
(_punk_up_override / _lowest_hp_override) or the base neural policy.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path
from typing import Any

import torch

from bc_agent import load_policy, predict_action


torch.set_num_threads(1)

# Released SDK SelectContext / OptionType constants (identical to external main.py).
_DAMAGE_COUNTER = 13
_DAMAGE = 15
_REMOVE_DAMAGE_COUNTER = 16
_ATTACH_FROM = 21
_TARGET_CONTEXTS = {_DAMAGE_COUNTER, _DAMAGE, _REMOVE_DAMAGE_COUNTER}
_CARD_OPTION = 3

# Exactly the observed ``Punk Up`` selection: a Darkness Energy card selected
# by the Grimmsnarl effect.  The rule is never used for any other attachment.
_PUNK_UP_EFFECT = 648
_DARKNESS_ENERGY = 7
_MARNIE_TARGET_IDS = {646, 647, 648}


def _checkpoint_path() -> Path:
    raw_file = globals().get("__file__")
    candidates: list[Path] = []
    if raw_file:
        candidates.append(Path(str(raw_file)).resolve().with_name("model.pt"))
    candidates.extend((Path.cwd() / "model.pt", Path("/kaggle_simulations/agent/model.pt")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"model.pt was not found in: {candidates}")


_MODEL = load_policy(_checkpoint_path(), "cpu")


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _target_card(obs: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve an in-play CARD option without relying on hidden card data."""
    current = obs.get("current")
    if not isinstance(current, dict):
        return None
    players = current.get("players")
    player_index = _integer(option.get("playerIndex"), -1)
    if not isinstance(players, list) or not 0 <= player_index < len(players):
        return None
    player = players[player_index]
    if not isinstance(player, dict):
        return None
    area = _integer(option.get("area"), -1)
    index = _integer(option.get("index"), -1)
    if area == 4:  # ACTIVE
        active = player.get("active")
        if isinstance(active, list) and 0 <= index < len(active):
            card = active[index]
            return card if isinstance(card, dict) else None
        if isinstance(active, dict) and index == 0:
            return active
    if area == 5:  # BENCH
        bench = player.get("bench")
        if isinstance(bench, list) and 0 <= index < len(bench):
            card = bench[index]
            return card if isinstance(card, dict) else None
    return None


def _lowest_hp_override(obs: dict[str, Any], base_action: list[int]) -> list[int] | None:
    """Use the demonstrated lowest-remaining-HP target in three forced contexts."""
    select = obs.get("select")
    if not isinstance(select, dict) or _integer(select.get("context")) not in _TARGET_CONTEXTS:
        return None
    options = select.get("option")
    if not isinstance(options, list) or not base_action:
        return None
    resolved: list[tuple[int, int]] = []
    for option_index, option in enumerate(options):
        if not isinstance(option, dict) or _integer(option.get("type")) != _CARD_OPTION:
            continue
        card = _target_card(obs, option)
        hp = card.get("hp") if isinstance(card, dict) else None
        if isinstance(hp, bool):
            continue
        try:
            resolved.append((int(hp), option_index))
        except (TypeError, ValueError):
            continue
    count = min(len(base_action), len(resolved))
    if count <= 0:
        return None
    resolved.sort()
    return [option_index for _, option_index in resolved[:count]]


# This compact tree is trained solely on replay-disjoint public decisions for
# the above Punk Up effect.  It scores each legal target independently; its
# only inputs are the visible board, legal options, and select metadata.
_TREE_LEFT = (1, -1, 3, 4, 5, 6, -1, -1, 9, -1, -1, 12, 13, -1, -1, 16, -1, -1, 19, 20, 21, -1, -1, 24, -1, -1, 27, 28, -1, -1, 31, -1, -1)
_TREE_RIGHT = (2, -1, 18, 11, 8, 7, -1, -1, 10, -1, -1, 15, 14, -1, -1, 17, -1, -1, 26, 23, 22, -1, -1, 25, -1, -1, 30, 29, -1, -1, 32, -1, -1)
_TREE_FEATURE = (13, -2, 1, 2, 27, 9, -2, -2, 0, -2, -2, 4, 14, -2, -2, 9, -2, -2, 2, 11, 2, -2, -2, 13, -2, -2, 26, 10, -2, -2, 15, -2, -2)
_TREE_THRESHOLD = (1.5, -2.0, 1.5, 210.0, 0.5, 0.5, -2.0, -2.0, 647.5, -2.0, -2.0, 4.5, 0.5, -2.0, -2.0, 0.5, -2.0, -2.0, 190.0, 1.5, 135.0, -2.0, -2.0, 2.5, -2.0, -2.0, 2.5, 2.5, -2.0, -2.0, 6.0, -2.0, -2.0)
_TREE_POSITIVE = (0.5, 1.0, 0.46118, 0.56167, 0.43398, 0.66016, 0.74724, 0.02478, 0.07571, 0.06305, 0.58291, 0.94841, 1.0, 1.0, 1.0, 0.87481, 0.96918, 0.7176, 0.24811, 0.72379, 0.24967, 0.13441, 0.58291, 0.87481, 0.985, 0.52787, 0.09627, 0.0333, 0.02487, 0.21845, 0.36376, 0.0624, 0.50811)


def _punk_up_features(options: list[dict[str, Any]], option_index: int, cards: list[dict[str, Any]], select: dict[str, Any]) -> list[int]:
    card = cards[option_index]
    energies = [len(item.get("energies") or []) for item in cards]
    ids = [_integer(item.get("id"), -1) for item in cards]
    card_id = ids[option_index]
    energy_count = energies[option_index]
    values = [
        card_id, energy_count, _integer(card.get("hp")), _integer(card.get("maxHp")),
        _integer(options[option_index].get("area")), _integer(options[option_index].get("index")),
        int(card_id == 646), int(card_id == 647), int(card_id == 648),
        sum(energy < energy_count for energy in energies),
        sum(energy == energy_count for energy in energies), min(energies), max(energies), len(options),
    ]
    for target_id in (646, 647, 648):
        target_energies = [energy for energy, observed_id in zip(energies, ids) if observed_id == target_id]
        values.extend((
            len(target_energies),
            min(target_energies) if target_energies else 9,
            max(target_energies) if target_energies else 9,
            sum(energy < 2 for energy in target_energies),
            sum(energy < 3 for energy in target_energies),
        ))
    values.extend((
        _integer(select.get("remainEnergyCost")), _integer(select.get("minCount")), _integer(select.get("maxCount")),
    ))
    return values


def _tree_score(features: list[int]) -> float:
    node = 0
    while _TREE_FEATURE[node] >= 0:
        node = _TREE_LEFT[node] if features[_TREE_FEATURE[node]] <= _TREE_THRESHOLD[node] else _TREE_RIGHT[node]
    return _TREE_POSITIVE[node]


def _punk_up_override(obs: dict[str, Any]) -> list[int] | None:
    """Allocate one Punk Up energy with the replay-trained visible-board tree."""
    select = obs.get("select")
    if not isinstance(select, dict) or _integer(select.get("context")) != _ATTACH_FROM:
        return None
    if _integer(select.get("minCount")) != 1 or _integer(select.get("maxCount")) != 1:
        return None
    effect = select.get("effect")
    context_card = select.get("contextCard")
    if not (isinstance(effect, dict) and isinstance(context_card, dict)):
        return None
    if _integer(effect.get("id")) != _PUNK_UP_EFFECT or _integer(context_card.get("id")) != _DARKNESS_ENERGY:
        return None
    options = select.get("option")
    if not isinstance(options, list) or not options:
        return None
    cards: list[dict[str, Any]] = []
    for option in options:
        if not isinstance(option, dict) or _integer(option.get("type")) != _CARD_OPTION:
            return None
        card = _target_card(obs, option)
        if not isinstance(card, dict) or _integer(card.get("id"), -1) not in _MARNIE_TARGET_IDS:
            return None
        cards.append(card)
    ranked = sorted(
        ((_tree_score(_punk_up_features(options, index, cards, select)), index) for index in range(len(options))),
        key=lambda pair: (-pair[0], pair[1]),
    )
    return [ranked[0][1]]


# ── Per-decision trace logging (v070; format mirrors submissions/v0NN_*/main.py) ──
# Kaggle captures per-move stdout/stderr in the downloadable agent logs
# (episode-{id}-agent-{n}.json). There is no heuristic "reason" text for this
# neural pilot, so we log the model's own per-option logits (score_values) and
# the no_action_value threshold in place of score/margin, plus which decision
# path was taken (base bc_model vs. one of the two deterministic overrides).
# Never throws; toggle DECISION_LOG=False to silence.
DECISION_LOG = True


def _dlog(msg: str) -> None:
    if DECISION_LOG:
        try:
            print(f"[dec] {msg}", file=_sys.stderr, flush=True)
        except Exception:
            pass


def _score_values_for_log(obs_dict: dict[str, Any]):
    """Best-effort re-derivation of the model's per-option logits for logging
    only. Mirrors bc_agent.predict_action's scoring call; any failure here
    (e.g. before _DECK_COUNTER is primed, or a non-model fallback path) is
    swallowed and simply omitted from the log line — never raised to the
    caller, and never affects the actual chosen action."""
    from bc_agent import _cards, _int, encode_decision, _DECK_COUNTER as _dc  # type: ignore

    select = obs_dict.get("select")
    if not isinstance(select, dict):
        return None, None
    options = _cards(select.get("option"))
    if not options or _dc is None:
        return None, None
    state_numeric, state_cards, option_cats, option_numeric = encode_decision(obs_dict, _dc)
    device = next(_MODEL.parameters()).device
    with torch.inference_mode():
        scores, no_action = _MODEL(
            torch.from_numpy(state_numeric).unsqueeze(0).to(device),
            torch.from_numpy(state_cards).unsqueeze(0).to(device),
            torch.from_numpy(option_cats).unsqueeze(0).to(device),
            torch.from_numpy(option_numeric).unsqueeze(0).to(device),
        )
    return scores[0].detach().cpu().tolist(), float(no_action[0].detach().cpu())


def agent(obs_dict: dict) -> list[int]:
    """Choose legal actions, replacing only two replay-validated subdecisions."""
    select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if not isinstance(select, dict):
        # Deck-select call (obs.select is None) — just the 60-card decklist, no log.
        return predict_action(obs_dict, _MODEL)

    base_action = predict_action(obs_dict, _MODEL)
    punk_up = _punk_up_override(obs_dict)
    lowest_hp = None if punk_up is not None else _lowest_hp_override(obs_dict, base_action)
    final_action = punk_up if punk_up is not None else (lowest_hp if lowest_hp is not None else base_action)

    if DECISION_LOG:
        try:
            turn = None
            current = obs_dict.get("current")
            if isinstance(current, dict):
                turn = current.get("turn")
            ctx = _integer(select.get("context"), -1)
            n_opts = len(select.get("option") or [])

            if punk_up is not None:
                path = "override:punk_up"
            elif lowest_hp is not None:
                path = "override:lowest_hp"
            else:
                path = "bc_model"

            score_values, no_action_value = _score_values_for_log(obs_dict)
            if score_values:
                order = sorted(range(len(score_values)), key=lambda i: (-score_values[i], i))
                top_i = order[0]
                top_s = score_values[top_i]
                if len(order) > 1:
                    second_i = order[1]
                    second_s = score_values[second_i]
                    sec = f" | 2nd #{second_i} s={second_s:.3f}"
                else:
                    sec = ""
                na = f" no_action={no_action_value:.3f}" if no_action_value is not None else ""
                _dlog(f"t{turn} ctx{ctx} n{n_opts} -> #{top_i} '{path}' s={top_s:.3f}{sec}{na} "
                      f"pick={final_action}")
            else:
                _dlog(f"t{turn} ctx{ctx} n{n_opts} -> '{path}' pick={final_action}")
        except Exception:
            pass

    return final_action
