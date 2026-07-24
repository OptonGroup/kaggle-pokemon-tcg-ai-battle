#!/usr/bin/env python3
"""Render a Kaggle cabt replay JSON into a human-readable game digest.

Usage:
    python3 scripts/replay_digest.py <replay.json> <our_index> [out.txt]
    python3 scripts/replay_digest.py --batch <download_dir>   # uses index.csv

The digest contains:
  - header: episode, our index, result, decks -> archetypes
  - event timeline from OUR player's observations (card names resolved)
  - board snapshot at every turn start (prizes/active/bench/hand/deck)
  - final RESULT reason

Digest is designed to be read by an analyst (human or LLM) next to the
agent decision-trace ([dec]/[alt]/[galt] stderr lines).
"""
import csv
import json
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AREA = {1: "DECK", 2: "HAND", 3: "DISCARD", 4: "ACTIVE", 5: "BENCH",
        6: "PRIZE", 7: "STADIUM", 8: "ENERGY", 9: "TOOL", 10: "PREEVO",
        11: "PLAYER", 12: "LOOKING"}

RESULT_REASON = {1: "opponent took all prizes / 0 prize cards left",
                 2: "deck out (0 deck cards at turn start)",
                 3: "no Pokemon in Active Spot (bench depleted)",
                 4: "card effect"}

sys.path.insert(0, os.path.join(REPO, "scripts"))
from classify_opponent_archetype import classify           # noqa: E402
from cluster_archetypes import cluster_and_name, load_card_meta   # noqa: E402

# Archetype naming lives in classify_opponent_archetype.py — the same table the
# WR report uses — so a matchup can't be called "Lucario" in a digest and
# "lucario" in the summary, and a newly-added archetype lands in both at once.
_CARD_META = None


def load_cards():
    names, moves = {}, defaultdict(list)
    path = os.path.join(REPO, "data/EN_Card_Data.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): (v or "").strip() for k, v in row.items()}
            try:
                cid = int(row["Card ID"])
            except (ValueError, KeyError):
                continue
            names[cid] = row.get("Card Name", f"#{cid}")
            mv = row.get("Move Name", "")
            if mv and mv.lower() != "n/a":
                moves[cid].append((mv, row.get("Damage", "")))
    return names, dict(moves)


def deck_archetype(deck_ids, names):
    """Shared label first; anything the table doesn't know gets named from the
    card data by clustering rather than falling into an opaque "unknown" — a
    deck nobody wrote a rule for is exactly the one worth seeing in forensics.

    Clustering a single deck degenerates to "the line it runs the most copies
    of, walked up its evolution chain", which is the right answer here.
    """
    global _CARD_META
    label = classify(set(deck_ids), [names.get(c, str(c)) for c in deck_ids])
    if label:
        return label
    try:
        if _CARD_META is None:
            _CARD_META = load_card_meta()
        return "auto:" + cluster_and_name([list(deck_ids)], _CARD_META)[0]
    except Exception:
        return "unknown"


def render(replay_path, our_index, names, moves, header_extra=""):
    d = json.load(open(replay_path))
    steps = d["steps"]
    rewards = d.get("rewards", [None, None])
    our_reward = rewards[our_index] if rewards else None
    opp_index = 1 - our_index

    def nm(cid):
        return names.get(cid, f"#{cid}")

    def side(pi):
        return "US " if pi == our_index else "OPP"

    out = []
    # decks from step0 visualize
    decks = None
    try:
        vis = steps[0][0].get("visualize") or steps[0][1].get("visualize")
        if vis and isinstance(vis, list) and vis and "action" in vis[0]:
            decks = vis[0]["action"]
    except Exception:
        pass
    our_arch = opp_arch = "unknown"
    if decks and len(decks) == 2:
        our_arch = deck_archetype(decks[our_index], names)
        opp_arch = deck_archetype(decks[opp_index], names)
        opp_counts = defaultdict(int)
        for c in decks[opp_index]:
            opp_counts[nm(c)] += 1
        opp_list = ", ".join(f"{v}x {k}" for k, v in
                             sorted(opp_counts.items(), key=lambda x: -x[1]))
    else:
        opp_list = "n/a"

    res = "WIN" if our_reward == 1 else ("LOSS" if our_reward == -1 else f"reward={our_reward}")
    out.append(f"=== EPISODE {os.path.basename(replay_path).split('.')[0]} — {res} for us ===")
    if header_extra:
        out.append(header_extra)
    out.append(f"our_index={our_index}  our deck archetype: {our_arch}  |  OPP archetype: {opp_arch}")
    out.append(f"OPP decklist: {opp_list}")
    out.append("")

    last_turn_printed = -1

    def emit_snapshot(i, cur):
        nonlocal last_turn_printed
        if not (cur and cur.get("players")):
            return
        turn = cur.get("turn", -1)
        if turn is None or turn <= last_turn_printed:
            return
        last_turn_printed = turn
        lines = []
        for tag, pi in (("US ", our_index), ("OPP", opp_index)):
            p = cur["players"][pi]
            act = p.get("active") or []
            if act and act[0]:
                a = act[0]
                acts = f"{nm(a['id'])} {a['hp']}/{a['maxHp']}hp E{len(a.get('energies', []))}"
            else:
                acts = "none"
            bench = ", ".join(f"{nm(b['id'])} {b['hp']}hp" for b in p.get("bench", []))
            prz = len([x for x in (p.get("prize") or [])])
            lines.append(f"    {tag}: prz_left={prz} active=[{acts}] bench=[{bench}] "
                         f"hand={p.get('handCount')} deck={p.get('deckCount')}")
        out.append(f"[s{i:03d}] == board @turn {turn} ==")
        out.extend(lines)

    def emit_logs(i, logs, skip_keys=None):
        # obs.logs is refreshed only when the engine prompts the agent; INACTIVE
        # steps re-serve the same stale window, so callers pass fresh windows only
        for lg in logs:
            if not isinstance(lg, dict):
                continue
            if skip_keys is not None and lg.get("type") not in (2, 3, 22, 23) \
                    and json.dumps(lg, sort_keys=True) in skip_keys:
                continue
            t = lg.get("type")
            pi = lg.get("playerIndex", -1)
            if t == 2:
                out.append(f"[s{i:03d}] ---- TURN START {side(pi)} ----")
            elif t == 3:
                pass  # turn end: noise
            elif t == 4:
                pass  # our draws: noise (deck counts in snapshots)
            elif t == 6:
                fr, to = AREA.get(lg.get("fromArea"), "?"), AREA.get(lg.get("toArea"), "?")
                cid = lg.get("cardId")
                sr = lg.get("serial")
                if to == "DISCARD" and fr in ("ACTIVE", "BENCH"):
                    out.append(f"[s{i:03d}] {side(pi)} KO/discard from {fr}: {nm(cid)} (serial {sr})")
                elif fr == "PRIZE":
                    out.append(f"[s{i:03d}] {side(pi)} took a PRIZE card: {nm(cid)}")
                elif to == "ACTIVE" and fr in ("HAND", "BENCH"):
                    out.append(f"[s{i:03d}] {side(pi)} promoted/placed ACTIVE: {nm(cid)}")
                elif to == "BENCH" and fr == "HAND":
                    out.append(f"[s{i:03d}] {side(pi)} benched: {nm(cid)}")
                elif to == "DISCARD" and fr == "HAND":
                    out.append(f"[s{i:03d}] {side(pi)} discarded from hand: {nm(cid)}")
            elif t == 7:
                fr, to = AREA.get(lg.get("fromArea"), "?"), AREA.get(lg.get("toArea"), "?")
                if fr == "PRIZE":
                    out.append(f"[s{i:03d}] {side(pi)} took a PRIZE card (facedown)")
            elif t == 8:
                out.append(f"[s{i:03d}] {side(pi)} SWITCH: {nm(lg.get('cardIdBench'))} -> active, "
                           f"{nm(lg.get('cardIdActive'))} -> bench (retreat/switch)")
            elif t == 9:
                out.append(f"[s{i:03d}] {side(pi)} CHANGE active: {nm(lg.get('cardIdBefore'))} -> "
                           f"{nm(lg.get('cardIdAfter'))} (gust/forced)")
            elif t == 10:
                out.append(f"[s{i:03d}] {side(pi)} played: {nm(lg.get('cardId'))}")
            elif t == 11:
                out.append(f"[s{i:03d}] {side(pi)} attached {nm(lg.get('cardId'))} -> {nm(lg.get('cardIdTarget'))}")
            elif t == 12:
                out.append(f"[s{i:03d}] {side(pi)} EVOLVED {nm(lg.get('cardIdTarget'))} -> {nm(lg.get('cardId'))}")
            elif t == 13:
                out.append(f"[s{i:03d}] {side(pi)} devolved {nm(lg.get('cardIdTarget'))}")
            elif t == 15:
                cid = lg.get("cardId")
                mv = moves.get(cid)
                mv_s = f" (moves: {', '.join(f'{m}[{dm}]' for m, dm in mv)})" if mv else ""
                out.append(f"[s{i:03d}] {side(pi)} ATTACK with {nm(cid)} attackId={lg.get('attackId')}{mv_s}")
            elif t == 16:
                v = lg.get("value")
                cid = lg.get("cardId")
                kind = "damage counters" if lg.get("putDamageCounter") else "HP"
                out.append(f"[s{i:03d}]   {side(pi)} {nm(cid)} {kind} {v:+d}")
            elif t in (17, 18, 19, 20, 21):
                cond = {17: "POISONED", 18: "BURNED", 19: "ASLEEP", 20: "PARALYZED", 21: "CONFUSED"}[t]
                rec = " recovered" if lg.get("isRecover") else ""
                out.append(f"[s{i:03d}] {side(pi)} {nm(lg.get('cardId'))} {cond}{rec}")
            elif t == 22:
                out.append(f"[s{i:03d}] COIN: {'HEADS' if lg.get('head') else 'TAILS'}")
            elif t == 23:
                r = lg.get("result")
                who = "DRAW" if r == 2 else ("US WIN" if r == our_index else "OPP WIN")
                out.append(f"[s{i:03d}] **** RESULT: {who} — reason: "
                           f"{RESULT_REASON.get(lg.get('reason'), lg.get('reason'))} ****")

    # main pass: only FRESH windows of OUR observation (ACTIVE = we are being
    # prompted right now; a DONE window counts if it differs from the previous one)
    our_windows = []
    prev_logs = None
    last_our_prompt = -1
    for i, pair in enumerate(steps):
        entry = pair[our_index] if our_index < len(pair) else None
        if not isinstance(entry, dict):
            continue
        obs = entry.get("observation") or {}
        logs = obs.get("logs") or []
        status = entry.get("status")
        if status == "ACTIVE":
            last_our_prompt = i
        elif not (status == "DONE" and logs != prev_logs):
            continue
        prev_logs = logs
        our_windows.append((i, logs))
        emit_logs(i, logs)
        emit_snapshot(i, obs.get("current"))

    # tail pass: a game that ends during OPP's turn never reaches our observation —
    # recover those events from the opponent's fresh windows after our last prompt
    seen = set()
    for _, logs in our_windows[-6:]:
        for lg in logs:
            if isinstance(lg, dict):
                seen.add(json.dumps(lg, sort_keys=True))
    tail_start = len(out)
    prev_opp = None
    for i in range(last_our_prompt + 1, len(steps)):
        pair = steps[i]
        entry = pair[opp_index] if opp_index < len(pair) else None
        if not isinstance(entry, dict) or entry.get("status") not in ("ACTIVE", "DONE"):
            continue
        obs = entry.get("observation") or {}
        logs = obs.get("logs") or []
        if logs == prev_opp:
            continue
        prev_opp = logs
        emit_logs(i, logs, skip_keys=seen)
        emit_snapshot(i, obs.get("current"))
    if len(out) > tail_start:
        out.insert(tail_start, "== ENDGAME после нашего последнего решения "
                               "(восстановлено из наблюдений соперника) ==")

    out.append(f"**** FINAL: {res} for us ****")
    return "\n".join(out) + "\n"


def main():
    names, moves = load_cards()
    if sys.argv[1] == "--batch":
        dl = sys.argv[2]
        outdir = os.path.join(dl, "replay_digest")
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(dl, "index.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            ep = r["episode_id"]
            rp = os.path.join(dl, "replays", f"{ep}.json")
            if not os.path.exists(rp):
                print(f"skip {ep}: no replay", file=sys.stderr)
                continue
            hdr = (f"copy={r['copy']} opponent={r['opp_team_name']} "
                   f"(sub {r['opp_submission_id']}) start={r['create_time']}")
            txt = render(rp, int(r["our_index"]), names, moves, header_extra=hdr)
            with open(os.path.join(outdir, f"{ep}.txt"), "w") as f:
                f.write(txt)
        print(f"wrote {len(rows)} digests to {outdir}")
    else:
        replay, our_index = sys.argv[1], int(sys.argv[2])
        txt = render(replay, our_index, names, moves)
        if len(sys.argv) > 3:
            open(sys.argv[3], "w").write(txt)
        else:
            sys.stdout.write(txt)


if __name__ == "__main__":
    main()
