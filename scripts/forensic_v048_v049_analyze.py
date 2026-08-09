"""Forensic analyzer for v048 vs v049 live replays.

Reads out/forensic_v048_v049/{v048,v049}/{replays,logs} and emits:
  - per-episode records (out/forensic_v048_v049/{v}/analysis.json)
  - aggregate tables printed to stdout + out/forensic_v048_v049/AGGREGATE.md

Per episode extracts:
  result, our_idx (TeamNames), num_steps
  opp_real_archetype (from ALL cards opp ever revealed on board+discard)
  detected_label_sequence + final + flip_flopped (from [dbg] OPPONENT DECK lines)
  mismatch (detected_final vs opp_real)
  US bench-wipe events (bench=0 after setup; bench=0 at end)
  prize progress (our_taken, opp_taken), end US/opp active+bench
  judge plays count, opp max hand, attack count
"""
import json, os, re, collections, statistics

ROOT = "out/forensic_v048_v049"

# Archetype line sets (from v049 main.py)
LINES = {
    "crustle": {344, 345, 532},
    "starmie": {1030, 1031},
    "lucario": {677, 678},
    "hop": {288, 289, 299, 304, 307, 308, 309, 310, 878, 879},
    "grimmsnarl": {646, 647, 648},
    "dipplin": {89, 90, 92, 93, 247},
    "abomasnow": {722, 723},
    "dragapult": {119, 120, 121},
    "team_rocket_mewtwo": {431},
    "n_zoroark": {292, 293},
    "cynthia_garchomp": {379, 380, 381},
    "iono_bellibolt": {268, 269},
    "alakazam": {741, 742, 743},
    "archaludon_mirror": {169, 190, 666},  # Duraludon/Archaludon ex/Cinderace
    "hariyama": {673, 674},
}
# priority: mirror and alakazam are most diagnostic; single-id lines are weak evidence
SINGLE_ID_LINES = {"team_rocket_mewtwo", "n_zoroark", "iono_bellibolt", "starmie", "dragapult"}

def classify_opp(revealed):
    """revealed: set of card ids opp ever showed. Return best archetype or None."""
    hits = []
    for name, ids in LINES.items():
        inter = revealed & ids
        if inter:
            # weight: count of distinct matched ids; mirror/alakazam weighted higher
            w = len(inter)
            if name in ("archaludon_mirror", "alakazam"):
                w += 0.5
            hits.append((w, name, sorted(inter)))
    if not hits:
        return None, []
    hits.sort(reverse=True)
    return hits[0][1], hits

def parse_dbg(logpath):
    """Return list of [dbg] lines."""
    try:
        d = json.load(open(logpath))
    except Exception:
        return []
    out = []
    for step in d:
        ents = step if isinstance(step, list) else [step]
        for ent in ents:
            if not isinstance(ent, dict):
                continue
            blob = (ent.get("stderr") or "") + (ent.get("stdout") or "")
            for ln in blob.splitlines():
                if "[dbg]" in ln:
                    out.append(ln.strip())
    return out

DECK_RE = re.compile(r"OPPONENT DECK:\s*(\S+)?\s*->\s*(\S+)\s*\(source=(\w+)")
CHOICE_RE = re.compile(r"turn=(\d+)\s+type=(\d+)\s+matchup=(\S+)\s+chosen=\[([^\]]*)\]\s+top_score=(-?\d+)\s+reason=(.*)")

def matchup_trace(dbg):
    seq = []
    for ln in dbg:
        m = DECK_RE.search(ln)
        if m:
            seq.append(m.group(2))
    return seq

def analyze_episode(v, ep_id, idx_from_summary):
    repath = f"{ROOT}/{v}/replays/episode-{ep_id}-replay.json"
    logpath = f"{ROOT}/{v}/logs/episode-{ep_id}-agent-{idx_from_summary}-logs.json"
    r = json.load(open(repath))
    info = r["info"]
    teams = info["TeamNames"]
    our_idx = teams.index("Maxim Simakov") if "Maxim Simakov" in teams else idx_from_summary
    opp_idx = 1 - our_idx
    opp_name = teams[opp_idx] if opp_idx < len(teams) else "?"
    rewards = r["rewards"]
    result = "WIN" if rewards[our_idx] > 0 else ("LOSS" if rewards[our_idx] < 0 else "DRAW")
    steps = r["steps"]

    # scan board across steps from OUR perspective
    revealed = set()
    bench_zero_steps = []      # step indices where US bench had 0 pokemon after turn>=1
    bench_zero_any = False
    our_active_zero_steps = [] # US got KO'd / no active
    end_us = None; end_opp = None
    our_prizes_end = 6; opp_prizes_end = 6
    our_taken = opp_taken = 0
    opp_max_hand = 0
    us_ever_had_pokemon = False
    last_turn = 0
    # track prize progress over time
    our_prize_seq = []; opp_prize_seq = []
    for si, st in enumerate(steps):
        if len(st) <= our_idx: continue
        obs = st[our_idx].get("observation")
        if not obs: continue
        cur = obs.get("current")
        if not cur: continue
        last_turn = cur.get("turn", last_turn)
        players = cur.get("players", [])
        if len(players) < 2: continue
        us = players[our_idx]; opp = players[opp_idx]
        # opp revealed cards: active, bench, discard (NOT hand, NOT deck)
        for zone in ("active", "bench", "discard"):
            for c in (opp.get(zone) or []):
                if isinstance(c, dict) and "id" in c:
                    revealed.add(c["id"])
        bench_n = len([b for b in (us.get("bench") or []) if b])
        active_n = len([a for a in (us.get("active") or []) if a])
        if active_n > 0 or bench_n > 0:
            us_ever_had_pokemon = True
        if cur.get("turn", 0) >= 1 and bench_n == 0 and us_ever_had_pokemon:
            bench_zero_steps.append(si); bench_zero_any = True
        if active_n == 0 and us_ever_had_pokemon:
            our_active_zero_steps.append(si)
        oh = opp.get("handCount") or 0
        if oh > opp_max_hand: opp_max_hand = oh
        our_pr = len(us.get("prize") or [])
        opp_pr = len(opp.get("prize") or [])
        our_prize_seq.append((si, 6 - our_pr)); opp_prize_seq.append((si, 6 - opp_pr))
        our_prizes_end = our_pr; opp_prizes_end = opp_pr
        end_us = us; end_opp = opp
    our_taken = 6 - our_prizes_end; opp_taken = 6 - opp_prizes_end

    real_arch, hits = classify_opp(revealed)

    dbg = parse_dbg(logpath)
    seq = matchup_trace(dbg)
    final_det = seq[-1] if seq else "generic"
    flip = len(set(seq)) > 2 or (len(seq) >= 3 and seq != [seq[0]]*len(seq) and len(set(seq))>1 and any(seq[i]!=seq[i+1] for i in range(len(seq)-1)))
    # simpler flip def: label changes more than once, or oscillates
    changes = sum(1 for i in range(1,len(seq)) if seq[i]!=seq[i-1])
    flip = changes >= 2

    judge_plays = sum(1 for ln in dbg if "Judge" in ln and ("reason=Judge" in ln or "defuse" in ln or "hand-disruption" in ln))
    attack_count = sum(1 for ln in dbg if CHOICE_RE.search(ln) and "attack" in ln.lower())

    # end-state board
    def endcnt(p):
        if not p: return (-1,-1,-1,-1)
        return (len([a for a in (p.get("active") or []) if a]),
                len([b for b in (p.get("bench") or []) if b]),
                p.get("deckCount",-1), p.get("handCount",-1))
    end_us_cnt = endcnt(end_us); end_opp_cnt = endcnt(end_opp)

    rec = {
        "version": v, "ep": ep_id, "our_idx": our_idx, "opp_name": opp_name,
        "result": result, "num_steps": len(steps), "last_turn": last_turn,
        "opp_real_archetype": real_arch, "opp_revealed_ids": sorted(revealed),
        "opp_class_hits": [(n, sorted(i)) for _,n,i in hits],
        "detected_seq": seq, "detected_final": final_det, "flip_flopped": flip,
        "mismatch": (final_det != real_arch) if real_arch else None,
        "bench_zero_after_setup": bench_zero_any,
        "bench_zero_steps_count": len(bench_zero_steps),
        "end_us_active": end_us_cnt[0], "end_us_bench": end_us_cnt[1],
        "end_opp_active": end_opp_cnt[0], "end_opp_bench": end_opp_cnt[1],
        "our_prizes_taken": our_taken, "opp_prizes_taken": opp_taken,
        "opp_max_hand": opp_max_hand,
        "judge_plays": judge_plays, "attack_count": attack_count,
        "end_us_deck": end_us_cnt[2],
    }
    return rec

def load_index(v):
    return json.load(open(f"{ROOT}/{v}/dossier_index.json"))

for v in ["v048", "v049"]:
    idx = load_index(v)
    recs = []
    for e in idx:
        try:
            recs.append(analyze_episode(v, e["episode_id"], e["our_idx"]))
        except Exception as ex:
            recs.append({"version": v, "ep": e["episode_id"], "error": str(ex)})
    json.dump(recs, open(f"{ROOT}/{v}/analysis.json","w"), indent=1, ensure_ascii=False)
    print(f"\n[{v}] analyzed {len(recs)} episodes, errors={sum(1 for r in recs if 'error' in r)}")