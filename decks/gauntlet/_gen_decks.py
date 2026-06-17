"""Generate pilot-SAFE Mega Lucario ex deck variants for the Phase-3 gauntlet.

The v002 pilot (src/agents/heuristic_agent.py) is HARD-CODED to Mega Lucario ex
(card IDs 673-678, Mega Brave attack 983, fighting weakness x2). It scores known
cards by special-case logic and unknown cards by generic fallbacks. To keep the
pilot working we ONLY recount cards already in the base list — no new cards with
costs/effects the pilot can't drive.

Legality: <=4 copies of any non-basic-energy card by ID, exactly 60 cards,
>=1 Basic Pokemon. Basic Fighting Energy (id 6) is unlimited.

Run locally (pure Python, no cg needed):
    python decks/gauntlet/_gen_decks.py
"""
import os

# Card IDs (mirror the pilot's constants)
Makuhita, Hariyama = 673, 674
Lunatone, Solrock = 675, 676
Riolu, Mega_Lucario_ex = 677, 678
Dusk_Ball = 1102
Switch = 1123
Premium_Power_Pro = 1141
Fighting_Gong = 1142
Poke_Pad = 1152
Hero_Cape = 1159
Boss_Orders = 1182
Carmine = 1192
Lillie = 1227
Gravity_Mountain = 1252
Fighting_Energy = 6  # basic energy — unlimited copies

# Basic Pokemon (for the >=1 Basic legality check)
BASICS = {Makuhita, Lunatone, Solrock, Riolu, Mega_Lucario_ex}

BASE = {
    Makuhita: 2, Hariyama: 2,
    Lunatone: 2, Solrock: 3,
    Riolu: 3, Mega_Lucario_ex: 4,
    Dusk_Ball: 4,
    Switch: 2,
    Premium_Power_Pro: 4,
    Fighting_Gong: 4,
    Poke_Pad: 4,
    Hero_Cape: 1,
    Boss_Orders: 2,
    Carmine: 4,
    Lillie: 4,
    Gravity_Mountain: 2,
    Fighting_Energy: 13,
}


def variant(changes: dict[int, int]) -> dict[int, int]:
    d = dict(BASE)
    for cid, cnt in changes.items():
        if cnt <= 0:
            d.pop(cid, None)
        else:
            d[cid] = cnt
    return d


# Each variant = a single small, pilot-safe recount vs BASE.
VARIANTS = {
    "00_base":      BASE,                                          # = v002_lucario deck
    "01_riolu4":    variant({Riolu: 4, Fighting_Energy: 12}),      # thicker main line, -1 energy
    "02_boss3":     variant({Boss_Orders: 3, Gravity_Mountain: 1}),# more gust, -1 stadium
    "03_cape2":     variant({Switch: 3, Fighting_Energy: 12}),       # +1 Switch -1 energy (Hero_Cape is Ace Spec → max 1 copy; original intent was +Cape but illegal)
    "04_switch3":   variant({Switch: 3, Poke_Pad: 3}),             # more repositioning, -1 draw
}


ACE_SPEC_IDS = {Hero_Cape}  # Ace Spec cards: max 1 copy each

def validate(name: str, counts: dict[int, int]) -> None:
    total = sum(counts.values())
    assert total == 60, f"{name}: {total} cards (need 60)"
    for cid, cnt in counts.items():
        if cid != Fighting_Energy:
            assert cnt <= 4, f"{name}: card {cid} x{cnt} > 4"
        if cid in ACE_SPEC_IDS:
            assert cnt <= 1, f"{name}: Ace Spec card {cid} x{cnt} > 1 (Ace Spec limit)"
    assert any(counts.get(b, 0) > 0 for b in BASICS), f"{name}: no Basic Pokemon"


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for name, counts in VARIANTS.items():
        validate(name, counts)
        lines = []
        for cid, cnt in counts.items():
            lines.extend([str(cid)] * cnt)
        path = os.path.join(out_dir, f"{name}.csv")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        diff = {cid: counts.get(cid, 0) for cid in set(BASE) | set(counts)
                if counts.get(cid, 0) != BASE.get(cid, 0)}
        print(f"  {name:14s} 60 cards OK   diff vs base: {diff or '(none)'}")
    print(f"Wrote {len(VARIANTS)} decks to {out_dir}")


if __name__ == "__main__":
    main()
