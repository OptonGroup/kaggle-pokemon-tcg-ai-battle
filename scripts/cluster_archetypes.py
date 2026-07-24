#!/usr/bin/env python3
"""Data-driven archetype discovery: group 60-card decklists by cosine similarity
and name each group after the Pokemon most distinctive to it.

Why this exists: scripts/classify_opponent_archetype.py matches decks against a
hand-written table of key-card IDs (ARCHETYPE_IDS / NAME_KEYWORDS). Anything the
table doesn't know comes out "UNKNOWN" — so a newly-risen deck stays invisible in
forensics exactly when it matters most. Clustering needs no table: a deck nobody
wrote a rule for still gets found, grouped and named.

Ported from meetparekh1's "Know Your Bracket" Kaggle notebook (union-find over a
cosine-similarity graph; naming by idf-style distinctiveness, then copy count,
then evolution chain). Naming by raw HP was tried first and is wrong: fat generic
support Pokemon are everywhere, so an Alakazam deck comes out named "Fezandipiti ex".

Used as a library by classify_opponent_archetype.py (--cluster-unknown / --audit).
"""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Two decks count as the same archetype above this cosine similarity. 0.85 leaves
# room for the ~4-6 flex slots real lists differ by without merging distinct decks.
CLUSTER_THRESHOLD = 0.85


class CardMeta:
    """name / HP / evolution links, read out of EN_Card_Data.csv."""

    def __init__(self, csv_path: Path | None = None):
        csv_path = csv_path or (REPO / "data" / "EN_Card_Data.csv")
        self.name: dict[int, str] = {}
        self.hp: dict[int, int] = {}
        children: dict[str, set[str]] = defaultdict(set)
        with open(csv_path, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                try:
                    cid = int(row["Card ID"])
                except (TypeError, ValueError, KeyError):
                    continue
                nm = (row.get("Card Name") or "").strip()
                try:
                    hp = int(str(row.get("HP", "")).strip())
                except (TypeError, ValueError):
                    hp = 0
                self.name[cid] = nm
                self.hp[cid] = hp
                prev = (row.get("Previous stage") or "").strip()
                if nm and prev and prev.lower() not in ("n/a", "none", ""):
                    children[prev].add(nm)
        self.children = children
        # A Pokemon is the thing with HP. Do NOT test the stage column for "Pokemon":
        # it reads "Basic Pokemon" but ALSO "Pokemon Tool", which is a Trainer card.
        self.is_pokemon = {cid: h > 0 for cid, h in self.hp.items()}
        # Whole evolution families, for disambiguating two clusters that share a
        # headline Pokemon: excluding just the one card would make the "second"
        # signature card its own pre-evolution, which distinguishes nothing.
        adj: dict[str, set[str]] = defaultdict(set)
        for p, cs in children.items():
            for c in cs:
                adj[p].add(c)
                adj[c].add(p)
        self.family: dict[str, frozenset[str]] = {}
        for start in list(adj):
            if start in self.family:
                continue
            stack, seen = [start], set()
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                stack.extend(adj[x] - seen)
            fs = frozenset(seen)
            for x in fs:
                self.family[x] = fs

    def n_pokemon(self) -> int:
        return sum(1 for v in self.is_pokemon.values() if v)


def load_card_meta(csv_path: Path | None = None) -> CardMeta:
    meta = CardMeta(csv_path)
    if meta.n_pokemon() < 50:
        raise RuntimeError(
            f"Only {meta.n_pokemon()} Pokemon detected in the card CSV — every deck "
            "would come out unnamed. Check the file's columns."
        )
    return meta


def cosine(a: list[int], b: list[int]) -> float:
    ca, cb = Counter(a), Counter(b)
    dot = sum(v * cb[k] for k, v in ca.items())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def cluster(decks: list[list[int]], threshold: float = CLUSTER_THRESHOLD) -> list[int]:
    """Union-find over the similarity graph. Returns a root index per deck."""
    parent = list(range(len(decks)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(decks)):
        for j in range(i + 1, len(decks)):
            ri, rj = find(i), find(j)
            if ri != rj and cosine(decks[i], decks[j]) >= threshold:
                parent[max(ri, rj)] = min(ri, rj)
    return [find(i) for i in range(len(decks))]


def name_cluster(members: list[list[int]], meta: CardMeta, doc_freq: Counter,
                 n_groups: int, exclude: frozenset[str] | set[str] = frozenset()) -> str:
    """Name a group after the Pokemon most distinctive to it.

    Score each Pokemon by how reliably this group runs it against how many groups
    run it at all — ubiquitous support cards score near zero automatically. Then
    take the card the deck runs the MOST copies of within that signature line
    (a deck plays 3-4 of its engine and exactly one splashable support), and walk
    the evolution chain up so the label isn't a middle stage.
    """
    n = len(members)
    seen, copies = Counter(), Counter()
    for deck in members:
        for cid in set(deck):
            if meta.is_pokemon.get(cid):
                seen[cid] += 1
        for cid in deck:
            if meta.is_pokemon.get(cid):
                copies[cid] += 1
    if not seen:
        any_card = Counter()
        for deck in members:
            any_card.update(deck)
        return meta.name.get(any_card.most_common(1)[0][0], "unnamed deck") if any_card else "unnamed deck"

    core = [cid for cid, c in seen.items() if c >= 0.6 * n] or list(seen)
    if exclude:
        core = [c for c in core if meta.name.get(c) not in exclude] or core

    def distinctiveness(cid: int) -> float:
        inside = seen[cid] / n
        spread = math.log(n_groups / (1 + doc_freq.get(cid, 0)))
        return inside * spread

    best = max(distinctiveness(c) for c in core)
    signature = [c for c in core if distinctiveness(c) >= 0.6 * best] or core
    pick = max(signature, key=lambda c: (copies[c] / n, meta.hp.get(c, 0)))
    name = meta.name.get(pick, "unnamed deck")

    present = {meta.name.get(c) for c in seen}
    for _ in range(3):   # Basic -> Stage 1 -> Stage 2
        nxt = [c for c in meta.children.get(name, ()) if c in present and c not in exclude]
        if not nxt:
            break
        name = max(nxt, key=lambda c: max((meta.hp.get(i, 0) for i in seen
                                           if meta.name.get(i) == c), default=0))
    return name


def cluster_and_name(decks: list[list[int]], meta: CardMeta,
                     threshold: float = CLUSTER_THRESHOLD) -> list[str]:
    """One archetype label per input deck, in input order."""
    if not decks:
        return []
    roots = cluster(decks, threshold)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, root in enumerate(roots):
        groups[root].append(idx)

    doc_freq = Counter()   # how many distinct clusters each Pokemon turns up in
    for idxs in groups.values():
        for cid in {c for i in idxs for c in decks[i] if meta.is_pokemon.get(c)}:
            doc_freq[cid] += 1
    n_groups = max(len(groups), 2)

    naming = {root: name_cluster([decks[i] for i in idxs], meta, doc_freq, n_groups)
              for root, idxs in groups.items()}
    # Two clusters can share a headline Pokemon while being genuinely different
    # decks; name the collision by each cluster's SECOND signature card rather
    # than by an uninformative "#2".
    dupes = Counter(naming.values())
    for root in sorted(groups, key=lambda r: -len(groups[r])):
        if dupes[naming[root]] > 1:
            second = name_cluster([decks[i] for i in groups[root]], meta, doc_freq, n_groups,
                                  exclude=meta.family.get(naming[root], {naming[root]}))
            if second and second != naming[root]:
                naming[root] = f"{naming[root]} + {second}"
    return [naming[r] for r in roots]
