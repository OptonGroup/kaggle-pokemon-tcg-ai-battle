#!/usr/bin/env python3
"""
eval_deckbelief_holdout.py — измеряет реальную успешность DeckBeliefNet на
СВЕЖИХ деках, которых модель не видела при обучении, и считает top-1 archetype
accuracy + confusion-матрицу — то, чего нет в scripts/deckbelief_gate.py
(тот гоняет только 12 val-дек из data/belief/val.pt и не считает точность
архетип-классификации).

Без движка (libcg) — чистый torch, работает на macOS. Источник свежих дек:
  data/deck_scan/decks.jsonl (1880 реальных top-250 эпизодов, обе стороны) +
  submissions/*/deck.csv (пул гейта). Дедуп по сигнатуре (sorted multiset 60 карт).

Что считается, per milestone (1/2/4/8/12/20), N=--samples случайных reveal-ов на деку:
  - L1 (gate-определение: |pred_hidden - true_hidden|.sum() / (2*hidden_total))
  - recall@10 (доля true-hidden массы в top-10 предсказанных hidden-слотах)
  - top-1 archetype accuracy:
      * raw_argmax: ближайший прототип по cosine (без confidence-gate)
      * classified: с реальным порогом _classify_belief (sim>=0.55, margin>=0.08,
        min_evidence>=8) — может вернуть None ("generic")
  - confusion-матрица true -> predicted (raw_argmax) на milestone=8/12/20

Сплит IN/OUT-of-training: сигнатуры дек из data/belief/corpus_decks_{train,val}.pt
(142+12 = 154 уникальных дек обучения). IN = дека в обучающем корпусе, OUT = свежая.
Сравнение L1/recall@10 IN vs OUT показывает зазор запоминание vs обобщение.

ВАЖНО про архетипы: belief-net имеет 9 прототипов (crustle, starmie, lucario, hop,
alakazam, grimmsnarl, dipplin, abomasnow, archaludon). TRUE_LINES содержит 10
архетипов — archaludon_mirror {169,190,666} и dragapult {119,120,121} прототипа НЕ
ИМЕЮТ (они ловятся board-only веткой v048 L2, не нейросетью). Поэтому для этих двух
"правильный" ответ belief-net = None/generic; raw_argmax для них всегда "неверен" по
букве (но это не баг — net для них и не должен срабатывать). Считаем их отдельным
бакетом "board-only (no prototype)", не вмишиваем в accuracy по 9 прототипам.

Usage:
  .venv/bin/python scripts/eval_deckbelief_holdout.py
  .venv/bin/python scripts/eval_deckbelief_holdout.py --samples 200 --milestones 1 2 4 8 12 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import torch

from deckbelief_common import DeckBeliefNet, N_CARDS, build_card_caps_and_ace_mask

# 10 архетипов из scripts/deckbelief_l0_diagnose.py TRUE_LINES (включая L2-ветки).
TRUE_LINES = {
    "crustle": {344, 345, 532},
    "starmie": {1030, 1031},
    "lucario": {677, 678},
    "hop": {288, 289, 299, 304, 307, 308, 309, 310, 878, 879},
    "grimmsnarl": {646, 647, 648},
    "dipplin": {89, 90, 92, 93, 247},
    "abomasnow": {722, 723},
    "alakazam": {741, 742, 743},
    "archaludon_mirror": {169, 190, 666},
    "dragapult": {119, 120, 121},
}

# Архетипы, у которых ЕСТЬ прототип в deckbelief_prototypes.json (9 шт).
# archaludon_mirror / dragapult ловятся board-only, прототипа нет.
PROTOTYPE_ARCHETYPES = {
    "crustle", "starmie", "lucario", "hop", "alakazam",
    "grimmsnarl", "dipplin", "abomasnow", "archaludon",
}

DEFAULT_MILESTONES = (1, 2, 4, 8, 12, 20)


def _deck_signature(deck: list[int]) -> tuple[int, ...]:
    return tuple(sorted(deck))


def _true_archetype(deck: list[int]) -> str:
    ids = set(deck)
    hits = [name for name, line in TRUE_LINES.items() if ids & line]
    if len(hits) == 1:
        return hits[0]
    if hits:
        return "ambiguous:" + "+".join(sorted(hits))
    return "generic"


def _load_corpus_signatures(data_dir: Path) -> set[tuple[int, ...]]:
    sigs: set[tuple[int, ...]] = set()
    for name in ("corpus_decks_train.pt", "corpus_decks_val.pt"):
        p = data_dir / name
        if not p.exists():
            continue
        decks = torch.load(p, weights_only=True)
        # corpus_decks_*.pt — тензор (n_decks, 60) ID-карт
        if isinstance(decks, torch.Tensor):
            for row in decks.tolist():
                sigs.add(_deck_signature([int(x) for x in row]))
    return sigs


def _load_eval_decks() -> list[tuple[list[int], str]]:
    """Свежие деки из data/deck_scan/decks.jsonl + submissions/*/deck.csv.
    Возвращает [(deck_ids, source_tag), ...] с дедупом по сигнатуре."""
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[list[int], str]] = []

    # 1) data/deck_scan/decks.jsonl — реальные top-250 эпизоды (обе стороны)
    scan = REPO / "data" / "deck_scan" / "decks.jsonl"
    if scan.exists():
        with open(scan) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for deck in rec.get("decks", []):
                    if not isinstance(deck, list) or len(deck) != 60:
                        continue
                    sig = _deck_signature(deck)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    out.append(([int(x) for x in deck], "deck_scan"))

    # 2) submissions/*/deck.csv — пул гейта (наши + оппоненты)
    for deck_csv in sorted((REPO / "submissions").glob("*/deck.csv")):
        try:
            with open(deck_csv) as f:
                deck = [int(x.strip()) for x in f if x.strip()]
        except (ValueError, OSError):
            continue
        if len(deck) != 60:
            continue
        sig = _deck_signature(deck)
        if sig in seen:
            continue
        seen.add(sig)
        out.append((deck, "submissions"))

    return out


def _cosine_sim_sparse(proto: dict, vec: list[float]) -> float:
    num = 0.0
    a_sq = 0.0
    for cid_str, av in proto.items():
        cid = int(cid_str)
        num += av * vec[cid]
        a_sq += av * av
    b_sq = sum(v * v for v in vec)
    denom = (a_sq ** 0.5) * (b_sq ** 0.5)
    return num / denom if denom > 1e-9 else 0.0


def _deck_to_count_vector(deck: list[int]) -> torch.Tensor:
    v = torch.zeros(N_CARDS)
    for cid in deck:
        if 0 <= cid < N_CARDS:
            v[cid] += 1
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="out/deckbelief/model.pth")
    ap.add_argument("--prototypes", default="submissions/v048_deckbelief_judgefix/deckbelief_prototypes.json")
    ap.add_argument("--data-dir", default="data/belief")
    ap.add_argument("--samples", type=int, default=200,
                    help="случайных reveal-ов на деку на milestone")
    ap.add_argument("--milestones", type=int, nargs="+", default=list(DEFAULT_MILESTONES))
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--confusion-milestones", type=int, nargs="+", default=[8, 12, 20])
    ap.add_argument("--out", default="out/deckbelief/holdout_report.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    data_dir = REPO / args.data_dir
    card_caps, ace_mask = build_card_caps_and_ace_mask(csv_path=str(REPO / "data" / "EN_Card_Data.csv"))
    model = DeckBeliefNet(ace_mask, card_caps)
    state = torch.load(REPO / args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with open(REPO / args.prototypes) as f:
        proto_data = json.load(f)
    prototypes = proto_data.get("prototypes", {})
    sim_threshold = proto_data.get("sim_threshold", 0.55)
    sim_margin = proto_data.get("sim_margin", 0.08)
    min_evidence = proto_data.get("min_evidence", 8)
    proto_names = sorted(prototypes.keys())
    print(f"Loaded model + {len(proto_names)} prototypes: {proto_names}")
    print(f"  sim_threshold={sim_threshold} sim_margin={sim_margin} min_evidence={min_evidence}")

    corpus_sigs = _load_corpus_signatures(data_dir)
    print(f"Training corpus signatures: {len(corpus_sigs)}")

    eval_decks = _load_eval_decks()
    print(f"Eval decks (deduped): {len(eval_decks)} "
          f"(deck_scan + submissions/*/deck.csv)")

    # Разметка: true archetype + IN/OUT split
    labeled = []
    in_count = out_count = 0
    arch_counts = Counter()
    for deck, src in eval_decks:
        truth = _true_archetype(deck)
        sig = _deck_signature(deck)
        is_in = sig in corpus_sigs
        if is_in:
            in_count += 1
        else:
            out_count += 1
        arch_counts[truth] += 1
        labeled.append({"deck": deck, "truth": truth, "in_corpus": is_in, "src": src})

    print(f"  IN training corpus: {in_count}, OUT (fresh): {out_count}")
    print(f"  True-archetype distribution:")
    for name, c in arch_counts.most_common():
        tag = "✓proto" if name in PROTOTYPE_ARCHETYPES else ("board-only" if name in TRUE_LINES else "")
        print(f"    {name:>22}: {c:>4}  {tag}")
    print()

    # Метрики
    # per milestone: aggregates
    agg = {m: {
        "n_samples": 0,
        "l1_sum": 0.0, "recall10_sum": 0.0,
        # raw argmax top-1 over archetypes that HAVE prototypes
        "raw_correct_proto": 0, "raw_total_proto": 0,
        # classified (with confidence gate): correct/incorrect/none
        "cls_correct_proto": 0, "cls_incorrect_proto": 0, "cls_none_proto": 0,
        "cls_none_boardonly": 0, "cls_incorrect_boardonly": 0,
        "raw_total_boardonly": 0,
        # IN/OUT split (proto archetypes only), for memorization-vs-generalization
        "in_raw_correct": 0, "in_raw_total": 0, "in_l1_sum": 0.0, "in_rec_sum": 0.0, "in_n": 0,
        "out_raw_correct": 0, "out_raw_total": 0, "out_l1_sum": 0.0, "out_rec_sum": 0.0, "out_n": 0,
    } for m in args.milestones}

    # confusion: true -> predicted(raw argmax proto name), per milestone
    confusion = {m: defaultdict(Counter) for m in args.confusion_milestones}

    # Precompute prototype dense vectors + norms for vectorized cosine.
    proto_mat = torch.zeros(len(proto_names), N_CARDS)
    for i, name in enumerate(proto_names):
        for cid_str, av in prototypes[name].items():
            cid = int(cid_str)
            if 0 <= cid < N_CARDS:
                proto_mat[i, cid] = float(av)
    proto_norm = proto_mat.norm(dim=-1)  # (P,)

    for d in labeled:
        deck = d["deck"]
        truth = d["truth"]
        truth_vec = _deck_to_count_vector(deck)  # (N,)
        has_proto = truth in PROTOTYPE_ARCHETYPES
        is_boardonly = truth in {"archaludon_mirror", "dragapult"}
        deck_tensor = torch.tensor(deck, dtype=torch.long)

        for m in args.milestones:
            if m > 60 or m <= 0:
                continue
            S = args.samples
            # Build (S, N) reveal matrix via random draws with multiplicity.
            rev = torch.zeros(S, N_CARDS)
            for s in range(S):
                sample = rng.sample(deck, m)
                for cid in sample:
                    if 0 <= cid < N_CARDS:
                        rev[s, cid] += 1
            with torch.no_grad():
                pred = model(rev)  # (S, N)

            true_h = (truth_vec.unsqueeze(0) - rev).clamp_min(0)  # (S, N)
            pred_h = (pred - rev).clamp_min(0)
            hidden_total = true_h.sum(dim=-1).clamp_min(1.0)  # (S,)
            l1 = (pred_h - true_h).abs().sum(dim=-1) / (2 * hidden_total)  # (S,)
            top10 = torch.topk(pred_h, k=min(10, N_CARDS), dim=-1).indices
            rec10 = torch.gather(true_h, -1, top10).sum(dim=-1) / hidden_total  # (S,)

            # cosine to prototypes, vectorized: pred (S,N) · proto_mat (P,N)
            pred_norm = pred.norm(dim=-1)  # (S,)
            num = pred @ proto_mat.t()  # (S, P)
            denom = pred_norm.unsqueeze(1) * proto_norm.unsqueeze(0)  # (S, P)
            sims = num / denom.clamp_min(1e-9)  # (S, P)
            top2 = sims.topk(2, dim=-1)
            top_idx = top2.indices[:, 0]  # (S,)
            best_sim = top2.values[:, 0]
            second_sim = top2.values[:, 1] if sims.shape[1] > 1 else torch.full((S,), -1.0)
            raw_names = [proto_names[i] for i in top_idx.tolist()]

            a = agg[m]
            a["n_samples"] += S
            a["l1_sum"] += float(l1.sum().item())
            a["recall10_sum"] += float(rec10.sum().item())

            # confidence gate mask
            gate = (m >= min_evidence) & (best_sim >= sim_threshold) & ((best_sim - second_sim) >= sim_margin)

            if has_proto:
                a["raw_total_proto"] += S
                raw_correct_mask = torch.tensor([rn == truth for rn in raw_names])
                a["raw_correct_proto"] += int(raw_correct_mask.sum().item())
                fired = gate & raw_correct_mask
                a["cls_correct_proto"] += int(fired.sum().item())
                a["cls_incorrect_proto"] += int((gate & ~raw_correct_mask).sum().item())
                a["cls_none_proto"] += int((~gate).sum().item())
                # IN/OUT split
                key = "in" if d["in_corpus"] else "out"
                a[f"{key}_raw_total"] += S
                a[f"{key}_raw_correct"] += int(raw_correct_mask.sum().item())
                a[f"{key}_l1_sum"] += float(l1.sum().item())
                a[f"{key}_rec_sum"] += float(rec10.sum().item())
                a[f"{key}_n"] += S
            elif is_boardonly:
                a["raw_total_boardonly"] += S
                fired = gate
                a["cls_incorrect_boardonly"] += int(fired.sum().item())  # FP — плохо
                a["cls_none_boardonly"] += int((~fired).sum().item())

            if m in confusion:
                for rn in raw_names:
                    pred_label = rn if (has_proto or is_boardonly) else "generic"
                    confusion[m][truth][pred_label] += 1

    # --- Отчёт ---
    print("=" * 78)
    print(f"{'milestone':>9} {'n':>7} {'L1':>7} {'rec@10':>7} "
          f"{'raw_acc':>8} {'cls_acc':>8} {'cls_none':>8} {'boardonly_FP':>13}")
    rows = []
    for m in args.milestones:
        a = agg[m]
        n = a["n_samples"]
        if n == 0:
            continue
        l1 = a["l1_sum"] / n
        rec = a["recall10_sum"] / n
        raw_acc = a["raw_correct_proto"] / max(1, a["raw_total_proto"])
        tp = a["cls_correct_proto"] + a["cls_incorrect_proto"]
        cls_acc = a["cls_correct_proto"] / max(1, tp)
        cls_none_frac = a["cls_none_proto"] / max(1, a["raw_total_proto"])
        bo_fp = a["cls_incorrect_boardonly"] / max(1, a["raw_total_boardonly"])
        print(f"{m:>9} {n:>7} {l1:>7.4f} {rec:>7.3f} "
              f"{raw_acc:>8.3f} {cls_acc:>8.3f} {cls_none_frac:>8.3f} {bo_fp:>13.3f}")
        rows.append({
            "milestone": m, "n_samples": n, "l1": l1, "recall10": rec,
            "raw_top1_accuracy_proto": raw_acc,
            "classified_accuracy_proto": cls_acc,
            "classified_none_fraction": cls_none_frac,
            "n_proto_total": a["raw_total_proto"],
            "n_boardonly_total": a["raw_total_boardonly"],
            "boardonly_false_positive_rate": bo_fp,
        })

    # IN vs OUT split (generalization vs memorization) — собрано в основном цикле
    print()
    print("IN-vs-OUT split (proto archetypes only; generalization vs memorization):")
    print(f"  {'milestone':>9} {'side':>9} {'n':>7} {'L1':>7} {'rec@10':>7} {'raw_top1':>9}")
    split_report = {}
    for m in args.milestones:
        a = agg[m]
        split_report[m] = {}
        for tag, key in (("IN", "in"), ("OUT", "out")):
            n = a[f"{key}_n"]
            if n == 0:
                continue
            l1 = a[f"{key}_l1_sum"] / n
            rec = a[f"{key}_rec_sum"] / n
            acc = a[f"{key}_raw_correct"] / max(1, a[f"{key}_raw_total"])
            print(f"  {m:>9} {tag:>9} {n:>7} {l1:>7.4f} {rec:>7.3f} {acc:>9.3f}")
            split_report[m][tag] = {"n": n, "l1": l1, "recall10": rec, "raw_top1": acc}

    # Confusion matrices
    print()
    for m in args.confusion_milestones:
        cm = confusion[m]
        if not cm:
            continue
        print(f"Confusion matrix (raw argmax cosine), milestone={m}:")
        all_pred = sorted({p for c in cm.values() for p in c})
        true_order = [n for n in TRUE_LINES if n in cm] + sorted(
            n for n in cm if n not in TRUE_LINES)
        hdr = "  true\\pred".ljust(20) + " ".join(p[:10].ljust(10) for p in all_pred)
        print(hdr)
        for t in true_order:
            row = cm[t]
            cells = " ".join(str(row.get(p, 0)).ljust(10) for p in all_pred)
            print(f"  {t[:18].ljust(20)}{cells}")
        print()

    # Save
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": str(args.model),
        "prototypes": str(args.prototypes),
        "n_eval_decks": len(eval_decks),
        "in_corpus": in_count, "out_corpus": out_count,
        "archetype_counts": dict(arch_counts),
        "samples_per_deck_per_milestone": args.samples,
        "milestones": list(args.milestones),
        "rows": rows,
        "in_out_split_by_milestone": split_report,
        "confusion": {str(m): {t: dict(c) for t, c in cm.items()} for m, cm in confusion.items()},
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()