#!/usr/bin/env python3
"""
train_card_embedding.py — Supervised behavior-cloning trainer for the
card-embedding IL model (kiyotah's official MyModel encoder/decoder, reused
verbatim via scripts/cardemb_common.py — see that file's docstring).

This REPLACES kiyotah's self-play/MCTS training loop with plain supervised
learning on precomputed (encoder_sparse, decoder_sparse, expert_label) triples
built by scripts/build_card_embedding_data.py. No game engine (libcg) is
imported or required here — this script only needs torch + the precomputed
.pt tensors, so it runs identically locally (CPU, for a quick sanity check)
and on a Kaggle GPU kernel (just bundle cardemb_common.py + train.pt/val.pt
as a private dataset and point --data-dir at it).

Loss:
  - Primary: cross-entropy over the decoder policy head's per-option scores
    (MyModel's policy head is tanh-bounded to [-1,1]; used directly as
    bounded "logits" for softmax/cross-entropy — this does not hurt argmax
    accuracy, only caps achievable confidence/calibration; see report).
  - Optional auxiliary: Huber loss pulling the value head toward +1.0 (every
    training sample is from the WINNING side by construction, so this is a
    constant regression target — a fairly weak signal, included because the
    task spec asks for it "optionally"; default weight is small).

Metric: top-1 accuracy = argmax(policy_logits) == expert's chosen option
index, evaluated on a held-out val split (split by EPISODE upstream, see
build_il_meta_dataset.py / build_card_embedding_data.py — no leakage from
decisions of the same game appearing in both train and val).

Usage (local CPU sanity — small subset, 1-2 epochs):
  .venv/bin/python scripts/train_card_embedding.py --limit-train 500 --limit-val 200 --epochs 2 --device cpu

Usage (Kaggle GPU, full run):
  python train_card_embedding.py --data-dir /kaggle/input/<dataset-slug> --out-dir /kaggle/working/out --epochs 20 --device cuda
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

# Locate cardemb_common.py: on Kaggle it ships inside the input dataset (mounted
# read-only under /kaggle/input/<slug>/), not next to this script. Locally it
# lives right next to this file (scripts/cardemb_common.py). Whichever
# directory we find it in also holds train.pt/val.pt/meta.json, so it doubles
# as the default --data-dir.
_kaggle_hits = glob.glob("/kaggle/input/**/cardemb_common.py", recursive=True)
if _kaggle_hits:
    # Kaggle: our dataset packaging puts cardemb_common.py + train.pt/val.pt/
    # meta.json all in the same flat directory (see
    # scratchpad/gold/kaggle_cardemb_dataset/) -> reuse it as --data-dir too.
    DATASET_DIR = Path(_kaggle_hits[0]).parent
    DEFAULT_DATA_DIR = DATASET_DIR
    DEFAULT_OUT_DIR = Path("/kaggle/working/out")
else:
    # Local repo layout: code lives in scripts/, data in data/il/cardemb/.
    DATASET_DIR = HERE
    DEFAULT_DATA_DIR = HERE.parent / "data" / "il" / "cardemb"
    DEFAULT_OUT_DIR = HERE.parent / "out" / "cardemb"
sys.path.insert(0, str(DATASET_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import cardemb_common as cc  # noqa: E402


def load_samples(path: Path) -> list[dict]:
    samples = torch.load(path, weights_only=False)
    return samples


def build_encoder_batch(batch: list[dict], device):
    li = cc.LearnInput()
    for s in batch:
        li.add(s["enc_index"], s["enc_value"], s["enc_offset"])
    index = torch.tensor(li.index, dtype=torch.int32, device=device)
    value = torch.tensor(li.value, dtype=torch.float32, device=device)
    offset = torch.tensor(li.offset, dtype=torch.int32, device=device)
    return index, value, offset


def build_decoder_batch(batch: list[dict], device):
    """Pad every sample's decoder word count to this batch's max n_options.
    Padding words are empty EmbeddingBag bags (offset with no index/value
    entries -> zero vector), matching kiyotah's own padding trick."""
    pad_to = max(s["n_options"] for s in batch)

    index: list[int] = []
    value: list[float] = []
    offset: list[int] = []
    labels: list[int] = []
    mask: list[list[int]] = []

    for s in batch:
        n = s["n_options"]
        base = len(index)
        index.extend(s["dec_index"])
        value.extend(s["dec_value"])
        for o in s["dec_offset"]:
            offset.append(o + base)
        for _ in range(pad_to - n):
            offset.append(len(index))  # empty bag
        labels.append(s["label"])
        mask.append([1] * n + [0] * (pad_to - n))

    index_t = torch.tensor(index, dtype=torch.int32, device=device)
    value_t = torch.tensor(value, dtype=torch.float32, device=device)
    offset_t = torch.tensor(offset, dtype=torch.int32, device=device)
    label_t = torch.tensor(labels, dtype=torch.long, device=device)
    mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
    return index_t, value_t, offset_t, label_t, mask_t, pad_to


def run_epoch(model, samples, batch_size, device, optimizer=None, value_weight=0.05, shuffle=True):
    """One pass over `samples`. If optimizer is not None, trains; else eval-only."""
    is_train = optimizer is not None
    model.train(is_train)

    order = list(range(len(samples)))
    if shuffle:
        random.shuffle(order)

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_correct = 0
    total_n = 0

    n_batches = (len(order) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        idxs = order[bi * batch_size : (bi + 1) * batch_size]
        batch = [samples[i] for i in idxs]
        if not batch:
            continue

        enc_index, enc_value, enc_offset = build_encoder_batch(batch, device)
        dec_index, dec_value, dec_offset, label_t, mask_t, pad_to = build_decoder_batch(batch, device)

        ctx = torch.enable_grad() if is_train else torch.inference_mode()
        with ctx:
            value_out, policy_out = model(enc_index, enc_value, enc_offset, dec_index, dec_value, dec_offset)
            policy_masked = policy_out.masked_fill(~mask_t, -1e9)
            loss_policy = F.cross_entropy(policy_masked, label_t)

            # Value head target: per-sample backward-discounted game outcome
            # (built by scripts/build_card_embedding_data.py; field
            # "value_target"). Falls back to 1.0 for old datasets that don't
            # carry the field — matching the pre-fix constant-1.0 behavior so
            # this stays load-compatible with already-built .pt files.
            # value_out is shape (batch, 1); build a matching (batch, 1) tensor.
            target_v = torch.tensor(
                [[float(s.get("value_target", 1.0))] for s in batch],
                dtype=value_out.dtype,
                device=device,
            ).view_as(value_out)
            loss_value = F.huber_loss(value_out, target_v, delta=0.2)

            loss = loss_policy + value_weight * loss_value

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = len(batch)
        total_loss += loss.item() * bs
        total_policy_loss += loss_policy.item() * bs
        total_value_loss += loss_value.item() * bs
        with torch.no_grad():
            pred = policy_masked.argmax(dim=1)
            total_correct += (pred == label_t).sum().item()
        total_n += bs

    return {
        "loss": total_loss / max(total_n, 1),
        "policy_loss": total_policy_loss / max(total_n, 1),
        "value_loss": total_value_loss / max(total_n, 1),
        "accuracy": total_correct / max(total_n, 1),
        "n": total_n,
    }


def random_baseline_accuracy(samples: list[dict]) -> float:
    if not samples:
        return 0.0
    return sum(1.0 / s["n_options"] for s in samples) / len(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--enc-layers", type=int, default=1)
    ap.add_argument("--dec-layers", type=int, default=1)
    ap.add_argument("--value-weight", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW weight decay (regularization; v1 run used 0.0)")
    ap.add_argument("--dropout", type=float, default=0.1, help="Transformer encoder/decoder layer dropout (v1 run used 0.0)")
    ap.add_argument("--emb-dropout", type=float, default=0.1, help="Dropout on raw EmbeddingBag outputs (v1 run used 0.0)")
    ap.add_argument("--limit-train", type=int, default=None, help="Subsample train set (quick sanity runs)")
    ap.add_argument("--limit-val", type=int, default=None, help="Subsample val set (quick sanity runs)")
    ap.add_argument("--device", type=str, default=None, help="cuda|cpu; default: cuda if available else cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Some Kaggle GPU pool machines (e.g. Tesla P100, sm_60) are older than
        # the minimum CUDA compute capability the installed torch wheel was
        # built for (sm_70+) -> every kernel launch fails with
        # "CUDA error: no kernel image is available for execution on the
        # device". Detect that up front with a throwaway op and fall back to
        # CPU instead of crashing the whole run.
        try:
            _t = torch.nn.EmbeddingBag(4, 4, mode="sum").to(device)
            _t(torch.tensor([0], dtype=torch.long, device=device), torch.tensor([0], dtype=torch.long, device=device))
            del _t
        except RuntimeError as e:
            print(f"WARNING: CUDA device present but unusable ({e}); falling back to CPU.")
            device = torch.device("cpu")
    print(f"device: {device}")

    train_samples = load_samples(args.data_dir / "train.pt")
    val_samples = load_samples(args.data_dir / "val.pt")
    print(f"loaded {len(train_samples)} train / {len(val_samples)} val samples from {args.data_dir}")

    if args.limit_train:
        random.Random(args.seed).shuffle(train_samples)
        train_samples = train_samples[: args.limit_train]
    if args.limit_val:
        random.Random(args.seed).shuffle(val_samples)
        val_samples = val_samples[: args.limit_val]

    train_baseline = random_baseline_accuracy(train_samples)
    val_baseline = random_baseline_accuracy(val_samples)
    print(f"random-choice baseline accuracy: train={train_baseline:.4f}  val={val_baseline:.4f}")

    model = cc.MyModel(
        args.d_model, args.num_heads, args.d_ff, args.enc_layers, args.dec_layers,
        dropout=args.dropout, emb_dropout=args.emb_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model params: {n_params:,}  (encoder_size={cc.encoder_size} decoder_size={cc.decoder_size} "
        f"d_model={args.d_model} dropout={args.dropout} emb_dropout={args.emb_dropout})"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"optimizer: AdamW lr={args.lr} weight_decay={args.weight_decay}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_acc = -1.0

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_samples, args.batch_size, device, optimizer=optimizer, value_weight=args.value_weight)
        val_stats = run_epoch(model, val_samples, args.batch_size, device, optimizer=None, value_weight=args.value_weight, shuffle=False)

        elapsed = time.time() - t0
        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train loss={train_stats['loss']:.4f} acc={train_stats['accuracy']:.4f}  "
            f"val loss={val_stats['loss']:.4f} acc={val_stats['accuracy']:.4f}  "
            f"({elapsed:.1f}s elapsed)"
        )
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})

        if val_stats["accuracy"] > best_val_acc:
            best_val_acc = val_stats["accuracy"]
            torch.save(model.state_dict(), args.out_dir / "model_best.pth")

    torch.save(model.state_dict(), args.out_dir / "model_final.pth")
    torch.save(model.encoder_bag.weight.detach().cpu(), args.out_dir / "card_embeddings.pt")

    # ---- AUC GATE: value-head ROC-AUC vs linear-probe on held-out replays ----
    # Runs scripts/value_probe.compute_aucs with the JUST-TRAINED model's value
    # head. Fails the gate (printed PASS/FAIL) if value_head_auc < bar. Does NOT
    # crash training if replays aren't available locally (e.g. on a Kaggle GPU
    # kernel with no data/episodes/); logs a warning instead.
    value_head_auc = None
    linear_probe_auc = None
    auc_gate = {"available": False}
    AUC_BAR = 0.67
    try:
        sys.path.insert(0, str(HERE))
        import value_probe as vp  # noqa: E402
        # Deck for the encoder input: prefer v024's Crustle deck.csv (the model
        # is trained on many winner-side decks, so any fixed 60-card list works
        # for a deck-invariant value probe); fall back to any submission deck.
        deck = None
        for cand in [
            HERE.parent / "submissions" / "v024_cardemb_rl" / "deck.csv",
        ]:
            if cand.exists():
                deck = [int(x) for x in cand.read_text().splitlines() if x.strip()][:60]
                break
        if deck is None:
            import glob as _glob  # noqa: E402
            for p in sorted(_glob.glob(str(HERE.parent / "submissions" / "*" / "deck.csv"))):
                deck = [int(x) for x in Path(p).read_text().splitlines() if x.strip()][:60]
                break
        # The gate must evaluate the checkpoint we actually ship, i.e.
        # model_best.pth (saved at argmax val accuracy), NOT the model's
        # end-of-training state (model_final) which may have regressed. Reload
        # the saved best state_dict into a fresh CPU-eval instance.
        best_path = args.out_dir / "model_best.pth"
        if best_path.exists():
            probe_model = cc.MyModel(
                args.d_model, args.num_heads, args.d_ff, args.enc_layers, args.dec_layers,
                dropout=args.dropout, emb_dropout=args.emb_dropout,
            ).to("cpu")
            probe_model.load_state_dict(
                torch.load(best_path, map_location="cpu", weights_only=True)
            )
            probe_model.eval()
            print(f"[auc-gate] evaluating saved model_best.pth (not model_final)")
        else:
            probe_model = model.cpu().eval()
        auc_gate = vp.compute_aucs(
            model=probe_model, deck=deck,
            replay_paths=None, max_games=600, seed=0, verbose=True,
        )
        value_head_auc = auc_gate.get("value_head_auc")
        linear_probe_auc = auc_gate.get("linear_probe_auc")
    except Exception as e:  # noqa: BLE001 — gate must never crash training
        print(f"WARNING: AUC gate could not run ({type(e).__name__}: {e}); skipping.")
        auc_gate = {"available": False, "error": f"{type(e).__name__}: {e}"}

    if auc_gate.get("available") and value_head_auc is not None:
        verdict = "PASS" if value_head_auc >= AUC_BAR else "FAIL"
        print(f"\n=== AUC GATE (bar={AUC_BAR}) ===")
        print(f"  value_head_auc   = {value_head_auc:.4f}")
        print(f"  linear_probe_auc = {linear_probe_auc:.4f}")
        print(f"  -> {verdict} (value_head_auc {'>=' if verdict=='PASS' else '<'} {AUC_BAR})")
    elif auc_gate.get("available"):
        print(f"\n=== AUC GATE (bar={AUC_BAR}) ===")
        print(f"  value_head_auc = N/A (not enough evaluated states); linear_probe_auc={linear_probe_auc}")
        print("  -> SKIP (insufficient value-head evals)")
    else:
        print(f"\n=== AUC GATE (bar={AUC_BAR}) ===")
        print("  -> SKIP (no held-out replays available locally; re-run probe on a machine with data/episodes/)")

    metrics = {
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "train_random_baseline_acc": train_baseline,
        "val_random_baseline_acc": val_baseline,
        "best_val_accuracy": best_val_acc,
        "final_val_accuracy": history[-1]["val"]["accuracy"] if history else None,
        "final_train_accuracy": history[-1]["train"]["accuracy"] if history else None,
        "value_head_auc": value_head_auc,
        "linear_probe_auc": linear_probe_auc,
        "value_head_auc_bar": AUC_BAR,
        "auc_gate": auc_gate,
        "history": history,
        "args": vars(args) | {"data_dir": str(args.data_dir), "out_dir": str(args.out_dir), "device": str(device)},
    }
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print("\n=== DONE ===")
    print(f"  best val accuracy: {best_val_acc:.4f}  (random baseline: {val_baseline:.4f})")
    print(f"  wrote {args.out_dir}/model_best.pth, model_final.pth, card_embeddings.pt, metrics.json")


if __name__ == "__main__":
    main()
