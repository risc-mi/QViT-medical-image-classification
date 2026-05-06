#!/usr/bin/env python3
"""
Training engine: timm classical baselines (ResNet-18, DeiT-Tiny, Swin-Tiny)
and the Quixer QViT, on PneumoniaMNIST, BreastMNIST, or Br35H.

Exports:
    train(...)            -- run one (model, dataset, lr, seed) configuration
    create_model(...)     -- build a model from a key in {quixer, resnet18, deit, swin}
    evaluate(...)         -- batched evaluation returning (loss, acc, auc, bal_acc)

This module is the single training/evaluation entry point used by
``run_lr_sweep.py`` and ``run_size_experiment.py``. It can also be invoked as
a CLI for ad-hoc single-run experiments.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.6",
)
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data_loading import load_dataset
from quixer.quixer_model import Quixer

torch.cuda.set_per_process_memory_fraction(1.0)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def timm_model_name(model_key: str) -> str:
    model_key = model_key.lower()
    if model_key == "vit":
        return "vit_tiny_patch16_224"
    if model_key == "deit":
        return "deit_tiny_patch16_224"
    if model_key == "swin":
        return "swin_tiny_patch4_window7_224"
    if model_key == "resnet18":
        return "resnet18"
    raise ValueError(f"Unknown model: {model_key}")


def create_model(
    model_key: str,
    pretrained: bool,
    device: torch.device,
    batch_size: int,
    **quixer_args,
) -> nn.Module:
    if model_key.lower() == "quixer":
        return Quixer(
            n_qubits=quixer_args["qubits"],
            patch_size=quixer_args["patch_size"],
            qsvt_polynomial_degree=quixer_args["degree"],
            n_ansatz_layers=quixer_args["ansatz_layers"],
            num_classes=2,
            embedding_dimension=quixer_args["dimension"],
            dropout=0.1,
            batch_size=batch_size,
            device=device,
        )
    return timm.create_model(
        timm_model_name(model_key),
        pretrained=pretrained,
        num_classes=2,
        in_chans=1,
    )


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).squeeze(-1).long()

        output = model(x)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        loss = ce(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        total_correct += int((pred == y).sum().item())
        total_n += int(x.size(0))

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_scores.append(probs.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
        all_preds.append(pred.detach().cpu().numpy())

    avg_loss = total_loss / max(total_n, 1)
    acc = 100.0 * total_correct / max(total_n, 1)

    scores = np.concatenate(all_scores) if all_scores else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    preds = np.concatenate(all_preds) if all_preds else np.array([])

    try:
        auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    try:
        bal_acc = float(balanced_accuracy_score(labels, preds)) if len(np.unique(labels)) > 1 else float("nan")
    except Exception:
        bal_acc = float("nan")

    return avg_loss, acc, auc, bal_acc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    max_grad_norm: float,
) -> Tuple[float, float, float, float]:
    model.train()
    ce = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).squeeze(-1).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            output = model(x)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            loss = ce(logits, y)

        scaler.scale(loss).backward()
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        total_correct += int((pred == y).sum().item())
        total_n += int(x.size(0))

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_scores.append(probs.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
        all_preds.append(pred.detach().cpu().numpy())

    avg_loss = total_loss / max(total_n, 1)
    acc = 100.0 * total_correct / max(total_n, 1)

    scores = np.concatenate(all_scores) if all_scores else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    preds = np.concatenate(all_preds) if all_preds else np.array([])

    try:
        auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    try:
        bal_acc = float(balanced_accuracy_score(labels, preds)) if len(np.unique(labels)) > 1 else float("nan")
    except Exception:
        bal_acc = float("nan")

    return avg_loss, acc, auc, bal_acc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["vit", "deit", "swin", "resnet18", "quixer"])
    p.add_argument("--dataset", type=str, default="pneumoniamnist", choices=["pneumoniamnist", "breastmnist", "br35h"])
    p.add_argument("--pretrained", type=str, default="no", choices=["yes", "no"])
    p.add_argument("--normalize", type=str, default="medmnist", choices=["imagenet", "medmnist"])

    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--train-samples", type=int, default=None)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=5.0)

    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")

    p.add_argument("--quixer-qubits", type=int, default=6)
    p.add_argument("--quixer-patch-size", type=int, default=7)
    p.add_argument("--quixer-degree", type=int, default=3)
    p.add_argument("--quixer-ansatz-layers", type=int, default=6)
    p.add_argument("--quixer-dimension", type=int, default=128)

    p.add_argument("--save", type=str, default="./best_model.pt")
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--save-prefix", type=str, default=None)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--curves-csv", type=str, default=None)
    p.add_argument("--run-number", type=int, default=1)
    return p.parse_args()


def train(
    model: str,
    seed: int,
    device: str,
    pretrained: str,
    normalize: str,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    size: int = 224,
    train_samples: int = None,
    num_workers: int = 0,
    amp: bool = False,
    max_grad_norm: float = 1.0,
    data_root: str = "./data",
    save: str = "./best_model.pt",
    save_dir: str = None,
    save_prefix: str = None,
    curves_csv: str = None,
    run_number: int = 1,
    quixer_qubits: int = 6,
    quixer_patch_size: int = 7,
    quixer_degree: int = 3,
    quixer_ansatz_layers: int = 6,
    quixer_dimension: int = 128,
    return_metrics: bool = False,
    dataset: str = "pneumoniamnist",
) -> object:
    """
    Train one (model, dataset, lr, seed) configuration.

    When ``save_dir`` is given, two checkpoints are kept across the run:
    the best validation-AUC and best validation-BalAcc states (paper § 3.4).
    When ``curves_csv`` is given, per-epoch train/val metrics are appended.
    With ``return_metrics=True`` the function returns a dict of best-val
    metrics; otherwise it returns the best validation AUC.
    """
    seed_everything(seed)

    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    pretrained_bool = (pretrained == "yes")

    if model.lower() == "quixer":
        print("Model: Quixer")
        print(
            f"Quixer config: qubits={quixer_qubits}, patch_size={quixer_patch_size}, "
            f"degree={quixer_degree}, ansatz_layers={quixer_ansatz_layers}, "
            f"dimension={quixer_dimension}"
        )
    else:
        print(f"Model: {model} (timm={timm_model_name(model)})")
        print(f"Pretrained: {pretrained_bool}")
    print(f"Normalize: {normalize}")
    print(f"Device: {device}")
    print(f"LR: {lr} | WD: {weight_decay}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | Image size: {size}")
    if train_samples:
        print(f"Training samples: LIMITED to {train_samples}")

    print(f"Loading {dataset} dataset from: {data_root}")
    train_ds, val_ds, test_ds = load_dataset(
        name=dataset,
        size=size,
        normalize=normalize,
        root=data_root,
    )

    if train_samples is not None and train_samples < len(train_ds):
        if dataset == "br35h":
            labels = np.array([int(train_ds[i][1]) for i in range(len(train_ds))])
        else:
            labels = np.array([train_ds[i][1].item() for i in range(len(train_ds))])
        indices = np.arange(len(train_ds))

        selected_indices, _ = train_test_split(
            indices,
            train_size=train_samples,
            stratify=labels,
            random_state=seed,
        )
        train_ds = torch.utils.data.Subset(train_ds, selected_indices.tolist())

        selected_labels = labels[selected_indices]
        unique, counts = np.unique(selected_labels, return_counts=True)
        class_dist = dict(zip(unique.astype(int), counts.astype(int)))
        print(f"Using {len(train_ds)} training samples (stratified subset)")
        print(f"Class distribution: {class_dist}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    quixer_args = {
        "qubits": quixer_qubits,
        "patch_size": quixer_patch_size,
        "degree": quixer_degree,
        "ansatz_layers": quixer_ansatz_layers,
        "dimension": quixer_dimension,
    }

    model_obj = create_model(
        model,
        pretrained=pretrained_bool,
        device=device,
        batch_size=batch_size,
        **quixer_args,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model_obj.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_auc = -1.0
    best_val_acc = -1.0
    best_val_bal_acc = -1.0
    best_val_loss = float("inf")
    best_path = save

    save_dir_path = Path(save_dir) if save_dir else None
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)
        prefix = save_prefix or f"{model}_size{train_samples}_run{run_number}_seed{seed}"
        best_paths = {"auc": None, "bal_acc": None}
    else:
        best_paths = None

    curves_writer = None
    curves_file = None
    sample_size = None
    if curves_csv:
        curves_path = Path(curves_csv)
        curves_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = curves_path.exists()
        curves_file = curves_path.open("a", newline="")
        curves_writer = csv.writer(curves_file)

        if (not file_exists) or (curves_path.stat().st_size == 0):
            curves_writer.writerow(
                [
                    "model",
                    "sample_size",
                    "run_number",
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "train_auc",
                    "train_bal_acc",
                    "val_loss",
                    "val_acc",
                    "val_auc",
                    "val_bal_acc",
                ]
            )
        sample_size = len(train_ds)

    def save_checkpoint(path: Path) -> None:
        state = model_obj.state_dict()
        if model.lower() == "quixer":
            state.pop("torchquantum_device.states", None)
        torch.save(state, path)

    try:
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc, tr_auc, tr_bal_acc = train_one_epoch(
                model=model_obj,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                amp=(amp and device.type == "cuda"),
                max_grad_norm=max_grad_norm,
            )
            va_loss, va_acc, va_auc, va_bal_acc = evaluate(model_obj, val_loader, device)

            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d}/{epochs} | lr={lr_now:.2e} | "
                f"train: loss={tr_loss:.4f} acc={tr_acc:.2f}% auc={tr_auc:.4f} bal_acc={tr_bal_acc:.4f} | "
                f"val: loss={va_loss:.4f} acc={va_acc:.2f}% auc={va_auc:.4f} bal_acc={va_bal_acc:.4f}"
            )

            if curves_writer is not None:
                curves_writer.writerow(
                    [
                        model,
                        sample_size,
                        run_number,
                        epoch - 1,
                        tr_loss,
                        tr_acc,
                        tr_auc,
                        tr_bal_acc,
                        va_loss,
                        va_acc,
                        va_auc,
                        va_bal_acc,
                    ]
                )

            if best_paths is not None:
                if not np.isnan(va_loss) and va_loss < best_val_loss:
                    best_val_loss = va_loss
                if not np.isnan(va_acc) and va_acc > best_val_acc:
                    best_val_acc = va_acc
                if not np.isnan(va_auc) and va_auc > best_val_auc:
                    best_val_auc = va_auc
                    if best_paths["auc"] is not None:
                        best_paths["auc"].unlink(missing_ok=True)
                    best_paths["auc"] = save_dir_path / f"{prefix}_best_auc_epoch{epoch:03d}.pt"
                    save_checkpoint(best_paths["auc"])
                if not np.isnan(va_bal_acc) and va_bal_acc > best_val_bal_acc:
                    best_val_bal_acc = va_bal_acc
                    if best_paths["bal_acc"] is not None:
                        best_paths["bal_acc"].unlink(missing_ok=True)
                    best_paths["bal_acc"] = save_dir_path / f"{prefix}_best_bal_acc_epoch{epoch:03d}.pt"
                    save_checkpoint(best_paths["bal_acc"])
            else:
                if not np.isnan(va_auc) and va_auc > best_val_auc:
                    best_val_auc = va_auc
                    save_checkpoint(Path(best_path))

        if best_paths is not None:
            print(f"Best checkpoints saved: auc={best_paths['auc']} bal_acc={best_paths['bal_acc']}")
        else:
            print(f"Best checkpoint saved to: {best_path} (best val AUC={best_val_auc:.4f})")

        if curves_file is not None:
            curves_file.close()

        test_checkpoint = best_paths["auc"] if best_paths is not None else best_path
        model_obj.load_state_dict(torch.load(test_checkpoint, map_location=device), strict=False)
        te_loss, te_acc, te_auc, te_bal_acc = evaluate(model_obj, test_loader, device)
        print(f"TEST: loss={te_loss:.4f} acc={te_acc:.2f}% auc={te_auc:.4f} bal_acc={te_bal_acc:.4f}")

        if return_metrics:
            best_val_loss_out = best_val_loss if np.isfinite(best_val_loss) else float("nan")
            return {
                "best_val_loss": best_val_loss_out,
                "best_val_acc": best_val_acc,
                "best_val_auc": best_val_auc,
                "best_val_bal_acc": best_val_bal_acc,
            }
        return best_val_auc
    finally:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


def main() -> None:
    args = parse_args()
    train(
        model=args.model,
        seed=args.seed,
        device=args.device,
        pretrained=args.pretrained,
        normalize=args.normalize,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        size=args.size,
        train_samples=args.train_samples,
        num_workers=args.num_workers,
        amp=args.amp,
        max_grad_norm=args.max_grad_norm,
        data_root=args.data_root,
        save=args.save,
        save_dir=args.save_dir,
        save_prefix=args.save_prefix,
        curves_csv=args.curves_csv,
        run_number=args.run_number,
        quixer_qubits=args.quixer_qubits,
        quixer_patch_size=args.quixer_patch_size,
        quixer_degree=args.quixer_degree,
        quixer_ansatz_layers=args.quixer_ansatz_layers,
        quixer_dimension=args.quixer_dimension,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
