#!/usr/bin/env python3
"""
Stratified small-sample training and test evaluation (paper § 3.4).

Trains {QViT, ResNet-18, Swin-T, DeiT-Ti} on the given dataset across the
paper's training-set sizes, 5 seeds per setting, 250 epochs each. After each
training run, evaluates the test set on the best-val-AUC and best-val-BalAcc
checkpoints (paper § 3.4) and appends a row per checkpoint to test_results.csv.

Output is written under a timestamped subdirectory of ``--out-dir``:

    <out-dir>/<dataset>_<TS>/curves/curves_<model>_size<N>_seed<S>.csv
    <out-dir>/<dataset>_<TS>/checkpoints/<model>_size<N>_seed<S>_best_<sel>_epoch<E>.pt
    <out-dir>/<dataset>_<TS>/test_results.csv

Usage:
    python run_size_experiment.py --dataset pneumoniamnist
    python run_size_experiment.py --dataset breastmnist
    python run_size_experiment.py --dataset br35h --data-root /path/to/br35h
"""
from __future__ import annotations

import argparse
import csv
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from data_loading import load_dataset
from training import create_model, evaluate, train

MODELS: List[str] = ["quixer", "resnet18", "swin", "deit"]
SEEDS: List[int] = [43, 44, 45, 46, 47]

EPOCHS: int = 250
BATCH_SIZE: int = 32
WEIGHT_DECAY: float = 0.01
MAX_GRAD_NORM: float = 5.0
IMAGE_SIZE: int = 224

LR_BY_MODEL: Dict[str, float] = {
    "quixer": 1e-3,
    "resnet18": 1e-4,
    "deit": 1e-4,
    "swin": 1e-5,
}

QUIXER_CONFIG = dict(
    quixer_qubits=6,
    quixer_patch_size=7,
    quixer_degree=3,
    quixer_ansatz_layers=6,
    quixer_dimension=128,
)

SIZES_BY_DATASET: Dict[str, List[int]] = {
    "pneumoniamnist": [4708, 1000, 500, 100, 50],
    "br35h":          [2400, 1000, 500, 100, 50],
    "breastmnist":    [546, 100, 50],
}

SELECTORS: List[str] = ["auc", "bal_acc"]

TEST_CSV_COLUMNS: List[str] = [
    "dataset",
    "model",
    "sample_size",
    "seed",
    "selector",
    "epoch",
    "test_loss",
    "test_acc",
    "test_auc",
    "test_bal_acc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(SIZES_BY_DATASET.keys()),
        help="Dataset to train and evaluate on.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./size_experiment_results"),
        help="Parent directory for the timestamped run directory.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for the dataset files.",
    )
    return parser.parse_args()


def build_test_loader(dataset: str, data_root: str) -> DataLoader:
    _, _, test_ds = load_dataset(
        name=dataset,
        size=IMAGE_SIZE,
        normalize="medmnist",
        root=data_root,
    )
    return DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def parse_checkpoint_epoch(path: Path) -> int:
    return int(path.stem.split("_epoch")[-1])


def evaluate_run_checkpoints(
    *,
    model_key: str,
    save_prefix: str,
    ckpt_dir: Path,
    test_loader: DataLoader,
    device: torch.device,
) -> List[Dict[str, object]]:
    quixer_args = {k.removeprefix("quixer_"): v for k, v in QUIXER_CONFIG.items()}
    model_obj = create_model(
        model_key,
        pretrained=False,
        device=device,
        batch_size=BATCH_SIZE,
        **quixer_args,
    ).to(device)

    rows: List[Dict[str, object]] = []
    for selector in SELECTORS:
        matches = sorted(ckpt_dir.glob(f"{save_prefix}_best_{selector}_epoch*.pt"))
        if not matches:
            print(f"  WARNING: no {selector} checkpoint for {save_prefix}")
            continue
        checkpoint_path = matches[-1]
        state = torch.load(checkpoint_path, map_location=device)
        model_obj.load_state_dict(state, strict=False)
        loss, acc, auc, bal_acc = evaluate(model_obj, test_loader, device)
        rows.append(
            {
                "selector": selector,
                "epoch": parse_checkpoint_epoch(checkpoint_path),
                "test_loss": loss,
                "test_acc": acc,
                "test_auc": auc,
                "test_bal_acc": bal_acc,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    sizes = SIZES_BY_DATASET[args.dataset]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = args.out_dir / f"{args.dataset}_{timestamp}"
    curves_dir = run_dir / "curves"
    ckpt_dir = run_dir / "checkpoints"
    curves_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    test_csv = run_dir / "test_results.csv"
    with test_csv.open("w", newline="") as f:
        csv.writer(f).writerow(TEST_CSV_COLUMNS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = build_test_loader(args.dataset, args.data_root)

    total = len(MODELS) * len(sizes) * len(SEEDS)
    step = 0
    for size in sizes:
        for model in MODELS:
            for seed in SEEDS:
                step += 1
                save_prefix = f"{model}_size{size}_seed{seed}"
                curves_csv = curves_dir / f"curves_{save_prefix}.csv"
                print(
                    f"[{step}/{total}] dataset={args.dataset} model={model} "
                    f"size={size} seed={seed}"
                )

                train(
                    model=model,
                    seed=seed,
                    device="cuda",
                    pretrained="no",
                    normalize="medmnist",
                    lr=LR_BY_MODEL[model],
                    weight_decay=WEIGHT_DECAY,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    train_samples=size,
                    num_workers=2,
                    amp=False,
                    max_grad_norm=MAX_GRAD_NORM,
                    data_root=args.data_root,
                    save_dir=str(ckpt_dir),
                    save_prefix=save_prefix,
                    curves_csv=str(curves_csv),
                    run_number=seed,
                    dataset=args.dataset,
                    **QUIXER_CONFIG,
                )

                rows = evaluate_run_checkpoints(
                    model_key=model,
                    save_prefix=save_prefix,
                    ckpt_dir=ckpt_dir,
                    test_loader=test_loader,
                    device=device,
                )
                with test_csv.open("a", newline="") as f:
                    writer = csv.writer(f)
                    for row in rows:
                        writer.writerow(
                            [
                                args.dataset,
                                model,
                                size,
                                seed,
                                row["selector"],
                                row["epoch"],
                                row["test_loss"],
                                row["test_acc"],
                                row["test_auc"],
                                row["test_bal_acc"],
                            ]
                        )

                gc.collect()
                torch.cuda.empty_cache()

    print(f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
