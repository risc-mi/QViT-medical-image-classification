#!/usr/bin/env python3
"""
Learning-rate sweep on PneumoniaMNIST at N=100 (paper § 3.3).

Sweeps {QViT, ResNet-18, Swin-T, DeiT-Ti} × {1e-3, 1e-4, 1e-5} × 5 seeds for
500 epochs each. Writes one curves CSV per training run; aggregation, the
selection rule (best mean validation AUC across seeds), Figure 2 and Table 2
are produced by ``plot_lr_sweep.py``.

Usage:
    python run_lr_sweep.py [--out-dir lr_sweep_results] [--data-root ./data]
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import List

import torch

from training import train

MODELS: List[str] = ["quixer", "resnet18", "swin", "deit"]
LRS: List[float] = [1e-3, 1e-4, 1e-5]
SEEDS: List[int] = [43, 44, 45, 46, 47]

EPOCHS: int = 500
BATCH_SIZE: int = 32
WEIGHT_DECAY: float = 0.01
MAX_GRAD_NORM: float = 5.0
TRAIN_SAMPLES: int = 100
DATASET: str = "pneumoniamnist"

QUIXER_CONFIG = dict(
    quixer_qubits=6,
    quixer_patch_size=7,
    quixer_degree=3,
    quixer_ansatz_layers=6,
    quixer_dimension=128,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./lr_sweep_results"),
        help="Directory for per-run curves CSVs.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for the PneumoniaMNIST data files.",
    )
    return parser.parse_args()


def run_one(
    *,
    model: str,
    lr: float,
    seed: int,
    out_dir: Path,
    data_root: str,
) -> None:
    curves_csv = out_dir / f"curves_{model}_{lr:.0e}_{seed}.csv"
    train(
        model=model,
        seed=seed,
        device="cuda",
        pretrained="no",
        normalize="medmnist",
        lr=lr,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        train_samples=TRAIN_SAMPLES,
        num_workers=2,
        amp=False,
        max_grad_norm=MAX_GRAD_NORM,
        data_root=data_root,
        save=str(out_dir / "_tmp_best.pt"),
        curves_csv=str(curves_csv),
        run_number=seed,
        dataset=DATASET,
        **QUIXER_CONFIG,
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = len(MODELS) * len(LRS) * len(SEEDS)
    i = 0
    for model in MODELS:
        for lr in LRS:
            for seed in SEEDS:
                i += 1
                print(f"[{i}/{total}] model={model} lr={lr:.0e} seed={seed}")
                run_one(
                    model=model,
                    lr=lr,
                    seed=seed,
                    out_dir=args.out_dir,
                    data_root=args.data_root,
                )
                gc.collect()
                torch.cuda.empty_cache()

    tmp_ckpt = args.out_dir / "_tmp_best.pt"
    if tmp_ckpt.exists():
        tmp_ckpt.unlink()


if __name__ == "__main__":
    main()
