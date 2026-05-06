#!/usr/bin/env python3
"""
Plot validation learning curves for Figures 3, 4, 5 (paper § 4.2).

Reads per-run training-curve CSVs from ``<run-dir>/curves/`` and produces a
grid with one row per training-set size in {N_full, 100, 50} and one column
per metric in {val_auc, val_bal_acc}. Solid lines are the seed-mean, shaded
bands are ±1 standard deviation across seeds. A visual ellipsis is inserted
between the N_full row and the smaller-N rows.

Usage:
    python plot_validation_curves.py --dir size_experiment_results/<dataset>_<TS>
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER: List[str] = ["quixer", "resnet18", "swin", "deit"]
MODEL_LABELS: Dict[str, str] = {
    "quixer": "QViT",
    "resnet18": "ResNet",
    "swin": "Swin",
    "deit": "DeiT",
}
MODEL_COLORS: Dict[str, str] = {
    "quixer": "#ef476f",
    "resnet18": "#118ab2",
    "swin": "#06d6a0",
    "deit": "#ffd166",
}

METRICS: List[Tuple[str, str]] = [
    ("val_auc", "Validation AUC"),
    ("val_bal_acc", "Validation BalAcc"),
]

SMALL_N_ROWS: List[int] = [100, 50]

FONT_SCALE: float = 2.0
TITLE_FONTSIZE: int = int(10 * FONT_SCALE)
LABEL_FONTSIZE: int = int(10 * FONT_SCALE)
LEGEND_FONTSIZE: int = int(10 * FONT_SCALE)
TICK_FONTSIZE: int = int(8 * FONT_SCALE)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        required=True,
        type=Path,
        help="Run directory containing a 'curves/' subdirectory.",
    )
    return parser.parse_args()


def load_curves(run_dir: Path) -> pd.DataFrame:
    curves_dir = run_dir / "curves"
    csv_paths = sorted(curves_dir.glob("curves_*.csv"))
    if not csv_paths:
        raise SystemExit(f"No curves_*.csv files found in {curves_dir}")
    return pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)


def select_rows(sample_sizes: List[int]) -> Tuple[List[int], List[str]]:
    full = max(sample_sizes)
    sizes = [full] + [s for s in SMALL_N_ROWS if s in sample_sizes]
    labels = ["full"] + [str(s) for s in sizes[1:]]
    return sizes, labels


def plot(df: pd.DataFrame, run_dir: Path) -> Path:
    sample_sizes = sorted(df["sample_size"].unique())
    sizes, labels = select_rows(sample_sizes)

    aggregations = {metric: ["mean", "std"] for metric, _ in METRICS}
    grouped = df.groupby(["model", "sample_size", "epoch"]).agg(aggregations)

    n_data_rows = len(sizes)
    n_cols = len(METRICS)

    height_ratios = [1.0, 0.15] + [1.0] * (n_data_rows - 1)
    fig_height = 2.8 * (n_data_rows + 0.15)
    fig, all_axes = plt.subplots(
        n_data_rows + 1,
        n_cols,
        figsize=(5.5 * n_cols, fig_height),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.35},
    )
    if n_cols == 1:
        all_axes = all_axes.reshape(-1, 1)

    for ax in all_axes[1]:
        ax.axis("off")

    data_axes = np.array([all_axes[0]] + [all_axes[i] for i in range(2, n_data_rows + 1)])
    if data_axes.ndim == 1:
        data_axes = data_axes.reshape(1, -1)

    for row_idx, (size, label) in enumerate(zip(sizes, labels)):
        for col_idx, (metric, title) in enumerate(METRICS):
            ax = data_axes[row_idx, col_idx]
            for model in MODEL_ORDER:
                try:
                    sub = grouped.loc[(model, size)]
                except KeyError:
                    continue
                epochs = sub.index.get_level_values("epoch").values
                mean = sub[metric]["mean"].values
                std = sub[metric]["std"].values
                color = MODEL_COLORS[model]
                ax.plot(epochs, mean, color=color, linewidth=1.6, label=MODEL_LABELS[model])
                ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.18)

            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
            if row_idx == 0:
                ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
            if row_idx == n_data_rows - 1:
                ax.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
            if col_idx == 0:
                ax.set_ylabel(f"N={label}", fontsize=LABEL_FONTSIZE)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.9, ncol=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()

    fig.canvas.draw()
    spacer_pos = all_axes[1, 0].get_position()
    mid_y = (spacer_pos.y0 + spacer_pos.y1) / 2
    fig.text(0.5, mid_y, "•   •   •", ha="center", va="center", fontsize=20)

    out = run_dir / "validation_curves.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    df = load_curves(args.dir)
    out_path = plot(df, args.dir)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
