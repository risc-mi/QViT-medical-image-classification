#!/usr/bin/env python3
"""
Plot Figure 2 and emit Table 2 from learning-rate sweep curves (paper § 3.3).

Reads per-run validation curves named ``curves_<model>_<lr>_<seed>.csv`` from
``--in-dir`` and writes:

  <in-dir>/lr_sweep_validation_curves.pdf   -- Figure 2
  <in-dir>/lr_sweep_table.tex               -- Table 2 (LaTeX snippet)

The selection rule per (model, lr) is the highest epoch-wise mean validation
AUC across seeds (paper § 3.3). The plot's shaded band is the population
standard deviation across seeds at each epoch; the table's ± term is the
sample standard deviation (n-1) at the selected epoch.

Usage:
    python plot_lr_sweep.py [--in-dir lr_sweep_results]
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

MODELS: List[str] = ["quixer", "resnet18", "swin", "deit"]
LRS: List[float] = [1e-3, 1e-4, 1e-5]
LR_LABELS: Dict[float, str] = {1e-3: "1e-3", 1e-4: "1e-4", 1e-5: "1e-5"}
LR_COLORS: Dict[float, str] = {1e-3: "#ef476f", 1e-4: "#118ab2", 1e-5: "#06d6a0"}
MODEL_TITLES: Dict[str, str] = {
    "quixer": "QViT",
    "resnet18": "ResNet",
    "swin": "Swin Transformer",
    "deit": "DeiT",
}
TABLE_NAMES: Dict[str, str] = {
    "quixer": "QViT",
    "resnet18": "ResNet",
    "swin": "Swin",
    "deit": "DeiT",
}

FONT_SCALE: float = 2.0
LABEL_FONTSIZE: int = int(11 * FONT_SCALE)
TITLE_FONTSIZE: int = int(13 * FONT_SCALE)
LEGEND_FONTSIZE: int = int(10 * FONT_SCALE)
TICK_FONTSIZE: int = int(10 * FONT_SCALE * 1.2)

CURVES_FILENAME_RE = re.compile(r"^curves_(?P<model>[a-z0-9]+)_(?P<lr>[0-9.e+-]+)_(?P<seed>\d+)\.csv$")

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
        "--in-dir",
        type=Path,
        default=Path("./lr_sweep_results"),
        help="Directory containing curves_<model>_<lr>_<seed>.csv files.",
    )
    return parser.parse_args()


def read_val_aucs(csv_path: Path) -> List[float]:
    aucs: List[float] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                aucs.append(float(row["val_auc"]))
            except (KeyError, ValueError):
                aucs.append(float("nan"))
    return aucs


def collect_curves(in_dir: Path) -> Dict[Tuple[str, float], List[np.ndarray]]:
    grouped: Dict[Tuple[str, float], List[np.ndarray]] = defaultdict(list)
    for csv_path in sorted(in_dir.glob("curves_*.csv")):
        match = CURVES_FILENAME_RE.match(csv_path.name)
        if match is None:
            continue
        model = match["model"]
        try:
            lr = float(match["lr"])
        except ValueError:
            continue
        if model not in MODELS or lr not in LRS:
            continue
        grouped[(model, lr)].append(np.asarray(read_val_aucs(csv_path), dtype=float))
    return grouped


def stack_padded(curves: List[np.ndarray]) -> np.ndarray:
    max_len = max(len(c) for c in curves)
    padded = [np.pad(c, (0, max_len - len(c)), constant_values=np.nan) for c in curves]
    return np.vstack(padded)


def best_mean_auc(stacked: np.ndarray) -> Tuple[float, float, int]:
    mean_per_epoch = np.nanmean(stacked, axis=0)
    if np.all(np.isnan(mean_per_epoch)):
        return float("nan"), float("nan"), -1
    best_epoch = int(np.nanargmax(mean_per_epoch))
    best_mean = float(mean_per_epoch[best_epoch])
    vals = stacked[:, best_epoch]
    finite = vals[np.isfinite(vals)]
    if finite.size < 2:
        sample_std = 0.0
    else:
        sample_std = float(np.std(finite, ddof=1))
    return best_mean, sample_std, best_epoch


def plot_figure(
    grouped: Dict[Tuple[str, float], List[np.ndarray]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        for lr in LRS:
            curves = grouped.get((model, lr))
            if not curves:
                continue
            stacked = stack_padded(curves)
            mean = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)
            epochs = np.arange(stacked.shape[1])
            color = LR_COLORS[lr]
            ax.plot(epochs, mean, label=f"LR={LR_LABELS[lr]}", color=color, linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
        if idx == 0:
            ax.set_ylabel("Validation AUC", fontsize=LABEL_FONTSIZE)
        ax.set_title(MODEL_TITLES[model], fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.0)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_table(
    grouped: Dict[Tuple[str, float], List[np.ndarray]],
    digits: int = 4,
) -> str:
    best: Dict[Tuple[str, float], Tuple[float, float]] = {}
    for model in MODELS:
        for lr in LRS:
            curves = grouped.get((model, lr))
            if not curves:
                best[(model, lr)] = (float("nan"), float("nan"))
                continue
            mean, std, _ = best_mean_auc(stack_padded(curves))
            best[(model, lr)] = (mean, std)

    best_per_model = {
        model: max(best[(model, lr)][0] for lr in LRS if not math.isnan(best[(model, lr)][0]))
        for model in MODELS
    }

    lines: List[str] = []
    lines.append("\\hline")
    lr_header = " & ".join(LR_LABELS[lr] for lr in LRS)
    lines.append(f"Learning rate & {lr_header} \\\\")
    lines.append("\\hline")
    for model in MODELS:
        cells: List[str] = []
        for lr in LRS:
            mean, std = best[(model, lr)]
            if math.isnan(mean):
                cells.append("nan")
                continue
            mean_str = f"{mean:.{digits}f}"
            std_str = f"{std:.{digits}f}"
            if math.isclose(mean, best_per_model[model], rel_tol=1e-12, abs_tol=1e-12):
                cells.append(f"$\\textbf{{{mean_str}}} \\pm {std_str}$")
            else:
                cells.append(f"${mean_str} \\pm {std_str}$")
        lines.append(f"{TABLE_NAMES[model]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\hline")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.in_dir.exists():
        raise SystemExit(f"Input directory not found: {args.in_dir}")

    grouped = collect_curves(args.in_dir)
    if not grouped:
        raise SystemExit(f"No curves_*.csv files matching the expected schema found in {args.in_dir}")

    pdf_path = args.in_dir / "lr_sweep_validation_curves.pdf"
    tex_path = args.in_dir / "lr_sweep_table.tex"

    plot_figure(grouped, pdf_path)
    tex_path.write_text(render_table(grouped) + "\n")

    print(f"Wrote {pdf_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
