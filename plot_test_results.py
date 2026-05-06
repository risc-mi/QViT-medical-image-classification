#!/usr/bin/env python3
"""
Plot Figure 6 and emit the test-set LaTeX tables (paper § 4.3).

Reads ``test_results.csv`` from each provided run directory and writes:

  <out-dir>/test_results.pdf    -- Figure 6 (test AUC and BalAcc)
  <out-dir>/test_auc_table.tex
  <out-dir>/test_balacc_table.tex

Per paper § 3.4, test AUC is taken from rows with selector="auc" and test
balanced accuracy from rows with selector="bal_acc" (i.e. evaluated at the
checkpoints with the highest validation AUC and BalAcc respectively).

Datasets appear in the order in which their run directories are passed.

Usage:
    python plot_test_results.py --dirs <r_pneumoniamnist> <r_breastmnist> <r_br35h>
"""
from __future__ import annotations

import argparse
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

DATASET_TITLES: Dict[str, str] = {
    "pneumoniamnist": "PneumoniaMNIST",
    "breastmnist": "BreastMNIST",
    "br35h": "Br35H",
}

PLOT_METRICS: List[Tuple[str, str, str]] = [
    ("auc", "test_auc", "Test AUC"),
    ("bal_acc", "test_bal_acc", "Test BalAcc"),
]

TABLE_METRICS: List[Tuple[str, str, str, str, str]] = [
    ("auc", "test_auc", "Test AUC", "test_auc_table.tex", "tab:test_auc"),
    ("bal_acc", "test_bal_acc", "Test Balanced Accuracy", "test_balacc_table.tex", "tab:test_balacc"),
]

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
        "--dirs",
        nargs="+",
        required=True,
        type=Path,
        help="Run directories, each containing test_results.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory to write the figure and LaTeX tables.",
    )
    return parser.parse_args()


def load_test_results(run_dirs: List[Path]) -> List[Tuple[str, pd.DataFrame]]:
    loaded: List[Tuple[str, pd.DataFrame]] = []
    for run_dir in run_dirs:
        csv_path = run_dir / "test_results.csv"
        if not csv_path.exists():
            print(f"WARNING: missing {csv_path}; skipping.")
            continue
        df = pd.read_csv(csv_path)
        datasets = df["dataset"].unique()
        if len(datasets) != 1:
            raise SystemExit(f"{csv_path} contains multiple datasets: {list(datasets)}")
        loaded.append((str(datasets[0]), df))
    if not loaded:
        raise SystemExit("No usable test_results.csv files found.")
    return loaded


def title_for(dataset: str) -> str:
    return DATASET_TITLES.get(dataset, dataset)


def plot_figure(loaded: List[Tuple[str, pd.DataFrame]], out_path: Path) -> None:
    n_cols = len(loaded)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.6 * n_cols, 7.0), sharey="row")
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col_idx, (dataset, df) in enumerate(loaded):
        for row_idx, (selector, value_col, ylabel) in enumerate(PLOT_METRICS):
            ax = axes[row_idx, col_idx]
            df_sel = df[df["selector"] == selector]
            sample_sizes = sorted(df_sel["sample_size"].unique())
            if not sample_sizes:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            base_positions = np.arange(len(sample_sizes))
            n_models = len(MODEL_ORDER)
            box_width = 0.18
            offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * box_width

            for m_idx, model in enumerate(MODEL_ORDER):
                df_model = df_sel[df_sel["model"] == model]
                if df_model.empty:
                    continue
                values_per_size = []
                positions = []
                for s_idx, sample_size in enumerate(sample_sizes):
                    vals = df_model[df_model["sample_size"] == sample_size][value_col].values
                    if len(vals) == 0:
                        continue
                    values_per_size.append(vals)
                    positions.append(base_positions[s_idx] + offsets[m_idx])
                if not values_per_size:
                    continue

                color = MODEL_COLORS[model]
                ax.boxplot(
                    values_per_size,
                    positions=positions,
                    widths=box_width * 0.9,
                    patch_artist=True,
                    showfliers=False,
                    boxprops={"facecolor": color, "edgecolor": color, "linewidth": 1.2, "alpha": 0.85},
                    whiskerprops={"color": color, "linewidth": 1.0},
                    capprops={"color": color, "linewidth": 1.0},
                    medianprops={"color": "#111111", "linewidth": 1.2},
                )

            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xticks(base_positions)
            ax.set_xticklabels([str(s) for s in sample_sizes])
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
            ax.set_xlabel("Sample Size (N)", fontsize=LABEL_FONTSIZE)
            if row_idx == 0:
                ax.set_title(title_for(dataset), fontsize=TITLE_FONTSIZE, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
            if row_idx == 0 and col_idx == n_cols // 2:
                handles = [
                    plt.Line2D(
                        [0], [0],
                        color=MODEL_COLORS[m],
                        marker="s",
                        linestyle="",
                        markersize=8,
                        label=MODEL_LABELS[m],
                    )
                    for m in MODEL_ORDER
                ]
                ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE, framealpha=0.9, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def aggregate_mean_std(df: pd.DataFrame, selector: str, value_col: str) -> Dict[str, Dict[int, Tuple[float, float]]]:
    df_sel = df[df["selector"] == selector]
    result: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for model in MODEL_ORDER:
        df_model = df_sel[df_sel["model"] == model]
        stats: Dict[int, Tuple[float, float]] = {}
        for ss in sorted(df_model["sample_size"].unique()):
            vals = df_model[df_model["sample_size"] == ss][value_col].values
            stats[int(ss)] = (float(vals.mean()), float(vals.std(ddof=0)))
        result[model] = stats
    return result


def best_per_size(stats: Dict[str, Dict[int, Tuple[float, float]]], sizes: List[int]) -> Dict[int, str]:
    best: Dict[int, str] = {}
    for ss in sizes:
        best_val = None
        best_model = None
        for model in MODEL_ORDER:
            if ss not in stats.get(model, {}):
                continue
            mean, _ = stats[model][ss]
            if best_val is None or mean > best_val:
                best_val = mean
                best_model = model
        if best_model is not None:
            best[ss] = best_model
    return best


def format_cell(mean: float, std: float, bold: bool) -> str:
    if bold:
        return f"\\textbf{{{mean:.4f}}} $\\pm$ {std:.4f}"
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def render_table(
    loaded: List[Tuple[str, pd.DataFrame]],
    selector: str,
    value_col: str,
    metric_title: str,
    label: str,
) -> str:
    per_dataset_sizes: Dict[str, List[int]] = {}
    per_dataset_stats: Dict[str, Dict[str, Dict[int, Tuple[float, float]]]] = {}
    for dataset, df in loaded:
        stats = aggregate_mean_std(df, selector, value_col)
        sizes = sorted({ss for d in stats.values() for ss in d.keys()})
        per_dataset_sizes[dataset] = sizes
        per_dataset_stats[dataset] = stats

    max_cols = max(len(v) for v in per_dataset_sizes.values()) if per_dataset_sizes else 0
    col_spec = "l" + "c" * max_cols

    lines: List[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{metric_title} (mean $\\pm$ std) across datasets and sample sizes.}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    first = True
    for dataset, _ in loaded:
        sizes = per_dataset_sizes[dataset]
        stats = per_dataset_stats[dataset]
        best = best_per_size(stats, sizes)

        if not first:
            lines.append("\\midrule")
        first = False

        header = [f"\\textbf{{{title_for(dataset)}}}"] + [f"$N = {s}$" for s in sizes]
        header += [""] * (max_cols - len(sizes))
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\midrule")

        for model in MODEL_ORDER:
            row = [MODEL_LABELS[model]]
            for ss in sizes:
                if ss in stats.get(model, {}):
                    mean, std = stats[model][ss]
                    row.append(format_cell(mean, std, bold=best.get(ss) == model))
                else:
                    row.append("---")
            row += [""] * (max_cols - len(sizes))
            lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_test_results(args.dirs)

    pdf_path = args.out_dir / "test_results.pdf"
    plot_figure(loaded, pdf_path)
    print(f"Wrote {pdf_path}")

    for selector, value_col, metric_title, filename, label in TABLE_METRICS:
        tex = render_table(loaded, selector, value_col, metric_title, label)
        out_path = args.out_dir / filename
        out_path.write_text(tex + "\n")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
