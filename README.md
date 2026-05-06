# Quantum Vision Transformers for High-Resolution Medical Image Classification under Limited Training Data

Reproduction code for the paper *Quantum Vision Transformers for High-Resolution Medical Image Classification under Limited Training Data*. The pipeline produces every figure and table in paper.

## Acknowledgements and citation

The `quixer/` package is adapted from the [Quixer reference implementation](https://github.com/Quantinuum/Quixer) by Quantinuum, distributed under the Apache License 2.0 (see [`LICENSE`](LICENSE)). Find the original Quixer paper under:

> Khatri, N. and Matos, G. *Quixer: A Quantum Transformer Model.* [arXiv:2406.04305](https://arxiv.org/abs/2406.04305), 2024.

## Layout

```
quixer/quixer_model.py     QViT model
data_loading.py            PneumoniaMNIST, BreastMNIST, Br35H loaders
training.py                Training engine (timm baselines + Quixer)
run_lr_sweep.py            Run the LR sweep                  (paper § 3.3)
plot_lr_sweep.py           Figure 2 + Table 2                (paper § 4.1)
run_size_experiment.py     Run training + per-tuple test eval (paper § 3.4)
plot_validation_curves.py  Figures 3 / 4 / 5                 (paper § 4.2)
plot_test_results.py       Figure 6 + Tables 3 / 4           (paper § 4.3)
```

## Setup

The codebase was developed and verified with the toolchain described in paper § 3.5: PyTorch 2.3.0+cu121, TorchQuantum 0.1.8, timm 1.0.25, MedMNIST, Python 3.11, CUDA-capable GPU (RTX 4090).

Install dependencies from the included `pyproject.toml`:

```bash
pip install .
# or, with uv:
uv sync
```

### Datasets

- **PneumoniaMNIST and BreastMNIST**: downloaded automatically by `medmnist` into `--data-root`. The 224×224 NPZ files are fetched on first run.
- **Br35H**: not auto-downloaded. Get it from Kaggle ([Brain Tumor Detection](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection?resource=download)) and unzip; the archive contains an `archive/` directory with `yes/`, `no/`, `pred/`, and `Br35H-Mask-RCNN/` subfolders. Only `yes/` and `no/` are used. Point `--data-root` at the `archive/` directory (or any directory containing `yes/*.jpg` and `no/*.jpg`). The script does its own stratified 80/10/10 train/val/test split with `random_state=42`.

## Reproducing the paper

### § 3.3 — Learning-rate sweep (Figure 2, Table 2)

Sweeps {QViT, ResNet-18, Swin-T, DeiT-Ti} × {1e-3, 1e-4, 1e-5} × 5 seeds, 500 epochs each, on PneumoniaMNIST at N=100.

```bash
python run_lr_sweep.py
python plot_lr_sweep.py --in-dir lr_sweep_results
# → lr_sweep_results/lr_sweep_validation_curves.pdf   (Fig 2)
# → lr_sweep_results/lr_sweep_table.tex               (Table 2)
```

### § 3.4 — Stratified small-sample training and test evaluation

Trains all four models across the paper's training-set sizes for the chosen dataset (5 seeds, 250 epochs each), then evaluates the test set on the best-val-AUC and best-val-BalAcc checkpoints (paper § 3.4 selection rule). Run once per dataset:

```bash
python run_size_experiment.py --dataset pneumoniamnist
python run_size_experiment.py --dataset breastmnist
python run_size_experiment.py --dataset br35h --data-root /path/to/br35h
```

Each invocation writes a new timestamped subdirectory `size_experiment_results/<dataset>_<TS>/` containing:

- `curves/curves_<model>_size<N>_seed<S>.csv` — per-epoch train/val metrics
- `checkpoints/...` — best-val-AUC and best-val-BalAcc states
- `test_results.csv` — one row per (model, size, seed, selector)

The committed reference outputs in this repo live under un-timestamped names (`size_experiment_results/{pneumoniamnist,breastmnist,br35h}/`); fresh runs go into timestamped siblings so they never overwrite the reference. Pass either path to the plot scripts below.

### § 4.2 — Validation learning curves (Figures 3, 4, 5)

One invocation per dataset run directory:

```bash
python plot_validation_curves.py --dir size_experiment_results/pneumoniamnist
# → <run-dir>/validation_curves.pdf
```

### § 4.3 — Test-set comparison (Figure 6, Tables 3 and 4)

One combined invocation across all three dataset run directories. The order of `--dirs` controls the column order in Figure 6:

```bash
python plot_test_results.py --dirs \
    size_experiment_results/pneumoniamnist \
    size_experiment_results/breastmnist \
    size_experiment_results/br35h
# → ./test_results.pdf
# → ./test_auc_table.tex
# → ./test_balacc_table.tex
```

Pass `--out-dir <path>` to write the figure and tables elsewhere.

## Configuration

Per-run hyperparameters are set as module-level constants in each runner (`MODELS`, `SEEDS`, `EPOCHS`, `LR_BY_MODEL`, `QUIXER_CONFIG`, `SIZES_BY_DATASET`) and match the paper. Edit those constants for ablations; the runners only expose `--dataset`, `--out-dir`, and `--data-root` on the CLI.
