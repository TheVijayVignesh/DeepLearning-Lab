# Experiment 1 — Perceptron on Banknote Authentication

**Subject:** CS3807 — Deep Learning Laboratory
**Semester:** V (AY 2026–27)

## Contents of this folder

| File | What it is |
| --- | --- |
| `experiment_1.ipynb` | The Jupyter notebook with all code, outputs, and plots. |
| `experiment_1_report.pdf` | The compiled LaTeX report for submission. |
| `README.md` | This file — how to run the notebook. |

## What the experiment does

Implements a single-layer Perceptron from scratch on the UCI Banknote Authentication
dataset, then compares it against scikit-learn's `Perceptron`. Covers dataset
exploration, EDA, preprocessing, training, evaluation, and the mandatory plots.

## Dataset

**Banknote Authentication** from the UCI Machine Learning Repository.
The notebook downloads it directly from
`https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data.txt`
on first run. An internet connection is required the first time only — after that
the CSV is in memory.

Features: 4 (variance, skewness, curtosis, entropy of wavelet-transformed image)
Target: binary (0 = genuine, 1 = forged)

## Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Tested with:
- Python 3.11
- numpy ≥ 1.24
- pandas ≥ 2.0
- scikit-learn ≥ 1.3
- matplotlib ≥ 3.7
- seaborn ≥ 0.13

## How to run

### Option A — Jupyter

```bash
cd "experiment 1"
jupyter notebook experiment_1.ipynb
```

Then in the browser, `Kernel → Restart & Run All`.

### Option B — Headless (regenerate outputs from scratch)

```bash
cd "experiment 1"
jupyter nbconvert --to notebook --execute experiment_1.ipynb \
  --output experiment_1.ipynb --ExecutePreprocessor.timeout=600
```

This will take a few minutes, then re-saves the notebook with all cells executed
and outputs populated.

### Option C — Just view the executed notebook

Open `experiment_1.ipynb` in Jupyter / VS Code / Colab. The notebook was last
executed and all outputs are saved inline — no need to re-run unless you want
to verify reproducibility.

## Expected runtime

- Perceptron training: a few seconds.
- Total wall time for full re-run: under 2 minutes on a modern laptop.

## Reproducibility

- All random seeds are set inside the notebook (where applicable).
- The dataset is pulled from a fixed UCI URL.
- Python and library versions do not affect correctness of the
  Perceptron-from-scratch cells; scikit-learn's `Perceptron` is deterministic
  up to data shuffling.
