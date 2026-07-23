# Experiment 2 — MLP on Fashion-MNIST

**Subject:** CS3807 — Deep Learning Laboratory
**Semester:** V (AY 2026–27)

## Contents of this folder

| File | What it is |
| --- | --- |
| `experiment_2.ipynb` | The Jupyter notebook with all code, outputs, and plots. |
| `experiment_2_report.pdf` | The compiled LaTeX report for submission. |
| `README.md` | This file — how to run the notebook and re-generate the report. |

## What the experiment does

Builds a fully-connected Multi-Layer Perceptron (MLP) on the Fashion-MNIST dataset
with TensorFlow/Keras, trains a hand-picked baseline, then runs a randomized
hyperparameter search (RandomizedSearchCV + SciKeras) over 8 hyperparameters,
retrains the best-found model, and compares baseline vs optimized.

## Dataset

**Fashion-MNIST** — 60,000 training + 10,000 test 28×28 grayscale clothing images,
10 balanced classes. Loaded automatically by `tensorflow.keras.datasets.fashion_mnist`
and cached at `~/.keras/datasets/` on first use.

## Dependencies

```bash
pip install tensorflow scikit-learn scikeras matplotlib seaborn pandas numpy jupyter
```

Tested with:
- Python 3.11
- TensorFlow 2.15+ (CPU build is fine — no GPU required)
- scikit-learn 1.3+
- scikeras 0.13+
- matplotlib, seaborn, pandas, numpy

If you are on Apple Silicon, `pip install tensorflow` already pulls in the
`tensorflow-macos` build; no extra steps needed.

## How to run the notebook

### Option A — Jupyter (interactive)

```bash
cd "experiment 2"
jupyter notebook experiment_2.ipynb
```

Then in the browser, `Kernel → Restart & Run All`.

### Option B — Headless (regenerate outputs)

```bash
cd "experiment 2"
jupyter nbconvert --to notebook --execute experiment_2.ipynb \
  --output experiment_2.ipynb --ExecutePreprocessor.timeout=1800
```

The first cell downloads Fashion-MNIST (~30 MB) which takes a few seconds; later
runs use the cached copy. The full search + retrain + plots take **about 3–4
minutes on a modern laptop CPU**.

### Option C — Just view the executed notebook

Open `experiment_2.ipynb` in Jupyter / VS Code / Colab. The notebook was last
executed and all outputs (training curves, confusion matrices, search results)
are saved inline — no need to re-run unless you want to verify reproducibility.

## How to (re)compile the LaTeX report

The source for `experiment_2_report.pdf` is `Experiment_2_complete.tex`, kept in
the parent `Experiment - 2/` folder alongside the 8 PNG plot files
(`plot_01_*.png` through `plot_09_*.png`).

To recompile locally:

```bash
# Either install tectonic (small) or full MacTeX
brew install tectonic

cd "Experiment - 2"
tectonic Experiment_2_complete.tex
open Experiment_2_complete.pdf
```

Or use Overleaf (no install):
1. Create a new blank project at overleaf.com.
2. Copy-paste the contents of `Experiment_2_complete.tex` into `main.tex`.
3. Upload all 8 PNG files into the project.
4. Recompile — the PDF appears in the right pane.

## Expected runtime breakdown

| Step | Time |
| --- | --- |
| Data download (first run only) | 5–30 s |
| Baseline training (20 epochs) | ~40 s |
| Randomized search (10 trials × 3 folds on 6k subset) | ~2 min |
| Best-model retraining (30 epochs on full 60k) | ~30 s |
| Plotting | < 5 s |
| **Total** | **~3–4 min** |

## Reproducibility

- `numpy` and `tensorflow` seeds are set to 42 in the first cell.
- `RandomizedSearchCV` uses `random_state=42`.
- Search subset is drawn with `np.random.default_rng(42)`.
- Run-to-run variance on the final test accuracy is **±0.3 %** due to TF
  non-determinism on CPU.

## What the report contains

The 9 mandatory plots are all saved next to the .tex and embedded in the PDF:

1. Sample images (10 from the training set)
2. Class distribution
3–6. Baseline training/validation accuracy & loss (20 epochs)
7. Baseline confusion matrix
8. Hyperparameter search results (top 10 trials)
9. Best-model accuracy comparison (baseline vs optimized bar chart)

Plus the optimized model: confusion matrix, training curves, and the actual
numbers from the run are in the Results and Discussion sections.
