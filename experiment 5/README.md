# Experiment 5 — Comprehensive Study of CNN Training with MobileNetV2 on Oxford-IIIT Pet

**Subject:** CS3807 — Deep Learning Laboratory  
**Semester:** V (AY 2026–27)

## Contents of this folder

| File | What it is |
| --- | --- |
| `experiment_5.ipynb` | Jupyter notebook with all code, outputs and plots (single-line comments only). |
| `experiment_5_report.pdf` | Compiled LaTeX report for submission (from `Experiment - 5/Experiment_5_complete.tex`). |
| `experiment_5.py` | Stand-alone Python script that reproduces every figure and the JSON results. |
| `experiment_5_results.json` | Numerical results (initialization, optimizer, CV, final metrics). |
| `experiment_5_classification_report.txt` | Scikit-learn classification report for the final model on the 3669 test images. |
| `exp5_plot_*.png` | The 15 mandatory plots plus sample and distribution figures. |
| `README.md` | This file — how to run the notebook. |

## What the experiment does

Comprehensive study of CNN training and optimization using the **Oxford-IIIT Pet dataset** (37 cat/dog breeds) with images resized to **224×224×3** and **MobileNetV2** pretrained on ImageNet.

Covers: weight initialization (Zero, Random, Xavier/Glorot, He), regularization and overfitting, Batch Normalization (numerical example), Dropout, optimization algorithms (SGD, Momentum, RMSProp, Adam), CNN hyperparameters (learning rate, batch size, dropout, kernel/stride/padding concepts, depthwise separable convolution), transfer learning (feature extraction) vs fine-tuning (unfreezing last 20 layers with small LR), 5-fold cross-validation for model selection, and final evaluation on an independent test set.

Precomputation: MobileNetV2 pooled features (1280-D) are extracted once (3128 train / 552 val / 3669 test) so that initialization/optimizer/regularization studies train only the Dense(128) head — fast and isolates classifier behavior. Full-image MobileNetV2 is used for the frozen vs fine-tuned comparison. CV uses the pooled features (8 epochs per fold). Final model is the best CV configuration (C3: RMSProp lr 0.001 dropout 0.25) retrained on 3680 train images and tested on 3669.

## Dataset

**Oxford-IIIT Pet** — 37 breeds, 3680 training + 3669 test images (TFDS `oxford_iiit_pet` 4.0.0, split train/test). All images are RGB, variable size, resized to 224×224 and normalized to [-1,1] for MobileNetV2. Stratified 85/15 split of the 3680 train images gives 3128 train + 552 validation; test (3669) is untouched until final evaluation.

First run downloads the dataset via `tensorflow_datasets` to `~/tensorflow_datasets/oxford_iiit_pet/4.0.0` (~773 MB download, 5–6 min). Subsequent runs reuse the cached TFRecords. ImageNet MobileNetV2 weights (~14 MB) are also cached after first use.

## Dependencies

```bash
pip install tensorflow tensorflow-datasets scikit-learn matplotlib seaborn numpy jupyter
```

Tested with:
- Python 3.11
- TensorFlow 2.21 (CPU build; Apple Silicon `tensorflow-macos` works)
- tensorflow-datasets 4.9.10
- scikit-learn 1.9
- numpy 2.4
- matplotlib 3.11, seaborn 0.13

## How to run

### Option A — Jupyter (interactive)

```bash
cd "experiment 5"
jupyter notebook experiment_5.ipynb
```

Then `Kernel → Restart & Run All`. The first cells load Pet via TFDS (downloads if needed) and extract MobileNetV2 features (~40 s for 3128+552+3669 images on CPU). Head studies are seconds each; frozen training 10 epochs ~230 s, fine-tuning 5 epochs ~140 s, CV ~30 s.

### Option B — Headless via standalone script (regenerates plots from scratch)

```bash
cd "experiment 5"
python experiment_5.py
```

This reproduces every `exp5_plot_*.png` and `experiment_5_results.json` without needing the notebook. Wall time ~8–10 min on a modern laptop CPU (feature extraction dominates).

### Option C — Headless via notebook

```bash
cd "experiment 5"
jupyter nbconvert --to notebook --execute experiment_5.ipynb \
  --output experiment_5.ipynb --ExecutePreprocessor.timeout=1800
```

### Option D — Just view the executed notebook

Open `experiment_5.ipynb` in Jupyter / VS Code / Colab. The PDF report already embeds all 15 plots.

## Expected runtime breakdown

| Step | Time (CPU, 4 threads) |
| --- | --- |
| TFDS download (first run only) | 5–6 min (773 MB) |
| MobileNetV2 feature extraction (7349 images) | ~40 s |
| Weight init (4 × 10 epochs head) | ~8 s |
| Regularization (4 × 10 epochs) | ~8 s |
| Optimizers (4 × 10 epochs) | ~8 s |
| Hyperparams LR/BS/Dropout (7 × 10 epochs) | ~15 s |
| Frozen MobileNetV2 (10 epochs) | ~230 s |
| Fine-tuning (5 epochs, last 20 layers) | ~140 s |
| 5-fold CV (4 configs × 5 folds × 8 epochs) | ~30 s |
| Final retrain + confusion + misclassified | ~5 s |
| **Total (cached dataset)** | **~8–10 min** |
| **Total (first download)** | **~14–16 min** |

## How to (re)compile the LaTeX report

The source is `Experiment - 5/Experiment_5_complete.tex` with all `exp5_plot_*.png` alongside.

```bash
brew install tectonic
cd "Experiment - 5"
tectonic Experiment_5_complete.tex
open Experiment_5_complete.pdf
# copy to repo
cp Experiment_5_complete.pdf "../DL-Repo/experiment 5/experiment_5_report.pdf"
```

Or Overleaf: create blank project, upload `Experiment_5_complete.tex` as `main.tex` plus all PNGs, recompile.

## Reproducibility

- Seeds: `numpy` and `tensorflow` set to 42; `KFold` and `train_test_split` use `random_state=42`.
- Stratified split (85/15) preserves breed ratios.
- Feature extractor is `MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))` → GlobalAveragePooling → 1280-D.
- Head: `Dense(128, ReLU) → Dropout → Dense(37, Softmax)`; initializers as listed; optimizer comparison uses head with dropout 0.25.
- Fine-tuning: last 20 MobileNetV2 layers unfrozen, BatchNormalization kept frozen, Adam lr 1e-5, 5 epochs.
- CV: 5-fold on pooled features, 8 epochs per fold, RMSProp vs Adam etc.
- Final head: C3 (RMSProp lr 0.001 dropout 0.25) retrained 12 epochs on 3680 train, tested on 3669.

Run-to-run variance ±0.5% on test accuracy due to TF non-determinism on CPU.

## Results snapshot (from `experiment_5_results.json`)

- **Best initialization:** He (92.39% val) > Glorot (92.03%) > Random (91.85%) > Zero (2.72% failure)
- **Best regularization:** Dropout 0.5 (92.75%) vs BN 92.03% vs No reg 90.94% vs L2 88.22%
- **Best optimizer:** RMSProp 92.21% / Adam 91.49% / Momentum 90.76% / SGD 57.61%
- **Best LR:** 0.001 (92.39%) > 0.0001 (89.67%); batch 16 ≈ 32 ≈ 64; dropout 0.25 best (92.57%)
- **Transfer:** frozen val 92.75% test 89.42%; fine-tuned val 93.30% test 89.32%
- **5-fold CV (means):** C3 RMSProp 91.66±1.28%, C1 Adam 91.25±1.05%, C4 Dropout0.5 91.39±0.76%, C2 low-LR 89.54±1.81% → C3 selected
- **Final test:** 88.80% accuracy, 89.18% precision, 88.72% recall, 88.65% F1 (macro), 2.43M params

All 15 mandatory plots are in this folder and embedded in the PDF (Sections 5–12).

## What the report contains

The PDF follows the lab manual structure (Sections 1–18) with filled tables, the 15 plots, inference per plot, discussion answers, and references. Source at `https://github.com/TheVijayVignesh/DeepLearning-Lab`.

