# Experiment 7 - Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders

**Subject:** CS3807 - Deep Learning Laboratory  
**Semester:** V (AY 2026-27)

## Contents of this folder

| File | What it is |
| --- | --- |
| `experiment_7.ipynb` | Jupyter notebook with all code, outputs and plots. |
| `experiment_7_report.pdf` | Compiled LaTeX report for submission. |
| `README.md` | This file - how to run the notebook. |

## What the experiment does

Studies image representation, reconstruction, denoising and generative modeling
on MNIST with four related models:

1. **Fully Connected Autoencoder** (784-128-32-16, sigmoid output, BCE loss).
2. **Convolutional Autoencoder** (Conv/MaxPool encoder, UpSampling decoder).
3. **Denoising Convolutional Autoencoder** (trained on Gaussian-noisy inputs with clean targets).
4. **Variational Autoencoder** (latent dimension 2, reparameterization trick, BCE plus KL).

It also covers reconstruction metrics (MSE, MAE, SSIM), a latent-dimension study
on the FC-AE (`d_z` in {2, 8, 16, 32}), noise-level sweeps for Gaussian
(`sigma` in {0.1, 0.2, 0.3}) and salt-and-pepper (`p` in {0.05, 0.10, 0.20})
noise, VAE latent-space visualization, image generation, latent interpolation,
and a reconstruction-error analysis.

Protocol: 10,000 MNIST training images (9,000 fit / 1,000 validation) and 2,000
test images, pixels scaled to [0, 1], seed 42, Adam 1e-3, batch 128, 20 epochs.

## Dataset

**MNIST** handwritten digits, loaded with `tf.keras.datasets.mnist`. The full set
has 60,000 training and 10,000 test images of shape 28x28x1. This experiment uses
the recommended subset: 10,000 train and 2,000 test. Keras downloads the ~11 MB
archive on first use and caches it at `~/.keras/datasets/`.

## Dependencies

```bash
pip install tensorflow scikit-image matplotlib seaborn numpy jupyter
```

Tested with:
- Python 3.12
- TensorFlow 2.21 (CPU build)
- scikit-image 0.26
- numpy 2.x
- matplotlib 3.x, seaborn 0.13

## How to run

### Option A - Jupyter

```bash
cd "experiment 7"
jupyter notebook experiment_7.ipynb
```

Then `Kernel -> Restart & Run All`.

### Option B - Headless via standalone script

```bash
cd "Experiment - 7"
python experiment_7.py
```

This reproduces every `exp7_plot_*.png`, `experiment_7_results.json` and
`experiment_7_metrics.txt` from scratch. Wall time is about 3 to 4 minutes on a
modern laptop CPU for 20 epochs across all models.

For a fast smoke test, set `EXP7_QUICK=1` to shrink the run to 1,000 train /
500 test images and 2 epochs.

### Option C - Just view the executed notebook or PDF

Open `experiment_7.ipynb` in Jupyter / VS Code / Colab, or open the report PDF.
The PDF embeds all 15 figures.

## Expected runtime breakdown

| Step | Time (CPU) |
| --- | --- |
| MNIST download (first run only) | 5 to 30 s |
| FC-AE (20 epochs) | ~6 s |
| CAE (20 epochs) | ~50 s |
| Denoising CAE (20 epochs) | ~49 s |
| VAE (20 epochs) | ~40 s |
| Latent-dimension study (4 models x 20 epochs) | ~25 s |
| Metrics, prediction and plotting | ~10 s |
| **Total (cached dataset)** | **~3 to 4 min** |

## How to (re)compile the LaTeX report

The source is `Experiment - 7/Experiment_7.tex` with all `exp7_plot_*.png`
alongside.

```bash
brew install tectonic
cd "Experiment - 7"
tectonic Experiment_7.tex
open Experiment_7.pdf
```

Or Overleaf: create a blank project, upload `Experiment_7.tex` as `main.tex`
plus all PNGs, and recompile.

## Reproducibility

- Seeds: `numpy`, `tensorflow` and `tf.keras.utils` set to 42.
- Image noise is drawn from `numpy.random.default_rng(42)`.
- The same train/val/test split is used by all four models.
- The test set is kept out of training and hyperparameter choices.
- Run-to-run variance of about +/-0.5% on the metrics is possible because
  TensorFlow is not fully deterministic on CPU.

## Results snapshot (from `experiment_7_results.json`)

Test subset, 2,000 images, seed 42:

- **FC Autoencoder:** MSE 0.02114, MAE 0.05626, SSIM 0.7471, 211,040 params, 6.1 s.
- **Conv Autoencoder:** MSE 0.00257, MAE 0.01503, SSIM 0.9742, 74,497 params, 49.9 s.
- **Denoising CAE (Gaussian sigma 0.2):** MSE 0.00485, MAE 0.02194, SSIM 0.9427, 49.0 s.
- **VAE (latent 2):** MSE 0.04785, MAE 0.11142, SSIM 0.4376, 134,165 params, 40.0 s.
  Reconstruction loss 157.91, KL loss 135.96, total 293.87.
- **Gaussian denoising SSIM:** 0.9571 (0.1), 0.9427 (0.2), 0.8770 (0.3).
- **Salt-and-pepper denoising SSIM:** 0.9302 (0.05), 0.8826 (0.10), 0.7558 (0.20).
- **Latent-dimension study (FC-AE):** MSE 0.04759 (2), 0.03347 (8), 0.02335 (16), 0.02141 (32).

## What the report contains

The PDF follows the lab manual structure (Sections 1 to 31) with filled tables,
the 10 required plots plus supporting figures, per-plot inferences, the
discussion answers and references. Source at
`https://github.com/TheVijayVignesh/DeepLearning-Lab`.
