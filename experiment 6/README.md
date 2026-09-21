# Experiment 6 - RNN, LSTM and GRU for Sequence Learning and Video Understanding

**Subject:** CS3807 - Deep Learning Laboratory  
**Semester:** V (AY 2026-27)

## Contents of this folder

| File | What it is |
| --- | --- |
| `experiment_6.ipynb` | Jupyter notebook with all code, outputs and plots. |
| `experiment_6_report.pdf` | Compiled LaTeX report for submission. |
| `README.md` | This file - how to run the notebook. |

## What the experiment does

Studies recurrent sequence learning with the UCI Human Activity Recognition
dataset and extends it to video understanding.

1. **HAR sequence classification** with a plain RNN, an LSTM and a GRU, all using
   the same input, optimizer, batch size and epochs.
2. **Backpropagation through time** numerical exercise (hand calculation checked
   against the code).
3. **Sequence length study** with lengths 32, 64 and 128.
4. **Video understanding** with frozen MobileNetV2 frame features followed by an
   LSTM or a GRU, on five UCF101 action classes.
5. **Sequence to sequence** encoder-decoder LSTM on a synthetic digit-reversal
   task.

Input is the raw inertial signals arranged as (windows, 128 time steps, 9
channels). Protocol: 500 windows per class (3,000 total), 85/15 train/validation
split, full 2,947-window test set held out, seed 42, Adam 0.001, batch 32, 30
epochs, 32 recurrent units.

## Datasets

- **UCI HAR** raw inertial signals, downloaded from the UCI archive into
  `data/UCI HAR Dataset/` (train 7,352 windows, test 2,947 windows).
- **UCF101 subset** (five of the ten classes: Archery, BabyCrawling,
  BalanceBeam, BandMarching, Basketball) from the TensorFlow video tutorial
  subset, extracted into `data/ucf101/UCF101_subset/`.

Both are already downloaded in this folder. To fetch them again, see
`experiment_6.py` (the UCI link is in the report; the UCF101 subset is on
Hugging Face under `sayakpaul/ucf101-subset`).

## Dependencies

```bash
pip install tensorflow scikit-learn matplotlib seaborn pandas imageio imageio-ffmpeg nbformat jupyter
```

Tested with:
- Python 3.12
- TensorFlow 2.21 (CPU build)
- scikit-learn 1.9
- imageio 2.38 with the bundled ffmpeg
- numpy 2.x, matplotlib 3.x, seaborn 0.13

## How to run

### Option A - Jupyter

```bash
cd "experiment 6"
jupyter notebook experiment_6.ipynb
```

Then `Kernel -> Restart & Run All`.

### Option B - Headless via standalone script

```bash
cd "Experiment - 6"
python experiment_6.py
```

This reproduces every `exp6_plot_*.png`, `experiment_6_results.json` and
`experiment_6_metrics.txt`. Wall time is about 6 to 8 minutes on a modern laptop
CPU, mostly from the 30-epoch recurrent runs.

For a fast smoke test, set `EXP6_QUICK=1` to shrink to 60 windows per class and
2 epochs.

## Expected runtime breakdown

| Step | Time (CPU) |
| --- | --- |
| HAR data load and preprocessing | few seconds |
| RNN / LSTM / GRU (30 epochs each) | ~97 s |
| Sequence length study (3 lengths x 3 models) | ~3 to 4 min |
| Video frame extraction and CNN features (205 videos) | ~1 min |
| Video LSTM and GRU | ~9 s |
| Sequence-to-sequence | ~20 s |
| **Total** | **~6 to 8 min** |

## How to (re)compile the LaTeX report

```bash
brew install tectonic
cd "Experiment - 6"
tectonic Experiment_6.tex
open Experiment_6.pdf
```

## Reproducibility

- Seeds: `numpy`, `tensorflow` and `tf.keras.utils` set to 42.
- The balanced training subset is drawn with `numpy.random.default_rng(42)`.
- The test set is kept out of all training and model-selection steps.
- MobileNetV2 ImageNet weights are frozen and used only for feature extraction.
- Run-to-run variance of about +/-1% is possible because TensorFlow is not fully
  deterministic on CPU.

## Results snapshot (from `experiment_6_results.json`)

HAR test set (2,947 windows):

- **RNN:** accuracy 0.7940, macro F1 0.7875, 1,974 params, 16.7 s.
- **LSTM:** accuracy 0.8853, macro F1 0.8845, 6,006 params, 36.0 s.
- **GRU:** accuracy 0.9002, macro F1 0.8991, 4,758 params, 44.0 s.

Sequence length (macro F1): RNN 0.803/0.817/0.736, LSTM 0.863/0.882/0.893, GRU
0.887/0.883/0.909 for lengths 32/64/128.

Video (5 classes, 40 test clips): LSTM accuracy 0.950, F1 0.940, 168,677 params;
GRU accuracy 0.950, F1 0.954, 126,757 params.

Sequence-to-sequence reversal: token accuracy 0.9315, sequence accuracy 0.6180,
52,874 params.

## What the report contains

The PDF follows the lab manual structure (Sections 1 to 30) with the filled
tables, all the required plots (sensor signals, loss, accuracy, confusion
matrices, model comparison, sequence length, video frames, video curves, video
confusion, seq2seq training), the discussion answers and references. Source at
`https://github.com/TheVijayVignesh/DeepLearning-Lab`.
