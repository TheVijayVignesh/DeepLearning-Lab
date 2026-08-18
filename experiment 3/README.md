# Experiment 3 - CNN on CIFAR-10

**Subject:** CS3807 - Deep Learning Laboratory  
**Semester:** V (AY 2026-27)

## Contents of this folder

| File | What it is |
| --- | --- |
| `experiment_3.ipynb` | The Jupyter notebook containing the experiment code. |
| `experiment_3_report.pdf` | The completed laboratory report. |
| `README.md` | This file - how to run the notebook. |

## What the experiment does

Builds and evaluates a convolutional neural network on CIFAR-10. The experiment
covers convolution output dimensions, kernel sizes, stride, padding, feature-map
visualization, max pooling, average pooling and classification metrics.

The main CNN uses two convolution layers, ReLU activations, two pooling layers,
a 128-unit dense layer and a 10-class softmax output. Max pooling and average
pooling are trained with the same settings so their results can be compared.

## Dataset

**CIFAR-10** contains 50,000 training images and 10,000 test images. Each image
is a `32 x 32 x 3` RGB image belonging to one of ten balanced classes.

The notebook loads the standard CIFAR-10 archive from the Keras cache. If the
archive is not cached, TensorFlow downloads it on the first run.

## Dependencies

```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy jupyter
```

The experiment was run with Python 3.11, TensorFlow 2.21, scikit-learn 1.9,
NumPy 2.4, matplotlib 3.11 and seaborn 0.13.

## How to run

### Option A - Jupyter

```bash
cd "experiment 3"
jupyter notebook experiment_3.ipynb
```

Run all cells after opening the notebook. The full run trains both pooling
variants for 20 epochs and regenerates the figures.

### Option B - Headless

```bash
cd "experiment 3"
jupyter nbconvert --to notebook --execute experiment_3.ipynb \
  --output experiment_3.ipynb --ExecutePreprocessor.timeout=1800
```

The complete CPU run takes approximately 10 minutes on the machine used for
the report. A GPU will reduce this time.

## Reproducibility

- NumPy and TensorFlow seeds are set to 42.
- The CIFAR-10 archive is fixed by the official dataset source.
- The report was generated from the same CNN script included in the notebook.
- CPU training can show small run-to-run differences because TensorFlow is not
  fully deterministic for every operation.
