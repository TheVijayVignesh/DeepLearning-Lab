# Experiment 4 - Transfer Learning on CIFAR-10

**Subject:** CS3807 - Deep Learning Laboratory  
**Semester:** V (AY 2026-27)

## Contents of this folder

| File | What it is |
| --- | --- |
| `experiment_4.ipynb` | The Jupyter notebook containing the experiment code. |
| `experiment_4_report.pdf` | The completed laboratory report. |
| `README.md` | This file - how to run the notebook. |

## What the experiment does

Studies the development of CNN architectures from LeNet-5 to ResNet and then
implements transfer learning on CIFAR-10 using MobileNetV2 pretrained on
ImageNet.

The experiment first freezes the pretrained convolutional base and trains a new
CIFAR-10 classifier. It then unfreezes the final convolution block and fine-tunes
it with a smaller learning rate. Accuracy, precision, recall, F1-score, training
curves, a confusion matrix and misclassified examples are included in the report.

## Dataset and model

**CIFAR-10** contains 50,000 training images and 10,000 test images. Each image
is a `32 x 32 x 3` RGB image belonging to one of ten balanced classes.

MobileNetV2 uses ImageNet weights, global average pooling, a 128-unit ReLU
layer and a 10-unit softmax output. The other architectures in the report are
included as a theoretical size and design comparison. The manual permits one
pretrained model for the practical transfer-learning implementation.

## Dependencies

```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy jupyter
```

The experiment was run with Python 3.11, TensorFlow 2.21, scikit-learn 1.9,
NumPy 2.4, matplotlib 3.11 and seaborn 0.13.

## How to run

### Option A - Jupyter

```bash
cd "experiment 4"
jupyter notebook experiment_4.ipynb
```

Run all cells after opening the notebook. The first run downloads the ImageNet
MobileNetV2 weights and uses the cached CIFAR-10 archive.

### Option B - Headless

```bash
cd "experiment 4"
jupyter nbconvert --to notebook --execute experiment_4.ipynb \
  --output experiment_4.ipynb --ExecutePreprocessor.timeout=1800
```

The full CPU run takes approximately 7 minutes on the machine used for the
report. A GPU will reduce this time.

## Reproducibility

- NumPy and TensorFlow seeds are set to 42.
- The ImageNet MobileNetV2 weights come from the standard Keras application.
- The CIFAR-10 archive is fixed by the official dataset source.
- Fine-tuning uses the final 20 MobileNetV2 layers, excluding batch-normalisation
  layers, with Adam at learning rate `1e-5`.
