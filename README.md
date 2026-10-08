# Motor Imagery EEG Decoding Using Machine Learning

## Overview

This project explores the classification of left- and right-hand motor imagery from EEG recordings using statistical and machine learning methods.

The goal is to investigate how EEG spectral features and spatial filtering techniques can support neural decoding, with particular attention to generalization across recording runs.

## Dataset

The project uses the publicly available **PhysioNet EEG Motor Movement/Imagery Dataset**.

- Participant: Subject 1
- Recording runs: 4, 8, and 12
- EEG channels: 64
- Sampling frequency: 160 Hz
- Classification task: Left-hand vs. right-hand motor imagery

Each recording run provides 15 motor imagery trials.

## Methods

EEG data were processed using MNE-Python. Four-second epochs were extracted based on experimental event markers.

Two main classification approaches were investigated:

**1. Band-power features + Logistic Regression**

Mu-band (8–13 Hz) and beta-band (13–30 Hz) spectral power were extracted from electrodes C3 and C4 and used as features for logistic regression.

**2. Common Spatial Patterns (CSP) + Linear Discriminant Analysis (LDA)**

CSP was used to extract discriminative spatial features from the EEG signals. The resulting features were classified using LDA.

Additional experiments compared full-band, mu-band, and beta-band CSP models.

## Results

### Exploratory Model Comparison

| Model | Run 12 Accuracy |
|---|---:|
| Logistic Regression (4 band-power features) | 60.0% |
| CSP + LDA (8–30 Hz) | 66.7% |
| CSP + LDA (beta, 13–30 Hz) | 73.3% |
| CSP + LDA (mu, 8–13 Hz) | 80.0% |

### Leave-One-Run-Out Evaluation

To investigate robustness across recording runs, each run was held out for evaluation while the remaining two runs were used for training.

| Held-out Run | Accuracy |
|---|---:|
| Run 4 | 73.3% |
| Run 8 | 66.7% |
| Run 12 | 80.0% |
| **Mean** | **73.3%** |

Mu-band CSP + LDA achieved the highest observed accuracy in the exploratory model comparison and a mean leave-one-run-out accuracy of 73.3%.

## Limitations

This study is exploratory and includes only one participant and 45 trials across three recording runs.

The small evaluation sets introduce substantial uncertainty. Additionally, frequency-band and model selection were informed by Run 12 performance, so the reported accuracies should not be interpreted as unbiased estimates of generalization performance.

The CSP spatial patterns also did not consistently demonstrate clear sensorimotor localization.

## Future Work

Potential extensions include evaluating multiple participants, implementing filter-bank CSP, applying nested cross-validation, and investigating transfer learning and domain adaptation for more robust neural decoding.

## Tools

Python, MNE-Python, NumPy, pandas, Matplotlib, SciPy, and scikit-learn.

## Data Source

PhysioNet EEG Motor Movement/Imagery Dataset:

https://physionet.org/content/eegmmidb/1.0.0/
