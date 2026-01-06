# Machine Learning Project – Analysis of Classical Ensemble Models on Fashion-MNIST

This repository contains the implementation of a machine learning project focused on analyzing the behavior and limitations of classical ensemble models for image classification. The project uses the Fashion-MNIST dataset and investigates systematic misclassifications using Random Forest–based models and confidence-aware error analysis.

The goal of the project is not to maximize classification accuracy, but to understand model behavior, error patterns, and the impact of different modeling choices when working with raw pixel representations.

---

## Repository Structure
configs/<br>
&nbsp;baseline.yaml<br>
&nbsp;exp1_pca_rf.yaml<br>
&nbsp;exp2_extratrees.yaml<br>

data/<br>
&nbsp;X.npy<br>
&nbsp;y.npy<br>
&nbsp;splits/<br>
&nbsp;&nbsp;idx_train.npy<br>
&nbsp;&nbsp;idx_val.npy<br>
&nbsp;&nbsp;idx_test.npy<br>

results/<br>
&nbsp;Automatically generated experiment outputs

src/<br>
&nbsp;download_data.py<br>
&nbsp;split_data.py<br>
&nbsp;train_baseline.py<br>
&nbsp;train_pca_rf.py<br>
&nbsp;train_extratrees.py<br>
&nbsp;analyze_confidence.py<br>
---

## Environment Setup

The project was developed using Python 3.11.  
It is recommended to use a virtual environment.

### Install required packages

```bash
pip install numpy pandas scikit-learn pyyaml
```
---
## Dataset Preparation
Download and preprocess Fashion-MNIST (The dataset was not uploaded to github due to its large size)
```bash
python -m src.download_data --config configs/baseline.yaml
```
This script downloads the dataset, flattens the images into pixel vectors, and stores them in the data/ directory as X.npy and y.npy.

---
## Create train / validation / test splits
```bash
python -m src.split_data --config configs/baseline.yaml
```
The dataset is split into:
- 64% training
- 16% validation
- 20% test

The split indices are saved in data/splits/ and reused across all experiments to ensure reproducibility.

## Running the Experiments

All experiments are controlled via YAML configuration files located in the configs/ directory.

### Baseline: Random Forest
```bash
python -m src.train_baseline --config configs/baseline.yaml
```
### Experiment 1: PCA + Random Forest
```bash
python -m src.train_pca_rf --config configs/exp1_pca_rf.yaml
````
### Experiment 2: ExtraTrees Classifier
```bash
python -m src.train_extratrees --config configs/exp2_extratrees.yaml
```

### Output Files

Each experiment creates a timestamped directory inside results/ containing:
- metrics.json
- confusion_val.csv
- confusion_test.csv
- classification_report_val.txt
- classification_report_test.txt
- preds_val.csv
- preds_test.csv
- config_used.yaml


---
## Confidence-Based Error Analysis

To analyze high-confidence misclassifications, run:
```bash
python -m src.analyze_confidence
```
But fist you have to change the run_dir constant to the correct folder you want to analyze.
This script identifies incorrect predictions, analyzes confidence distributions, and highlights systematic high-confidence errors across classes

## Reproducibility

- Fixed random seeds are used throughout the project

- Train, validation, and test splits are explicitly stored

- All experiment configurations are saved together with results

Running the same scripts with the same configuration files reproduces the reported results.