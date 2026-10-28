# Week 13-14: Ensemble Methods & Regularization Project

## Overview
This project demonstrates the core concepts of **Bias-Variance Tradeoff**, **L1/L2 Regularization**, and **Ensemble Learning** (Bagging and Boosting) using a noisy synthetic dataset generated with scikit-learn.

## Concepts Covered
- **Bias-Variance & Regularization (Logistic Regression)**:
  - **High Bias (Underfitting)**: Heavy L2 regularization ($C = 0.001$)
  - **High Variance (Overfitting)**: Negligible L2 regularization ($C = 1000.0$)
  - **Feature Selection / Well-Balanced**: L1 Regularization (Lasso, $C = 1.0$)
- **Ensemble Methods**:
  - **Bagging**: Random Forest Classifier
  - **Boosting**: XGBoost Classifier (`XGBClassifier`) and LightGBM Classifier (`LGBMClassifier`)

## Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project
Run the execution script:
```bash
python main.py
```

The script evaluates each model's training and testing accuracy and exports a summary comparison table to `results_summary.csv`.
