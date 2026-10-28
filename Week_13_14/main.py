import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# Ensemble Models
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Linear Model with L1/L2 Regularization
from sklearn.linear_model import LogisticRegression

def main():
    print("=" * 60)
    print("ENSEMBLE METHODS & REGULARIZATION PROJECT EXECUTOR")
    print("=" * 60)

    # 1. GENERATE SYNTHETIC DATASET
    # Creating a dataset with noise to demonstrate overfitting/underfitting
    X, y = make_classification(
        n_samples=2000, 
        n_features=20, 
        n_informative=12, 
        n_redundant=4, 
        n_clusters_per_class=2, 
        flip_y=0.1,  # Added noise to test variance & overfitting
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Dataset generated: {X_train.shape[0]} Train samples, {X_test.shape[0]} Test samples.\n")

    # 2. BIAS-VARIANCE & OVERFITTING DEMONSTRATION (L1/L2 REGULARIZATION)
    print("--- 1. Bias-Variance & Regularization (Logistic Regression) ---")
    
    # Underfitting / High Bias: Heavy Regularization (C is tiny)
    underfit_model = LogisticRegression(C=0.001, penalty='l2', solver='liblinear')
    underfit_model.fit(X_train, y_train)
    
    # Overfitting / High Variance: No Regularization (C is huge)
    overfit_model = LogisticRegression(C=1000.0, penalty='l2', solver='liblinear')
    overfit_model.fit(X_train, y_train)
    
    # Well-Balanced (L1 Regularization - Lasso)
    l1_model = LogisticRegression(C=1.0, penalty='l1', solver='liblinear')
    l1_model.fit(X_train, y_train)

    print(f"Underfit Model (High Bias)    -> Train Acc: {underfit_model.score(X_train, y_train):.4f} | Test Acc: {underfit_model.score(X_test, y_test):.4f}")
    print(f"Overfit Model (High Variance) -> Train Acc: {overfit_model.score(X_train, y_train):.4f} | Test Acc: {overfit_model.score(X_test, y_test):.4f}")
    print(f"L1 Regularized Model (Lasso)  -> Train Acc: {l1_model.score(X_train, y_train):.4f} | Test Acc: {l1_model.score(X_test, y_test):.4f}\n")

    # 3. ENSEMBLE METHODS: BAGGING & BOOSTING
    print("--- 2. Ensemble Methods Evaluation ---")

    models = {
        "Bagging (Random Forest)": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Boosting (XGBoost)": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, reg_alpha=0.1, reg_lambda=1.0, random_state=42, eval_metric='logloss'),
        "Boosting (LightGBM)": LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results.append({"Model": name, "Train Accuracy": train_acc, "Test Accuracy": test_acc})
        print(f"{name:25s} -> Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    # 4. EXPORT SUMMARY TABLE
    df_results = pd.DataFrame(results)
    df_results.to_csv("results_summary.csv", index=False)
    print("\nResults saved successfully to 'results_summary.csv'.")

if __name__ == "__main__":
    main()