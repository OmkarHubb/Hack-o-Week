"""
Loan Approval Prediction Using Machine Learning
=================================================
This script demonstrates a complete ML pipeline for predicting loan approval
using Logistic Regression. It covers:

  1. Data Loading & Exploration
  2. Missing Data Handling
  3. Feature Engineering
  4. Categorical Encoding
  5. Train/Test Split (80/20, stratified)
  6. Feature Scaling (StandardScaler, no data leakage)
  7. Logistic Regression Model
  8. 5-Fold Cross-Validation
  9. Final Evaluation (Confusion Matrix, Precision, Recall, F1, ROC-AUC)
 10. Visualizations (saved to outputs/)

Author : Loan Prediction ML Project
Date   : 2026
"""

# ──────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────
import os
import sys
import warnings

# Fix Windows console encoding for Unicode characters
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set a consistent visual style for all plots
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.size": 11,
})

# ══════════════════════════════════════════════
# SECTION 1: DATA LOADING & EXPLORATION
# ══════════════════════════════════════════════
print("=" * 65)
print("  SECTION 1: DATA LOADING & EXPLORATION")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset Shape : {df.shape}")
print(f"Number of Rows: {df.shape[0]}")
print(f"Number of Cols: {df.shape[1]}")

print("\n--- First 5 Rows ---")
print(df.head().to_string())

print("\n--- Column Names ---")
print(list(df.columns))

print("\n--- Data Types ---")
print(df.dtypes.to_string())

print("\n--- Basic Statistical Summary (Numerical) ---")
print(df.describe().to_string())

print("\n--- Basic Statistical Summary (Categorical) ---")
print(df.describe(include="object").to_string())

# Identify numerical and categorical columns
# (Exclude Loan_ID since it's just an identifier)
id_col = "Loan_ID"
target_col = "Loan_Status"

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols = [c for c in categorical_cols if c not in [id_col, target_col]]

print(f"\nNumerical Columns  : {numerical_cols}")
print(f"Categorical Columns: {categorical_cols}")

# ── Missing-Value Analysis ──
print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(
    "Missing Count", ascending=False
)
print(missing_df.to_string())

# ── Target Variable Distribution ──
print("\n--- Target Variable Distribution ---")
target_counts = df[target_col].value_counts()
print(target_counts.to_string())
print(f"  Y (Approved) : {target_counts.get('Y', 0)} ({target_counts.get('Y', 0)/len(df)*100:.1f}%)")
print(f"  N (Rejected) : {target_counts.get('N', 0)} ({target_counts.get('N', 0)/len(df)*100:.1f}%)")


# ══════════════════════════════════════════════
# SECTION 2: VISUALIZATIONS (EDA)
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECTION 2: EXPLORATORY VISUALIZATIONS")
print("=" * 65)

# --- Plot 1: Loan Approval Distribution ---
fig, ax = plt.subplots(figsize=(6, 5))
colors = ["#2ecc71", "#e74c3c"]
target_counts.plot(kind="bar", color=colors, edgecolor="black", ax=ax)
ax.set_title("Loan Approval Distribution", fontweight="bold")
ax.set_xlabel("Loan Status")
ax.set_ylabel("Count")
ax.set_xticklabels(["Approved (Y)", "Rejected (N)"], rotation=0)
for i, v in enumerate(target_counts):
    ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loan_approval_distribution.png"), dpi=150)
plt.close()
print("  ✓ Saved: loan_approval_distribution.png")

# --- Plot 2: Missing Values Summary ---
if not missing_df.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    missing_df["Missing Count"].plot(
        kind="barh", color="#3498db", edgecolor="black", ax=ax
    )
    ax.set_title("Missing Values by Column", fontweight="bold")
    ax.set_xlabel("Number of Missing Values")
    ax.set_ylabel("Column")
    for i, v in enumerate(missing_df["Missing Count"]):
        ax.text(v + 0.3, i, f"{v} ({missing_df['Percentage (%)'].iloc[i]}%)", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "missing_values.png"), dpi=150)
    plt.close()
    print("  ✓ Saved: missing_values.png")

# --- Plot 3: Income Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df["ApplicantIncome"].dropna(), bins=40, color="#9b59b6", edgecolor="black", alpha=0.8)
axes[0].set_title("Applicant Income Distribution", fontweight="bold")
axes[0].set_xlabel("Applicant Income")
axes[0].set_ylabel("Frequency")

axes[1].hist(
    df["CoapplicantIncome"][df["CoapplicantIncome"] > 0].dropna(),
    bins=40,
    color="#e67e22",
    edgecolor="black",
    alpha=0.8,
)
axes[1].set_title("Coapplicant Income Distribution (Non-Zero)", fontweight="bold")
axes[1].set_xlabel("Coapplicant Income")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "income_distribution.png"), dpi=150)
plt.close()
print("  ✓ Saved: income_distribution.png")

# --- Plot 4: Loan Amount Distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["LoanAmount"].dropna(), bins=40, color="#1abc9c", edgecolor="black", alpha=0.8)
ax.set_title("Loan Amount Distribution", fontweight="bold")
ax.set_xlabel("Loan Amount (in thousands)")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loan_amount_distribution.png"), dpi=150)
plt.close()
print("  ✓ Saved: loan_amount_distribution.png")


# ══════════════════════════════════════════════
# SECTION 3: MISSING DATA HANDLING
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECTION 3: MISSING DATA HANDLING")
print("=" * 65)

# Numerical columns: fill with median
# Categorical columns: fill with mode (most frequent)
#
# We handle missing values manually here for transparency during EDA,
# but we also include imputation inside the Pipeline for the model
# to prevent data leakage.

num_impute_cols = [c for c in numerical_cols if df[c].isnull().any()]
cat_impute_cols = [c for c in categorical_cols if df[c].isnull().any()]

print(f"\n  Numerical columns with missing data  : {num_impute_cols}")
print(f"  Categorical columns with missing data : {cat_impute_cols}")

for col in num_impute_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"    → {col}: filled with median = {median_val}")

for col in cat_impute_cols:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)
    print(f"    → {col}: filled with mode = '{mode_val}'")

print(f"\n  Remaining missing values: {df.isnull().sum().sum()}")


# ══════════════════════════════════════════════
# SECTION 4: FEATURE ENGINEERING
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECTION 4: FEATURE ENGINEERING")
print("=" * 65)

# Feature 1: TotalIncome
# Combining applicant and coapplicant income gives a better picture
# of the household's total earning capacity.
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
print("\n  ✓ Created 'TotalIncome' = ApplicantIncome + CoapplicantIncome")
print("    → Rationale: Total household income is a stronger predictor than")
print("      individual incomes alone.")

# Feature 2: LoanIncomeRatio
# Measures how large the loan is relative to the total income.
# A higher ratio means the applicant is borrowing more compared to earnings.
df["LoanIncomeRatio"] = df["LoanAmount"] / (df["TotalIncome"] / 1000 + 1)
print("\n  ✓ Created 'LoanIncomeRatio' = LoanAmount / (TotalIncome/1000 + 1)")
print("    → Rationale: Captures affordability — a high ratio may signal risk.")
print("    → The +1 prevents division by zero.")

# Feature 3: LogTotalIncome
# Income distributions are typically right-skewed. A log transform
# brings the distribution closer to normal, which benefits Logistic Regression.
df["LogTotalIncome"] = np.log1p(df["TotalIncome"])
print("\n  ✓ Created 'LogTotalIncome' = log(1 + TotalIncome)")
print("    → Rationale: Reduces skewness in income for better model performance.")

print(f"\n  New feature columns: ['TotalIncome', 'LoanIncomeRatio', 'LogTotalIncome']")

# Update column lists
numerical_cols = numerical_cols + ["TotalIncome", "LoanIncomeRatio", "LogTotalIncome"]


# ══════════════════════════════════════════════
# SECTION 5: ENCODE TARGET VARIABLE
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECTION 5: ENCODE TARGET VARIABLE")
print("=" * 65)

# Map Loan_Status: Y → 1 (Approved), N → 0 (Rejected)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
print("\n  Loan_Status encoding: Y → 1 (Approved), N → 0 (Rejected)")
print(f"  Positive class (1): Approved")
print(f"  Negative class (0): Rejected")


# ══════════════════════════════════════════════
# SECTION 6: PREPARE FEATURES & TARGET
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECTION 6: PREPARE FEATURES & TARGET")
print("=" * 65)

# Drop Loan_ID (not a feature) and Loan_Status (target)
X = df.drop(columns=[id_col, target_col])
y = df[target_col]

# Re-identify column types for the pipeline
feature_numerical = X.select_dtypes(include=[np.number]).columns.tolist()
feature_categorical = X.select_dtypes(include=["object"]).columns.tolist()

print(f"\n  Feature matrix shape: {X.shape}")
print(f"  Numerical features  ({len(feature_numerical)}): {feature_numerical}")
print(f"  Categorical features ({len(feature_categorical)}): {feature_categorical}")
print(f"  Target shape: {y.shape}")


# ══════════════════════════════════════════════
# SECTION 7: TRAIN / TEST SPLIT
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECTION 7: TRAIN / TEST SPLIT")
print("=" * 65)

# Split 80% training, 20% testing.
# - random_state=42 ensures reproducibility.
# - stratify=y ensures the class distribution is preserved in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\n  Training set : {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"  Test set     : {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"\n  Training target distribution:")
print(f"    Approved (1): {(y_train == 1).sum()}")
print(f"    Rejected (0): {(y_train == 0).sum()}")
print(f"\n  Test target distribution:")
print(f"    Approved (1): {(y_test == 1).sum()}")
print(f"    Rejected (0): {(y_test == 0).sum()}")
print("\n  Note: The test set remains UNSEEN until final evaluation.")
print("  Stratification ensures both sets have similar class proportions.")


# ══════════════════════════════════════════════
# SECTION 8: PREPROCESSING PIPELINE
# (Feature Scaling + Categorical Encoding)
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECTION 8: PREPROCESSING PIPELINE (Scaling + Encoding)")
print("=" * 65)

# The preprocessing pipeline ensures:
#   1. Numerical features → Imputation (median) → StandardScaler
#   2. Categorical features → Imputation (mode) → One-Hot Encoding
#
# Using a Pipeline + ColumnTransformer means the scaler is fitted ONLY
# on the training data, preventing data leakage.

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, feature_numerical),
    ("cat", categorical_pipeline, feature_categorical),
])

print("\n  ColumnTransformer created with:")
print("    • Numerical pipeline   → SimpleImputer(median) → StandardScaler")
print("    • Categorical pipeline → SimpleImputer(mode) → OneHotEncoder")
print("\n  Why StandardScaler?")
print("    StandardScaler transforms features to have mean=0 and std=1.")
print("    This is important for Logistic Regression because it uses")
print("    gradient-based optimization, which converges faster when")
print("    features are on a similar scale.")
print("\n  Why Pipeline?")
print("    The pipeline ensures the scaler is fitted ONLY on training data.")
print("    If we scaled before splitting, test-set statistics would leak")
print("    into the training process (data leakage).")


# ══════════════════════════════════════════════
# SECTION 9: MODEL — LOGISTIC REGRESSION
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECTION 9: LOGISTIC REGRESSION MODEL")
print("=" * 65)

# Combine preprocessing + model into one pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )),
])

print("\n  Full pipeline:")
print("    Input → ColumnTransformer → LogisticRegression")
print("\n  Why Logistic Regression?")
print("    • Designed for binary classification problems")
print("    • Outputs probabilities (useful for ROC-AUC)")
print("    • Simple, interpretable, and effective baseline")
print("    • Works well with properly scaled features")


# ══════════════════════════════════════════════
# SECTION 10: 5-FOLD CROSS-VALIDATION
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECTION 10: 5-FOLD CROSS-VALIDATION")
print("=" * 65)

# Cross-validation is performed on the TRAINING data only.
# The test set remains completely untouched.
#
# In 5-fold CV:
#   - The training data is split into 5 equal parts (folds).
#   - The model is trained on 4 folds and validated on the remaining 1.
#   - This is repeated 5 times, each fold serving as validation once.
#   - The average score gives a more reliable performance estimate.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation: Accuracy
cv_accuracy_scores = cross_val_score(
    model_pipeline, X_train, y_train, cv=cv, scoring="accuracy"
)

# Cross-validation: ROC-AUC
cv_auc_scores = cross_val_score(
    model_pipeline, X_train, y_train, cv=cv, scoring="roc_auc"
)

print("\n  5-Fold Cross-Validation Results (on TRAINING data):")
print(f"\n  Accuracy per fold : {np.round(cv_accuracy_scores, 4)}")
print(f"  Mean CV Accuracy  : {cv_accuracy_scores.mean():.4f} ± {cv_accuracy_scores.std():.4f}")
print(f"\n  ROC-AUC per fold  : {np.round(cv_auc_scores, 4)}")
print(f"  Mean CV ROC-AUC   : {cv_auc_scores.mean():.4f} ± {cv_auc_scores.std():.4f}")
print("\n  Why cross-validate on training data?")
print("    Cross-validation provides a robust estimate of model performance")
print("    without touching the test set. If we used the test set here,")
print("    we would be 'peeking' at the final evaluation data, leading to")
print("    overly optimistic estimates.")


# ══════════════════════════════════════════════
# SECTION 11: TRAIN FINAL MODEL & EVALUATE
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECTION 11: FINAL MODEL EVALUATION (on TEST set)")
print("=" * 65)

# Train the final model on ALL training data
model_pipeline.fit(X_train, y_train)

# Predict on the unseen test set
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]  # Probability of class 1

# ── Accuracy ──
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\n  Accuracy: {test_accuracy:.4f}")

# ── Confusion Matrix ──
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n  Confusion Matrix:")
print(f"                    Predicted Rejected    Predicted Approved")
print(f"    Actual Rejected     {tn:>3} (TN)             {fp:>3} (FP)")
print(f"    Actual Approved     {fn:>3} (FN)             {tp:>3} (TP)")
print(f"\n    True Positives  (TP) = {tp} : Correctly predicted as Approved")
print(f"    True Negatives  (TN) = {tn} : Correctly predicted as Rejected")
print(f"    False Positives (FP) = {fp} : Wrongly predicted as Approved (actually Rejected)")
print(f"    False Negatives (FN) = {fn} : Wrongly predicted as Rejected (actually Approved)")

# ── Precision ──
test_precision = precision_score(y_test, y_pred)
print(f"\n  Precision: {test_precision:.4f}")
print(f"    → Of all loans the model predicted as Approved,")
print(f"      {test_precision*100:.1f}% were actually Approved.")
print(f"    → Higher precision = fewer incorrectly approved loans.")

# ── Recall ──
test_recall = recall_score(y_test, y_pred)
print(f"\n  Recall: {test_recall:.4f}")
print(f"    → Of all loans that should have been Approved,")
print(f"      the model correctly identified {test_recall*100:.1f}% of them.")
print(f"    → Higher recall = fewer missed worthy applicants.")

# ── F1 Score ──
test_f1 = f1_score(y_test, y_pred)
print(f"\n  F1 Score: {test_f1:.4f}")
print(f"    → Harmonic mean of Precision and Recall.")
print(f"    → Balances the trade-off between the two metrics.")

# ── ROC-AUC ──
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n  ROC-AUC: {test_roc_auc:.4f}")
print(f"    → Measures the model's ability to distinguish between")
print(f"      Approved and Rejected across all classification thresholds.")
print(f"    → AUC = 1.0 means perfect; AUC = 0.5 means random guessing.")

# ── Classification Report ──
print("\n  Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))


# ══════════════════════════════════════════════
# SECTION 12: CONFUSION MATRIX HEATMAP
# ══════════════════════════════════════════════
print("=" * 65)
print("  SECTION 12: CONFUSION MATRIX VISUALIZATION")
print("=" * 65)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Rejected (0)", "Approved (1)"],
    yticklabels=["Rejected (0)", "Approved (1)"],
    linewidths=1,
    linecolor="gray",
    cbar_kws={"label": "Count"},
    annot_kws={"size": 16, "fontweight": "bold"},
    ax=ax,
)
ax.set_title("Confusion Matrix — Loan Approval Prediction", fontweight="bold", fontsize=14)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("Actual Label", fontsize=12)

# Add TP/TN/FP/FN annotations
labels = [
    [f"TN = {tn}", f"FP = {fp}"],
    [f"FN = {fn}", f"TP = {tp}"],
]
for i in range(2):
    for j in range(2):
        ax.text(
            j + 0.5,
            i + 0.75,
            labels[i][j],
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("  ✓ Saved: confusion_matrix.png")


# ══════════════════════════════════════════════
# SECTION 13: ROC CURVE
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECTION 13: ROC CURVE")
print("=" * 65)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    fpr, tpr,
    color="#2980b9",
    linewidth=2.5,
    label=f"Logistic Regression (AUC = {test_roc_auc:.4f})",
)
ax.plot(
    [0, 1], [0, 1],
    color="gray",
    linestyle="--",
    linewidth=1,
    label="Random Classifier (AUC = 0.5)",
)
ax.fill_between(fpr, tpr, alpha=0.15, color="#2980b9")
ax.set_title("ROC Curve — Loan Approval Prediction", fontweight="bold", fontsize=14)
ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
plt.close()
print("  ✓ Saved: roc_curve.png")


# ══════════════════════════════════════════════
# SECTION 14: FINAL RESULTS SUMMARY
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("  FINAL RESULTS SUMMARY")
print("=" * 65)

print(f"""
  ┌─────────────────────────────────────────────┐
  │  Metric                      │  Value       │
  ├─────────────────────────────────────────────┤
  │  Accuracy                    │  {test_accuracy:.4f}       │
  │  Precision                   │  {test_precision:.4f}       │
  │  Recall                      │  {test_recall:.4f}       │
  │  F1 Score                    │  {test_f1:.4f}       │
  │  ROC-AUC                     │  {test_roc_auc:.4f}       │
  │  Mean CV Accuracy            │  {cv_accuracy_scores.mean():.4f}       │
  │  Mean CV ROC-AUC             │  {cv_auc_scores.mean():.4f}       │
  └─────────────────────────────────────────────┘
""")

print("  Interpretations:")
print(f"    • Accuracy ({test_accuracy:.4f}): The model correctly classifies")
print(f"      {test_accuracy*100:.1f}% of all loan applications.")
print(f"    • Precision ({test_precision:.4f}): When the model predicts Approved,")
print(f"      it is correct {test_precision*100:.1f}% of the time.")
print(f"    • Recall ({test_recall:.4f}): The model identifies {test_recall*100:.1f}% of")
print(f"      all actually approved loans.")
print(f"    • F1 Score ({test_f1:.4f}): Balances precision and recall into a single")
print(f"      metric. Useful when class distribution is imbalanced.")
print(f"    • ROC-AUC ({test_roc_auc:.4f}): The model's ability to distinguish")
print(f"      between Approved and Rejected across all thresholds.")

print("\n" + "=" * 65)
print("  PROJECT COMPLETE — All outputs saved to outputs/")
print("=" * 65)


# ══════════════════════════════════════════════
# SECTION 15: SAVE RESULTS TO FILE (for report)
# ══════════════════════════════════════════════
results = {
    "accuracy": test_accuracy,
    "precision": test_precision,
    "recall": test_recall,
    "f1_score": test_f1,
    "roc_auc": test_roc_auc,
    "cv_accuracy_mean": cv_accuracy_scores.mean(),
    "cv_accuracy_std": cv_accuracy_scores.std(),
    "cv_auc_mean": cv_auc_scores.mean(),
    "cv_auc_std": cv_auc_scores.std(),
    "cv_accuracy_folds": cv_accuracy_scores.tolist(),
    "cv_auc_folds": cv_auc_scores.tolist(),
    "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
}

import json

results_path = os.path.join(OUTPUT_DIR, "results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results also saved to: {results_path}")
