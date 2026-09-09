# Loan Approval Prediction Using Machine Learning

A complete, beginner-friendly machine learning project that predicts whether a loan application should be **Approved** or **Rejected** based on applicant information.

---

## Project Objective

Build a binary classification model using **Logistic Regression** to predict loan approval status. The project demonstrates 10 core ML concepts:

1. Train/Test Split
2. Cross-Validation
3. Confusion Matrix
4. Precision
5. Recall
6. F1 Score
7. ROC-AUC
8. Feature Engineering
9. Feature Scaling
10. Handling Missing Data

---

## Dataset

The dataset (`data/loan_data.csv`) contains **614 loan applications** with the following features:

| Feature | Description |
|---------|-------------|
| Loan_ID | Unique loan identifier |
| Gender | Male / Female |
| Married | Yes / No |
| Dependents | Number of dependents (0, 1, 2, 3+) |
| Education | Graduate / Not Graduate |
| Self_Employed | Yes / No |
| ApplicantIncome | Applicant's monthly income |
| CoapplicantIncome | Coapplicant's monthly income |
| LoanAmount | Loan amount (in thousands) |
| Loan_Amount_Term | Loan repayment term (in months) |
| Credit_History | Credit history (1 = good, 0 = bad) |
| Property_Area | Urban / Semiurban / Rural |
| **Loan_Status** | **Target: Y (Approved) / N (Rejected)** |

---

## Technologies Used

- **Python 3.8+**
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computing
- **Matplotlib** — Static visualizations
- **Seaborn** — Statistical data visualization
- **Scikit-learn** — Machine learning library

---

## ML Algorithm

**Logistic Regression** — A simple and effective algorithm for binary classification that outputs probabilities, making it ideal for this approval/rejection problem.

---

## Preprocessing

1. **Missing Data Handling**
   - Numerical columns → Median imputation
   - Categorical columns → Mode (most frequent) imputation

2. **Feature Engineering**
   - `TotalIncome` = ApplicantIncome + CoapplicantIncome
   - `LoanIncomeRatio` = LoanAmount / (TotalIncome/1000 + 1)
   - `LogTotalIncome` = log(1 + TotalIncome)

3. **Categorical Encoding** — One-Hot Encoding for nominal categories

4. **Feature Scaling** — StandardScaler on numerical features (inside a Pipeline to prevent data leakage)

---

## Train/Test Split

- **80% Training / 20% Testing**
- Stratified split to preserve class distribution
- Fixed `random_state=42` for reproducibility

---

## Cross-Validation

- **5-Fold Stratified Cross-Validation** on training data
- Metrics: Accuracy and ROC-AUC
- Ensures robust performance estimation without using the test set

---

## Evaluation Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.8862 |
| Precision | 0.9143 |
| Recall | 0.9505 |
| F1 Score | 0.9320 |
| ROC-AUC | 0.8492 |
| Mean CV Accuracy | 0.8859 |
| Mean CV ROC-AUC | 0.8677 |

---

## Installation

1. **Clone or download** this project

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

```bash
python src/loan_prediction.py
```

The script will:
1. Load and explore the dataset
2. Handle missing values
3. Engineer new features
4. Encode categorical variables
5. Split data into training and test sets
6. Build a preprocessing pipeline with scaling
7. Train a Logistic Regression model
8. Perform 5-fold cross-validation
9. Evaluate on the test set
10. Save all plots to `outputs/`

---

## Expected Outputs

### Console Output
- Dataset exploration summary
- Missing value analysis
- Feature engineering details
- Cross-validation scores
- Complete evaluation metrics with interpretations

### Generated Files (in `outputs/`)
| File | Description |
|------|-------------|
| `loan_approval_distribution.png` | Bar chart of approved vs rejected loans |
| `missing_values.png` | Horizontal bar chart of missing values per column |
| `income_distribution.png` | Histograms of applicant and coapplicant income |
| `loan_amount_distribution.png` | Histogram of loan amounts |
| `confusion_matrix.png` | Heatmap showing TP, TN, FP, FN |
| `roc_curve.png` | ROC curve with AUC score |
| `results.json` | All metrics in JSON format |

---

## Project Structure

```
loan-approval-prediction/
│
├── data/
│   └── loan_data.csv              # Dataset (614 rows, 13 columns)
│
├── src/
│   └── loan_prediction.py         # Complete ML pipeline script
│
├── outputs/
│   ├── loan_approval_distribution.png
│   ├── missing_values.png
│   ├── income_distribution.png
│   ├── loan_amount_distribution.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── results.json
│
├── report/
│   └── project_explanation.md     # Detailed project explanation
│
├── requirements.txt               # Python dependencies
│
└── README.md                      # This file
```

---

## Detailed Explanation

For a complete walkthrough of every concept and every file in this project, see:

📄 [`report/project_explanation.md`](report/project_explanation.md)

---

## License

This project is for educational purposes.
