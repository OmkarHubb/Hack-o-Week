# Electricity Consumption & Usage Prediction Using Machine Learning

## Project Objective
This case-study machine learning application performs **both** Regression and Classification on electricity consumption data:
1. **Regression**: Predicts continuous electricity consumption in kWh.
2. **Classification**: Classifies consumption as **Normal Usage (0)** or **High Usage (1)** based on a 75th-percentile threshold.

---

## Mandated Algorithms & Implementation

### 1. Regression Models (`src/regression.py`)
- **Linear Regression**: Standard OLS baseline model.
- **Polynomial Regression**: Degree=2 feature expansion combined with Linear Regression.
- **Ridge Regression**: L2-regularized linear model.
- **Lasso Regression**: L1-regularized linear model for feature selection.

**Evaluation Metrics**: MAE, MSE, RMSE, R² Score

### 2. Classification Models (`src/classification.py`)
- **Logistic Regression**: Linear binary classifier.
- **K-Nearest Neighbors (KNN)**: Non-parametric distance-based classifier ($k=5$).

**Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## Dataset Features
- **Temperature (°C)**
- **Humidity (%)**
- **Hour (0–23)**
- **Day of Week (0–6)**
- **Number of Occupants (1–8)**
- **Appliance Usage Hours (0–24)**
- **Previous Consumption (kWh)**
- **Electricity Consumption (Target kWh)**
- **Usage Class (Target Binary 0/1)**

---

## Project Structure
```
Week_7_8/
│
├── data/
│   └── electricity_data.csv          # Generated synthetic dataset
├── models/                           # Model artifacts directory
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Data creation, cleaning, train/test split, scaling
│   ├── regression.py                 # 4 Regression models training & evaluation
│   ├── classification.py             # 2 Classification models training & evaluation
│   └── prediction.py                 # Real-time single-sample inference function
│
├── app.py                            # Streamlit web application
├── requirements.txt                  # Python package dependencies
└── README.md                         # Documentation
```

---

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Streamlit App**:
   ```bash
   python -m streamlit run app.py
   ```
