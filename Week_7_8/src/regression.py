import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate_regression(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train and evaluate all four mandated regression models:
    1. Linear Regression
    2. Polynomial Regression
    3. Ridge Regression
    4. Lasso Regression
    """
    
    models = {}
    predictions = {}
    metrics = []
    
    # ==========================================
    # ALGORITHM 1: Linear Regression
    # ==========================================
    # Standard Ordinary Least Squares (OLS) Linear Regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)
    models['Linear Regression'] = lin_reg
    pred_lin = lin_reg.predict(X_test_scaled)
    predictions['Linear Regression'] = pred_lin
    
    # ==========================================
    # ALGORITHM 2: Polynomial Regression
    # ==========================================
    # Polynomial features expansion (Degree = 2) combined with Linear Regression
    poly_pipeline = Pipeline([
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('linear_reg', LinearRegression())
    ])
    poly_pipeline.fit(X_train_scaled, y_train)
    models['Polynomial Regression'] = poly_pipeline
    pred_poly = poly_pipeline.predict(X_test_scaled)
    predictions['Polynomial Regression'] = pred_poly
    
    # ==========================================
    # ALGORITHM 3: Ridge Regression
    # ==========================================
    # L2 Regularized Linear Regression to prevent overfitting
    ridge_reg = Ridge(alpha=1.0, random_state=42)
    ridge_reg.fit(X_train_scaled, y_train)
    models['Ridge Regression'] = ridge_reg
    pred_ridge = ridge_reg.predict(X_test_scaled)
    predictions['Ridge Regression'] = pred_ridge
    
    # ==========================================
    # ALGORITHM 4: Lasso Regression
    # ==========================================
    # L1 Regularized Linear Regression for feature selection & regularization
    lasso_reg = Lasso(alpha=0.1, random_state=42)
    lasso_reg.fit(X_train_scaled, y_train)
    models['Lasso Regression'] = lasso_reg
    pred_lasso = lasso_reg.predict(X_test_scaled)
    predictions['Lasso Regression'] = pred_lasso
    
    # ==========================================
    # Evaluation Metrics Calculation
    # ==========================================
    for model_name, preds in predictions.items():
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        metrics.append({
            'Model': model_name,
            'MAE (kWh)': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE (kWh)': round(rmse, 4),
            'R² Score': round(r2, 4)
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    # Identify the best model based strictly on highest R² score
    best_model_name = metrics_df.sort_values(by='R² Score', ascending=False).iloc[0]['Model']
    best_model = models[best_model_name]
    
    return {
        'models': models,
        'predictions': predictions,
        'metrics_df': metrics_df,
        'best_model_name': best_model_name,
        'best_model': best_model
    }
