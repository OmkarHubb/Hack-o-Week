import numpy as np
import pandas as pd
from src.preprocessing import FEATURE_COLUMNS

def predict_electricity_usage(input_dict, scaler, regression_model, classification_model):
    """
    Given a dictionary of user input features, perform scaling and predict:
    1. Electricity consumption in kWh (Regression)
    2. Usage Category: Normal Usage or High Usage (Classification)
    """
    # Create input DataFrame matching feature columns
    input_df = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
    
    # Scale features using pre-fitted scaler
    input_scaled = scaler.transform(input_df)
    
    # Predict regression target
    pred_consumption = float(regression_model.predict(input_scaled)[0])
    pred_consumption = max(0.0, round(pred_consumption, 2))
    
    # Predict classification target
    pred_class_code = int(classification_model.predict(input_scaled)[0])
    pred_class_label = "High Usage" if pred_class_code == 1 else "Normal Usage"
    
    return {
        'predicted_consumption_kwh': pred_consumption,
        'predicted_class_code': pred_class_code,
        'predicted_class_label': pred_class_label
    }
