import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define feature columns
FEATURE_COLUMNS = [
    'Temperature',
    'Humidity',
    'Hour',
    'Day_of_Week',
    'Occupants',
    'Appliance_Usage_Hours',
    'Previous_Consumption'
]

def generate_synthetic_data(num_records=1200, random_state=42):
    """
    Generate a realistic synthetic electricity consumption dataset.
    """
    np.random.seed(random_state)
    
    temperature = np.round(np.random.uniform(15.0, 42.0, num_records), 1)
    humidity = np.round(np.random.uniform(20.0, 90.0, num_records), 1)
    hour = np.random.randint(0, 24, num_records)
    day_of_week = np.random.randint(0, 7, num_records)
    occupants = np.random.randint(1, 9, num_records)
    appliance_usage = np.round(np.random.uniform(1.0, 16.0, num_records), 1)
    previous_consumption = np.round(np.random.uniform(5.0, 50.0, num_records), 1)
    
    # Peak hour effect (higher consumption between 17:00 and 22:00)
    peak_hour_effect = np.where((hour >= 17) & (hour <= 22), 4.5, 0.0)
    
    # Weekend effect (higher usage on Sat/Sun: days 5 and 6)
    weekend_effect = np.where(day_of_week >= 5, 2.0, 0.0)
    
    # Realistic consumption formula with noise
    noise = np.random.normal(0, 2.5, num_records)
    electricity_consumption = (
        0.35 * temperature +
        0.05 * humidity +
        0.90 * occupants +
        1.60 * appliance_usage +
        0.45 * previous_consumption +
        peak_hour_effect +
        weekend_effect +
        noise
    )
    
    # Ensure positive consumption values
    electricity_consumption = np.round(np.maximum(electricity_consumption, 3.0), 2)
    
    # Define binary classification target based on 75th percentile threshold
    threshold = np.percentile(electricity_consumption, 75)
    usage_class = (electricity_consumption >= threshold).astype(int)
    
    df = pd.DataFrame({
        'Temperature': temperature,
        'Humidity': humidity,
        'Hour': hour,
        'Day_of_Week': day_of_week,
        'Occupants': occupants,
        'Appliance_Usage_Hours': appliance_usage,
        'Previous_Consumption': previous_consumption,
        'Electricity_Consumption': electricity_consumption,
        'Usage_Class': usage_class
    })
    
    return df, threshold

def load_or_create_dataset(data_path="data/electricity_data.csv"):
    """
    Load dataset from CSV or create synthetic dataset if not found.
    """
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        threshold = np.percentile(df['Electricity_Consumption'], 75)
        if 'Usage_Class' not in df.columns:
            df['Usage_Class'] = (df['Electricity_Consumption'] >= threshold).astype(int)
    else:
        df, threshold = generate_synthetic_data(num_records=1200, random_state=42)
        df.to_csv(data_path, index=False)
        
    return df, threshold

def preprocess_data(df):
    """
    Clean dataset, handle missing values, split into train/test, and scale features.
    """
    # 1. Clean data & handle missing values
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.dropna()
    
    # 2. Features and Targets
    X = df_cleaned[FEATURE_COLUMNS]
    y_reg = df_cleaned['Electricity_Consumption']
    y_clf = df_cleaned['Usage_Class']
    
    # 3. Train / Test Split
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )
    
    # 4. Feature Scaling using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'df': df_cleaned,
        'X': X,
        'y_reg': y_reg,
        'y_clf': y_clf,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_reg_train': y_reg_train,
        'y_reg_test': y_reg_test,
        'y_clf_train': y_clf_train,
        'y_clf_test': y_clf_test,
        'scaler': scaler
    }
