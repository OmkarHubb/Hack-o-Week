import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import load_or_create_dataset, preprocess_data, FEATURE_COLUMNS
from src.regression import train_and_evaluate_regression
from src.classification import train_and_evaluate_classification
from src.prediction import predict_electricity_usage

# Page Configuration
st.set_page_config(
    page_title="Electricity Consumption ML Case Study",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.2rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    .result-badge-normal {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.6rem 1rem;
        border-radius: 0.4rem;
        font-weight: 600;
        font-size: 1.2rem;
        display: inline-block;
    }
    .result-badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.6rem 1rem;
        border-radius: 0.4rem;
        font-weight: 600;
        font-size: 1.2rem;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Cache Data and Pipeline Execution for smooth UI performance
@st.cache_data
def load_and_train_pipeline():
    df, threshold = load_or_create_dataset("data/electricity_data.csv")
    data_dict = preprocess_data(df)
    
    # Train Regression Models
    reg_results = train_and_evaluate_regression(
        data_dict['X_train_scaled'],
        data_dict['X_test_scaled'],
        data_dict['y_reg_train'],
        data_dict['y_reg_test']
    )
    
    # Train Classification Models
    clf_results = train_and_evaluate_classification(
        data_dict['X_train_scaled'],
        data_dict['X_test_scaled'],
        data_dict['y_clf_train'],
        data_dict['y_clf_test']
    )
    
    return df, threshold, data_dict, reg_results, clf_results

# Execute Pipeline
df, threshold, data_dict, reg_results, clf_results = load_and_train_pipeline()

# Title and Objective
st.markdown('<div class="main-header">⚡ Electricity Consumption & Usage Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning Case Study: Regression (Consumption kWh) & Classification (Normal vs High Usage)</div>', unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Dataset / Overview",
    "2. Regression Results",
    "3. Classification Results",
    "4. Prediction"
])

# ==========================================
# TAB 1: DATASET / OVERVIEW
# ==========================================
with tab1:
    st.header("Dataset Overview & Feature Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features Count", len(FEATURE_COLUMNS))
    with col3:
        st.metric("Mean Consumption", f"{df['Electricity_Consumption'].mean():.2f} kWh")
    with col4:
        st.metric("High Usage Threshold (75th percentile)", f"{threshold:.2f} kWh")
        
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Feature Statistics")
    st.dataframe(df.describe().T[['mean', 'std', 'min', '50%', 'max']], use_container_width=True)

# ==========================================
# TAB 2: REGRESSION RESULTS
# ==========================================
with tab2:
    st.header("Regression Model Performance Comparison")
    st.caption("Algorithms: Linear Regression, Polynomial Regression (Deg 2), Ridge Regression, Lasso Regression")
    
    # Model Comparison Table
    st.subheader("Model Evaluation Metrics")
    st.dataframe(reg_results['metrics_df'], use_container_width=True)
    
    best_reg_name = reg_results['best_model_name']
    best_r2 = reg_results['metrics_df'].loc[reg_results['metrics_df']['Model'] == best_reg_name, 'R² Score'].values[0]
    st.success(f"🏆 **Strongest Regression Model:** `{best_reg_name}` with **R² Score = {best_r2:.4f}**")
    
    st.subheader("Visualizations")
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Actual vs Predicted Plot
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        y_test = data_dict['y_reg_test']
        preds = reg_results['predictions'][best_reg_name]
        
        ax1.scatter(y_test, preds, alpha=0.6, color='#2563EB', edgecolors='k')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal Prediction")
        ax1.set_xlabel("Actual Consumption (kWh)")
        ax1.set_ylabel("Predicted Consumption (kWh)")
        ax1.set_title(f"Actual vs Predicted ({best_reg_name})")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig1)
        
    with col_b:
        # Regression R2 and RMSE Comparison Chart
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        metrics_df = reg_results['metrics_df']
        sns.barplot(data=metrics_df, x='Model', y='R² Score', palette='Blues_d', ax=ax2)
        ax2.set_title("R² Score Comparison Across Models")
        ax2.set_ylim(0, 1.05)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15)
        for p in ax2.patches:
            ax2.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
        ax2.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig2)

    # Feature Importance / Coefficients (Linear Regression)
    st.subheader("Feature Coefficients (Linear & Regularized Models)")
    lin_model = reg_results['models']['Linear Regression']
    coef_df = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Linear Coef': lin_model.coef_,
        'Ridge Coef': reg_results['models']['Ridge Regression'].coef_,
        'Lasso Coef': reg_results['models']['Lasso Regression'].coef_
    })
    st.dataframe(coef_df, use_container_width=True)

# ==========================================
# TAB 3: CLASSIFICATION RESULTS
# ==========================================
with tab3:
    st.header("Classification Model Performance Comparison")
    st.caption("Algorithms: Logistic Regression, K-Nearest Neighbors (KNN)")
    
    # Model Comparison Table
    st.subheader("Model Evaluation Metrics")
    st.dataframe(clf_results['metrics_df'], use_container_width=True)
    
    best_clf_name = clf_results['best_model_name']
    best_f1 = clf_results['metrics_df'].loc[clf_results['metrics_df']['Model'] == best_clf_name, 'F1-Score'].values[0]
    st.success(f"🏆 **Top Classification Model:** `{best_clf_name}` with **F1-Score = {best_f1:.4f}**")
    
    # Confusion Matrices Visualization
    st.subheader("Confusion Matrices")
    col_c1, col_c2 = st.columns(2)
    
    cm_log = clf_results['confusion_matrices']['Logistic Regression']
    cm_knn = clf_results['confusion_matrices']['KNN']
    
    with col_c1:
        fig_cm1, ax_cm1 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Normal (0)', 'High (1)'],
                    yticklabels=['Normal (0)', 'High (1)'], ax=ax_cm1)
        ax_cm1.set_title("Logistic Regression Confusion Matrix")
        ax_cm1.set_xlabel("Predicted Label")
        ax_cm1.set_ylabel("Actual Label")
        st.pyplot(fig_cm1)
        
    with col_c2:
        fig_cm2, ax_cm2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', cbar=False,
                    xticklabels=['Normal (0)', 'High (1)'],
                    yticklabels=['Normal (0)', 'High (1)'], ax=ax_cm2)
        ax_cm2.set_title("KNN Classifier Confusion Matrix")
        ax_cm2.set_xlabel("Predicted Label")
        ax_cm2.set_ylabel("Actual Label")
        st.pyplot(fig_cm2)

# ==========================================
# TAB 4: PREDICTION
# ==========================================
with tab4:
    st.header("Interactive Electricity Consumption & Usage Predictor")
    st.write("Enter feature values below to get real-time predictions for consumption (kWh) and usage category.")
    
    with st.form("prediction_form"):
        col_i1, col_i2 = st.columns(2)
        
        with col_i1:
            temperature = st.slider("Temperature (°C)", min_value=10.0, max_value=45.0, value=28.0, step=0.5)
            humidity = st.slider("Humidity (%)", min_value=10.0, max_value=100.0, value=55.0, step=1.0)
            hour = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=19, step=1)
            day_of_week = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6],
                                       format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
        
        with col_i2:
            occupants = st.number_input("Number of Occupants", min_value=1, max_value=12, value=4, step=1)
            appliance_hours = st.slider("Appliance Usage Hours", min_value=0.0, max_value=24.0, value=8.5, step=0.5)
            prev_consumption = st.number_input("Previous Consumption (kWh)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
            
        submit_button = st.form_submit_button("Predict Electricity Usage", use_container_width=True)
        
    if submit_button:
        input_data = {
            'Temperature': temperature,
            'Humidity': humidity,
            'Hour': hour,
            'Day_of_Week': day_of_week,
            'Occupants': occupants,
            'Appliance_Usage_Hours': appliance_hours,
            'Previous_Consumption': prev_consumption
        }
        
        # Use trained scaler, best regression model, and best classification model
        result = predict_electricity_usage(
            input_data,
            data_dict['scaler'],
            reg_results['best_model'],
            clf_results['best_model']
        )
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(
                label="Predicted Electricity Consumption",
                value=f"{result['predicted_consumption_kwh']:.2f} kWh",
                delta=f"Regression Model: {reg_results['best_model_name']}"
            )
            
        with res_col2:
            label = result['predicted_class_label']
            if result['predicted_class_code'] == 1:
                st.markdown(f"**Predicted Usage Category:** <br><span class='result-badge-high'>🚨 {label} (≥ {threshold:.2f} kWh)</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**Predicted Usage Category:** <br><span class='result-badge-normal'>✅ {label} (< {threshold:.2f} kWh)</span>", unsafe_allow_html=True)
