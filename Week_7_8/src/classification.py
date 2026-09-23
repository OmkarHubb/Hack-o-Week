import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_evaluate_classification(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train and evaluate both mandated classification models:
    5. Logistic Regression
    6. K-Nearest Neighbors (KNN)
    """
    
    models = {}
    predictions = {}
    confusion_matrices = {}
    metrics = []
    
    # ==========================================
    # ALGORITHM 5: Logistic Regression
    # ==========================================
    # Binary classification using Logistic Regression model
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = log_reg
    pred_log = log_reg.predict(X_test_scaled)
    predictions['Logistic Regression'] = pred_log
    confusion_matrices['Logistic Regression'] = confusion_matrix(y_test, pred_log)
    
    # ==========================================
    # ALGORITHM 6: K-Nearest Neighbors (KNN)
    # ==========================================
    # Distance-based non-parametric classifier with k=5 neighbors
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train_scaled, y_train)
    models['KNN'] = knn_clf
    pred_knn = knn_clf.predict(X_test_scaled)
    predictions['KNN'] = pred_knn
    confusion_matrices['KNN'] = confusion_matrix(y_test, pred_knn)
    
    # ==========================================
    # Evaluation Metrics Calculation
    # ==========================================
    for model_name, preds in predictions.items():
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        metrics.append({
            'Model': model_name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4)
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    # Identify the best classification model based on highest F1-Score
    best_model_name = metrics_df.sort_values(by='F1-Score', ascending=False).iloc[0]['Model']
    best_model = models[best_model_name]
    
    return {
        'models': models,
        'predictions': predictions,
        'confusion_matrices': confusion_matrices,
        'metrics_df': metrics_df,
        'best_model_name': best_model_name,
        'best_model': best_model
    }
