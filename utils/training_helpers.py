"""
Training helper functions for Prompt 6.
"""

import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def train_single_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Train a single model and record performance metrics."""
    start_time = time.time()
    
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42, max_depth=6)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    elif model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss')
        except ImportError:
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    model.fit(X_train, y_train)
    elapsed_time = round(time.time() - start_time, 3)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob) if len(np.unique(y_test)) > 1 else ([0, 1], [0, 1], None)
    
    # Feature importance or coefficients
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(X_train.shape[1])
        
    feat_imp_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    return {
        "model_name": model_name,
        "model_obj": model,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "training_time": elapsed_time,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "feature_importance": feat_imp_df
    }
