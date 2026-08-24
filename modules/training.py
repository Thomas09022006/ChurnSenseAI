"""
Model Training logic module for Prompt 6.
"""

import pandas as pd
import streamlit as st
import os
import joblib
from utils.training_helpers import train_single_model

def train_selected_models(
    models_to_train: list, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> dict:
    """Train all selected models and save best model."""
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(models_to_train)
    for idx, model_name in enumerate(models_to_train):
        status_text.text(f"⏳ Training {model_name}...")
        res = train_single_model(model_name, X_train, y_train, X_test, y_test)
        results[model_name] = res
        progress_bar.progress(int(((idx + 1) / total) * 100))
        
    status_text.text("✅ Training Complete!")
    
    # Identify best model based on Accuracy
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_res = results[best_model_name]
    
    # Save best model to disk in saved_models directory
    save_dir = os.path.join(os.getcwd(), "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.joblib")
    
    saved_bundle = {
        "best_model_name": best_model_name,
        "model": best_res["model_obj"],
        "accuracy": best_res["accuracy"],
        "roc_auc": best_res["roc_auc"],
        "precision": best_res["precision"],
        "recall": best_res["recall"],
        "feature_names": list(X_train.columns),
        "scaler": st.session_state.get("scaler", None)
    }
    joblib.dump(saved_bundle, best_model_path)
    
    # Store in session state
    st.session_state['trained_models'] = results
    st.session_state['best_model_name'] = best_model_name
    st.session_state['best_model_bundle'] = saved_bundle
    st.session_state['saved_model_path'] = best_model_path
    
    return results
