"""
Inference engine module for Prompt 7.
"""

import pandas as pd
import streamlit as st
import datetime
from utils.prediction_helpers import (
    prepare_customer_input_df, preprocess_single_customer, calculate_confidence_level, generate_prediction_summary
)

def run_customer_inference(inputs: dict, model_bundle: dict) -> dict:
    """Execute customer churn prediction pipeline using loaded model bundle."""
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]
    scaler = model_bundle.get("scaler", None)
    
    # Raw customer DataFrame
    raw_df = prepare_customer_input_df(inputs)
    
    # Preprocess row
    X_cust = preprocess_single_customer(raw_df, feature_names, scaler)
    
    # Predict
    pred_class = int(model.predict(X_cust)[0])
    prob_arr = model.predict_proba(X_cust)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
    churn_prob = float(prob_arr[1])
    
    confidence, risk_level, risk_badge, color = calculate_confidence_level(churn_prob)
    summary_bullets = generate_prediction_summary(inputs, churn_prob, risk_level)
    
    prediction_result = {
        "prediction_label": "Customer Likely to Churn" if pred_class == 1 else "Customer Likely to Stay",
        "pred_class": pred_class,
        "churn_prob": churn_prob,
        "stay_prob": 1.0 - churn_prob,
        "confidence": confidence,
        "risk_level": risk_level,
        "risk_badge": risk_badge,
        "color": color,
        "summary": summary_bullets,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": inputs,
        "X_cust": X_cust
    }
    
    # Save to session state
    st.session_state['last_prediction'] = prediction_result
    
    return prediction_result
