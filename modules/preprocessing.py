"""
Preprocessing Business Logic Module for Prompt 5.
"""

import pandas as pd
import streamlit as st
import joblib
import io
from utils.preprocessing_helpers import (
    clean_dataset, engineer_features, encode_and_scale, perform_train_test_split
)

def run_preprocessing_pipeline(df: pd.DataFrame) -> dict:
    """Execute complete data preprocessing pipeline and save results in session state."""
    # 1. Cleaning
    clean_df, cleaning_summary = clean_dataset(df)
    
    # 2. Feature Engineering
    fe_df = engineer_features(clean_df)
    
    # 3. Encoding & Scaling
    X_proc, y_proc, scaler, cat_cols, feature_names, encoding_summary = encode_and_scale(fe_df, target_col="Churn")
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = perform_train_test_split(X_proc, y_proc, test_size=0.2, random_state=42)
    
    results = {
        "cleaned_df": clean_df,
        "fe_df": fe_df,
        "X_processed": X_proc,
        "y_processed": y_proc,
        "scaler": scaler,
        "cat_cols": cat_cols,
        "feature_names": feature_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "cleaning_summary": cleaning_summary,
        "encoding_summary": encoding_summary
    }
    
    # Save to session state
    st.session_state['preprocessing_results'] = results
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.session_state['scaler'] = scaler
    st.session_state['feature_names'] = feature_names
    
    return results
