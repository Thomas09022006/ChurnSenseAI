"""
Model Loader module for Prompt 7.
"""

import streamlit as st
import os
import joblib

@st.cache_resource
def load_trained_best_model():
    """Load cached trained model bundle."""
    filepath = os.path.join(os.getcwd(), "saved_models", "best_model.joblib")
    if not os.path.exists(filepath):
        return None
    return joblib.load(filepath)
