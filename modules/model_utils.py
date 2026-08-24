"""
Model Utilities Module.
"""

import os
import joblib

def load_saved_model_bundle(filepath: str = None) -> dict:
    """Load model bundle from filepath or default saved_models directory."""
    if filepath is None:
        filepath = os.path.join(os.getcwd(), "saved_models", "best_model.joblib")
        
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Saved model bundle not found at {filepath}")
        
    bundle = joblib.load(filepath)
    return bundle
