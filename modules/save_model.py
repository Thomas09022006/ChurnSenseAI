"""
Model Saving helper module for Prompt 6.
"""

import os
import joblib

def get_saved_model_bytes(filepath: str = None) -> bytes:
    """Read saved joblib file bytes for user download."""
    if filepath is None:
        filepath = os.path.join(os.getcwd(), "saved_models", "best_model.joblib")
        
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return f.read()
    return b""
