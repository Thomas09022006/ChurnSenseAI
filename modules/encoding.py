"""
Encoding Display Module for Prompt 5.
"""

import pandas as pd
import streamlit as st

def render_encoding_summary(cat_cols: list, num_encoded_features: int):
    """Display categorical encoding status."""
    st.markdown("### 🏷️ Categorical Encoding")
    st.write(f"- **Categorical Features Encoded**: {len(cat_cols)}")
    st.write(f"- **Total Resulting Features**: {num_encoded_features}")
    st.info(f"One-Hot Encoding applied to categorical columns: `{', '.join(cat_cols[:6])}...`")
