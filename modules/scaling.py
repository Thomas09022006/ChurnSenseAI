"""
Feature Scaling Module for Prompt 5.
"""

import pandas as pd
import streamlit as st

def render_scaling_summary(X_proc: pd.DataFrame):
    """Display feature scaling details."""
    st.markdown("### 🔢 Feature Scaling")
    st.markdown("StandardScaler applied to numerical features (mean = 0, std = 1). Target column excluded.")
    st.dataframe(X_proc.describe().T[['mean', 'std', 'min', 'max']].head(5), use_container_width=True)
