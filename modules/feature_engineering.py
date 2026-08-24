"""
Feature Engineering Display Module for Prompt 5.
"""

import pandas as pd
import streamlit as st

def render_feature_engineering_summary(fe_df: pd.DataFrame):
    """Render details of newly engineered features."""
    st.markdown("### 💡 Feature Engineering")
    st.markdown("Automated creation of domain-specific features:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**`tenure_group`**")
        st.caption("Bucketed tenure into: 0–12m, 13–24m, 25–48m, 48+m")
    with col2:
        st.markdown("**`charge_category`**")
        st.caption("Categorized monthly charges: Low, Medium, High")
    with col3:
        st.markdown("**`avg_monthly_revenue`**")
        st.caption("Ratio of TotalCharges / tenure (division safe)")

    if 'tenure_group' in fe_df.columns:
        st.dataframe(fe_df[['tenure', 'tenure_group', 'MonthlyCharges', 'charge_category', 'avg_monthly_revenue']].head(5), use_container_width=True)
