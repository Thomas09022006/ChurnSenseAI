"""
Outlier Detection Module for Prompt 4.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from utils.eda_helpers import detect_outliers_iqr

def render_outlier_analysis(df: pd.DataFrame):
    """Render outlier detection section for key numerical columns."""
    st.markdown("### 🎯 Outlier Detection (IQR Method)")
    
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if "TotalCharges" in df.columns and "TotalCharges" not in num_cols:
        num_cols.append("TotalCharges")
        
    target_num_cols = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges'] if c in df.columns or c in num_cols]
    if not target_num_cols:
        target_num_cols = num_cols[:3]
        
    if not target_num_cols:
        st.info("No numerical columns available for outlier detection.")
        return
        
    selected_col = st.selectbox("Select Numerical Feature for Outlier Analysis", target_num_cols)
    
    # Process numeric conversion if string column like TotalCharges
    working_df = df.copy()
    if selected_col == "TotalCharges":
        working_df["TotalCharges"] = pd.to_numeric(working_df["TotalCharges"], errors='coerce')
        
    num_outliers, pct_outliers, lower_bound, upper_bound, outliers = detect_outliers_iqr(working_df, selected_col)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Outlier Count", f"{num_outliers}")
    with col2:
        st.metric("Outlier Percentage", f"{pct_outliers:.2f}%")
    with col3:
        st.metric("Lower Bound", f"{lower_bound:.2f}" if lower_bound is not None else "N/A")
    with col4:
        st.metric("Upper Bound", f"{upper_bound:.2f}" if upper_bound is not None else "N/A")
        
    fig = px.box(
        working_df, 
        y=selected_col, 
        points="outliers",
        title=f"Outliers in {selected_col}",
        color_discrete_sequence=['#F59E0B']
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True)
