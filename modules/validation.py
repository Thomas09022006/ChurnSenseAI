"""
Validation Report Display Module for Prompt 2.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from utils.validation_helpers import validate_dataset

def render_validation_report(df: pd.DataFrame, report: dict):
    """Render validation status and summary."""
    st.markdown("### 📋 Validation Report")
    
    if report.get("is_overall_valid", False):
        st.success("✔ Dataset successfully validated! All required criteria met.")
    else:
        st.warning("⚠️ Dataset loaded with warnings or missing columns. Review the status below.")
        
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**✔ File Loaded**" if report.get("file_uploaded") else "**❌ File Error**")
    with c2:
        st.markdown("**✔ Format Valid**" if report.get("is_readable") else "**❌ Format Error**")
    with c3:
        st.markdown("**✔ Columns Valid**" if report.get("has_required_cols", False) else "**⚠️ Columns Differ**")
    with c4:
        st.markdown("**✔ Target Valid**" if report.get("target_valid") else "**❌ Target Invalid**")

def render_target_distribution_chart(df: pd.DataFrame, target_col: str = "Churn"):
    """Plot target distribution pie chart using Plotly."""
    if target_col not in df.columns:
        return
        
    counts = df[target_col].value_counts().reset_index()
    counts.columns = [target_col, 'Count']
    
    fig = px.pie(
        counts, 
        names=target_col, 
        values='Count', 
        title=f"Target Distribution ({target_col})",
        color_discrete_sequence=['#10B981', '#EF4444'],
        hole=0.4
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True)
