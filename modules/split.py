"""
Train Test Split Display Module for Prompt 5.
"""

import pandas as pd
import streamlit as st
import plotly.express as px

def render_split_summary(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    """Render Train/Test split stats and pie chart."""
    st.markdown("### ✂️ Train-Test Split (80/20)")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Training Samples", f"{len(X_train):,} (80%)")
    with c2:
        st.metric("Testing Samples", f"{len(X_test):,} (20%)")
    with c3:
        st.metric("Total Features", f"{X_train.shape[1]}")
    with c4:
        st.metric("Random State", "42 (Stratified)")
        
    split_df = pd.DataFrame({
        'Set': ['Training Set (80%)', 'Testing Set (20%)'],
        'Count': [len(X_train), len(X_test)]
    })
    fig = px.pie(
        split_df, 
        names='Set', 
        values='Count',
        title="Train / Test Split Ratio",
        color_discrete_sequence=['#6366F1', '#38BDF8'],
        hole=0.4
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        height=260,
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True)
