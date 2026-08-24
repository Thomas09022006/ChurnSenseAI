"""
SHAP Analysis Charts & Visualizations Module for Prompt 8.
"""

import pandas as pd
import streamlit as st
import plotly.express as px

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif")
)

def render_local_shap_waterfall(df_contrib: pd.DataFrame):
    """Plot interactive horizontal bar chart representing local SHAP feature contributions."""
    top_10 = df_contrib.head(10).copy()
    top_10['Impact'] = top_10['SHAP Value'].apply(lambda x: 'Increases Churn' if x > 0 else 'Reduces Churn')
    
    fig = px.bar(
        top_10,
        x='SHAP Value',
        y='Feature',
        color='Impact',
        orientation='h',
        title="Top 10 Feature Contributions (Local SHAP Analysis)",
        color_discrete_map={'Increases Churn': '#EF4444', 'Reduces Churn': '#10B981'}
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), **DARK_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

def render_global_shap_importance(df_contrib: pd.DataFrame):
    """Plot global feature importance descending chart."""
    top_15 = df_contrib.head(15)
    fig = px.bar(
        top_15,
        x='Abs SHAP',
        y='Feature',
        orientation='h',
        title="Global Feature Importance (SHAP Absolute Magnitude)",
        color='Abs SHAP',
        color_continuous_scale="Purples"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), **DARK_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
