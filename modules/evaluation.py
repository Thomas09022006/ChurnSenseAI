"""
Evaluation Plots & Confusion Matrix Module for Prompt 6.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif")
)

def render_roc_curves(results: dict):
    """Plot multi-model ROC Curve comparison."""
    fig = go.Figure()
    
    for model_name, res in results.items():
        fig.add_trace(go.Scatter(
            x=res["fpr"],
            y=res["tpr"],
            mode='lines',
            name=f"{model_name} (AUC = {res['roc_auc']:.4f})"
        ))
        
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        showlegend=False
    ))
    
    fig.update_layout(
        title="ROC Curves Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        **DARK_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

def render_confusion_matrices(results: dict):
    """Render interactive tabbed confusion matrices."""
    st.markdown("### 🔲 Confusion Matrices")
    tabs = st.tabs(list(results.keys()))
    
    for tab, (model_name, res) in zip(tabs, results.items()):
        with tab:
            cm = res["confusion_matrix"]
            fig = px.imshow(
                cm,
                text_auto=True,
                x=['Predicted Stay (0)', 'Predicted Churn (1)'],
                y=['Actual Stay (0)', 'Actual Churn (1)'],
                color_continuous_scale='Blues',
                title=f"Confusion Matrix - {model_name}"
            )
            fig.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
