"""
Prediction UI & Gauge Chart Module for Prompt 7.
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

def render_probability_gauge(prob: float, color: str):
    """Plot interactive Plotly gauge chart for prediction probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': "%"},
        title={'text': "Churn Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                {'range': [30, 60], 'color': "rgba(245, 158, 11, 0.2)"},
                {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.2)"}
            ]
        }
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        height=260,
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True)
