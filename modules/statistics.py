"""
Dataset Statistics logic for Prompt 3.
"""

import pandas as pd
import streamlit as st
import plotly.express as px

def render_datatype_analysis(df: pd.DataFrame):
    """Render pie chart and summary table for dataset column data types."""
    dtypes_counts = df.dtypes.astype(str).value_counts().reset_index()
    dtypes_counts.columns = ['Data Type', 'Count']
    
    fig = px.pie(
        dtypes_counts, 
        names='Data Type', 
        values='Count',
        title="Column Data Types Breakdown",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True)

def render_missing_value_analysis(df: pd.DataFrame):
    """Render missing value bar chart for columns with missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].reset_index()
    
    if len(missing) == 0:
        st.info("✔ No missing values found in any columns!")
        return
        
    missing.columns = ['Column', 'Missing Count']
    missing['Missing %'] = (missing['Missing Count'] / len(df)) * 100
    
    fig = px.bar(
        missing,
        x='Column',
        y='Missing Count',
        hover_data=['Missing %'],
        title="Missing Values by Column",
        color='Missing Count',
        color_continuous_scale='Reds'
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True)
