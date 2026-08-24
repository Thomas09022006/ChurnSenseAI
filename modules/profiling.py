"""
Profiling business logic module for Prompt 3.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.file_helpers import calculate_memory_usage
from utils.profiling_helpers import calculate_quality_score, generate_insights, generate_recommendations

def render_column_explorer(df: pd.DataFrame):
    """Display expandable cards for every column in the dataset."""
    st.markdown("### 🔍 Column Explorer")
    for col in df.columns:
        with st.expander(f"📌 Column: **{col}** ({df[col].dtype})"):
            col1, col2 = st.columns(2)
            missing_cnt = df[col].isnull().sum()
            missing_pct = (missing_cnt / len(df)) * 100
            unique_cnt = df[col].nunique()
            mem_col = calculate_memory_usage(df[[col]])
            
            with col1:
                st.write(f"- **Data Type**: `{df[col].dtype}`")
                st.write(f"- **Missing Values**: {missing_cnt} ({missing_pct:.2f}%)")
                st.write(f"- **Unique Values**: {unique_cnt}")
                st.write(f"- **Memory Usage**: {mem_col}")
                
            with col2:
                if pd.api.types.is_numeric_dtype(df[col]):
                    st.write(f"- **Min**: {df[col].min()}")
                    st.write(f"- **Max**: {df[col].max()}")
                    st.write(f"- **Mean**: {df[col].mean():.2f}")
                    st.write(f"- **Median**: {df[col].median():.2f}")
                    st.write(f"- **Std Dev**: {df[col].std():.2f}")
                else:
                    top_vals = df[col].value_counts().head(3).to_dict()
                    st.write(f"- **Top Values**: {top_vals}")

def render_quality_gauge(score_info: dict):
    """Render a gauge plot showing data quality score out of 100."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_info["score"],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Data Quality Score: {score_info['rating']}"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': score_info["color"]},
            'steps': [
                {'range': [0, 60], 'color': "rgba(239, 68, 68, 0.2)"},
                {'range': [60, 80], 'color': "rgba(245, 158, 11, 0.2)"},
                {'range': [80, 100], 'color': "rgba(16, 185, 129, 0.2)"}
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
