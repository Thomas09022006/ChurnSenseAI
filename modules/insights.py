"""
Business Insights and Export Module for Prompt 4.
"""

import pandas as pd
import streamlit as st
import json
from utils.eda_helpers import generate_business_insights

def render_business_insights_section(df: pd.DataFrame):
    """Render rule-based business insights cards."""
    st.markdown("### 💡 Business Insights")
    insights = generate_business_insights(df)
    
    for ins in insights:
        st.markdown(f"""
            <div class="custom-card" style="margin-bottom: 12px; padding: 16px;">
                {ins}
            </div>
        """, unsafe_allow_html=True)
        
    return insights

def export_eda_reports(df: pd.DataFrame, insights: list) -> tuple:
    """Export EDA report to CSV summary and JSON format."""
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    stats_df = df[num_cols].describe().T.reset_index().rename(columns={'index': 'Feature'})
    csv_report = stats_df.to_csv(index=False)
    
    json_report = json.dumps({
        "dataset_rows": len(df),
        "dataset_cols": len(df.columns),
        "insights": insights,
        "summary_statistics": stats_df.to_dict(orient='records')
    }, indent=4)
    
    return csv_report, json_report
