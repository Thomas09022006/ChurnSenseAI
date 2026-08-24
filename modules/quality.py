"""
Data Quality Module for Prompt 3.
"""

import pandas as pd
import streamlit as st
import json

def export_summary_data(df: pd.DataFrame, quality_info: dict, insights: list) -> tuple:
    """Generate downloadable CSV and JSON summaries of dataset statistics."""
    summary_dict = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": list(df.columns),
        "quality_score": quality_info["score"],
        "quality_rating": quality_info["rating"],
        "insights": insights
    }
    
    json_str = json.dumps(summary_dict, indent=4)
    
    col_summary_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.values.astype(str),
        "Null_Count": df.isnull().sum().values,
        "Unique_Count": [df[c].nunique() for c in df.columns]
    })
    csv_str = col_summary_df.to_csv(index=False)
    
    return csv_str, json_str
