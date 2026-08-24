"""
File loading and preview helper functions for ChurnSenseAI.
"""

import pandas as pd
import streamlit as st
import io

@st.cache_data
def load_dataset_from_bytes(file_bytes, file_name: str) -> pd.DataFrame:
    """Load dataset from bytes (CSV or XLSX)."""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel file.")
        return df
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

def calculate_memory_usage(df: pd.DataFrame) -> str:
    """Calculate and format memory usage of dataframe."""
    if df is None:
        return "0 KB"
    memory_bytes = df.memory_usage(deep=True).sum()
    if memory_bytes < 1024 * 1024:
        return f"{memory_bytes / 1024:.2f} KB"
    else:
        return f"{memory_bytes / (1024 * 1024):.2f} MB"

def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return dictionary of dataset summary info."""
    if df is None:
        return {}
    
    num_rows, num_cols = df.shape
    num_num = len(df.select_dtypes(include=['number']).columns)
    num_cat = len(df.select_dtypes(include=['object', 'category']).columns)
    total_missing = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    return {
        "rows": num_rows,
        "cols": num_cols,
        "num_features": num_num,
        "cat_features": num_cat,
        "total_missing": total_missing,
        "duplicate_rows": duplicate_rows,
        "memory": calculate_memory_usage(df)
    }
