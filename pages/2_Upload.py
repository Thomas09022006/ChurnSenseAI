"""
Prompt 2: Dataset Upload & Validation
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ui import apply_custom_theme, render_stepper, render_kpi, render_nav_buttons
from utils.file_helpers import load_dataset_from_bytes, calculate_memory_usage, get_dataset_info
from utils.validation_helpers import validate_dataset
from modules.upload import process_uploaded_file
from modules.validation import render_validation_report, render_target_distribution_chart

st.set_page_config(
    page_title="Upload Dataset - ChurnSenseAI",
    page_icon="📂",
    layout="wide"
)

apply_custom_theme()
render_stepper(1)

st.title("📂 Upload Dataset")
st.markdown("Upload your customer churn dataset (CSV or Excel) to begin the machine learning workflow.")

st.markdown("---")

# File Upload Section
col_u1, col_u2 = st.columns([3, 1])

with col_u1:
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file", 
        type=["csv", "xlsx", "xls"],
        help="Upload IBM Telco Customer Churn dataset or custom customer CSV/Excel file up to 100MB."
    )

with col_u2:
    st.markdown("**Quick Test**")
    if st.button("📥 Use Sample Dataset", use_container_width=True):
        sample_path = os.path.join(os.getcwd(), "assets", "sample_churn_data.csv")
        if os.path.exists(sample_path):
            sample_df = pd.read_csv(sample_path)
            st.session_state['df'] = sample_df
            st.session_state['dataset_name'] = "sample_churn_data.csv"
            st.session_state['upload_time'] = "Sample Dataset"
            val_rep = validate_dataset(sample_df)
            st.session_state['validation_status'] = val_rep['is_overall_valid']
            st.session_state['validation_report'] = val_rep
            st.success("✔ Sample dataset loaded successfully!")
            st.rerun()

if uploaded_file is not None:
    try:
        df = process_uploaded_file(uploaded_file)
        st.success(f"✔ File '{uploaded_file.name}' successfully uploaded.")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

# Read dataset from session state
df = st.session_state.get('df', None)
val_report = st.session_state.get('validation_report', {})

if df is not None:
    st.markdown("---")
    
    # Validation Report Section
    render_validation_report(df, val_report)
    
    st.markdown("---")
    
    # Dataset KPIs
    st.markdown("### 📊 Dataset Information")
    info = get_dataset_info(df)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi("Rows", f"{info.get('rows', 0):,}", "#6366F1")
    with k2:
        render_kpi("Columns", f"{info.get('cols', 0)}", "#10B981")
    with k3:
        render_kpi("Memory", f"{info.get('memory', '0 KB')}", "#38BDF8")
    with k4:
        render_kpi("Target Col", "Churn", "#F59E0B")
    with k5:
        render_kpi("Missing Cells", f"{info.get('total_missing', 0)}", "#EF4444")
        
    st.markdown("---")
    
    # Dataset Preview & Target Distribution
    c_left, c_right = st.columns([3, 2])
    
    with c_left:
        st.markdown("### 🔍 Dataset Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)
        
    with c_right:
        st.markdown("### 🎯 Target Distribution")
        render_target_distribution_chart(df, "Churn")

    st.markdown("---")
    
    # Column List Table
    st.markdown("### 📋 Column Breakdown")
    col_df = pd.DataFrame({
        "Column Name": df.columns,
        "Detected Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Missing %": ((df.isnull().sum() / len(df)) * 100).round(2).values
    })
    st.dataframe(col_df, use_container_width=True)

else:
    st.info("👆 Please upload a CSV/Excel file or click 'Use Sample Dataset' to proceed.")

# Navigation
nav_enabled = st.session_state.get('validation_status', False) or (df is not None)
render_nav_buttons(
    prev_page="pages/1_Home.py",
    next_page="pages/3_Overview.py",
    prev_label="Home",
    next_label="Dataset Overview",
    next_disabled=not nav_enabled
)
