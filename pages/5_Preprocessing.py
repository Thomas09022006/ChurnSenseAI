"""
Prompt 5: Data Preprocessing & Feature Engineering
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ui import apply_custom_theme, render_stepper, render_kpi, render_nav_buttons
from modules.preprocessing import run_preprocessing_pipeline
from modules.feature_engineering import render_feature_engineering_summary
from modules.encoding import render_encoding_summary
from modules.scaling import render_scaling_summary
from modules.split import render_split_summary

st.set_page_config(
    page_title="Data Preprocessing - ChurnSenseAI",
    page_icon="⚙️",
    layout="wide"
)

apply_custom_theme()
render_stepper(4)

st.title("⚙️ Data Preprocessing & Feature Engineering")
st.markdown("Automated data cleaning, feature engineering, categorical encoding, scaling, and train-test split.")

df = st.session_state.get('df', None)

if df is None:
    st.warning("⚠️ No dataset available. Please upload a dataset first.")
    if st.button("⬅️ Go to Upload Page"):
        st.switch_page("pages/2_Upload.py")
    st.stop()

st.markdown("---")

# Execute or fetch preprocessing results
if 'preprocessing_results' not in st.session_state:
    with st.spinner("⏳ Executing automated data cleaning, feature engineering, encoding & scaling..."):
        results = run_preprocessing_pipeline(df)
else:
    results = st.session_state['preprocessing_results']

cleaned_df = results["cleaned_df"]
fe_df = results["fe_df"]
X_proc = results["X_processed"]
y_proc = results["y_processed"]
X_train = results["X_train"]
X_test = results["X_test"]
y_train = results["y_train"]
y_test = results["y_test"]

# Dataset Status KPIs
st.markdown("### 📊 Preprocessing Status Summary")
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_kpi("Original Rows", f"{len(df):,}", "#6366F1")
with k2:
    render_kpi("Cleaned Rows", f"{len(cleaned_df):,}", "#10B981")
with k3:
    render_kpi("Encoded Features", f"{X_proc.shape[1]}", "#38BDF8")
with k4:
    render_kpi("Train Samples", f"{len(X_train):,}", "#F59E0B")
with k5:
    render_kpi("Test Samples", f"{len(X_test):,}", "#EC4899")

st.markdown("---")

# Data Cleaning Steps
st.markdown("### 🧹 Automated Data Cleaning Steps")
c_clean = results["cleaning_summary"]
st.success(f"✔ Removed **{c_clean['dups_removed']}** duplicate rows.")
st.success(f"✔ Converted `TotalCharges` to numeric float and filled **{c_clean['missing_totalcharges_filled']}** missing values using median imputation.")
st.success("✔ Stripped unwanted leading/trailing whitespace from categorical values.")

st.markdown("---")

# Feature Engineering
render_feature_engineering_summary(fe_df)

st.markdown("---")

# Encoding & Scaling
col_enc, col_scl = st.columns(2)
with col_enc:
    render_encoding_summary(results["cat_cols"], X_proc.shape[1])
with col_scl:
    render_scaling_summary(X_proc)

st.markdown("---")

# Train Test Split
render_split_summary(X_train, X_test, y_train, y_test)

st.markdown("---")

# Processed Data Preview
st.markdown("### 🔍 Processed Feature Matrix Preview")
st.dataframe(X_proc.head(10), use_container_width=True)

st.markdown("---")

# Export Processed Dataset
st.markdown("### 📥 Export Processed Dataset")
proc_csv = pd.concat([X_proc, y_proc.rename("Churn")], axis=1).to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Processed CSV", data=proc_csv, file_name="processed_churn_dataset.csv", mime="text/csv")

# Navigation
render_nav_buttons(
    prev_page="pages/4_EDA.py",
    next_page="pages/6_Model_Training.py",
    prev_label="EDA Dashboard",
    next_label="Model Training"
)
