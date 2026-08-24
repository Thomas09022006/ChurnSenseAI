"""
Prompt 4: Exploratory Data Analysis (EDA)
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ui import apply_custom_theme, render_stepper, render_kpi, render_nav_buttons
from modules.charts import generate_histogram, generate_boxplot, generate_bar_chart, generate_correlation_heatmap
from modules.outliers import render_outlier_analysis
from modules.insights import render_business_insights_section, export_eda_reports

st.set_page_config(
    page_title="EDA - ChurnSenseAI",
    page_icon="📊",
    layout="wide"
)

apply_custom_theme()
render_stepper(3)

st.title("📊 Exploratory Data Analysis (EDA)")
st.markdown("Explore customer behavior, distributions, correlations, and churn patterns.")

df = st.session_state.get('df', None)

if df is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset first.")
    if st.button("⬅️ Go to Upload Page"):
        st.switch_page("pages/2_Upload.py")
    st.stop()

st.markdown("---")

# EDA KPI Summary
st.markdown("### 📊 Dataset Feature Summary")
num_cols = df.select_dtypes(include=['number']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    render_kpi("Numerical", f"{len(num_cols)}", "#6366F1")
with k2:
    render_kpi("Categorical", f"{len(cat_cols)}", "#10B981")
with k3:
    render_kpi("Missing", f"{df.isnull().sum().sum()}", "#F59E0B")
with k4:
    render_kpi("Classes", "2 (Binary)", "#38BDF8")
with k5:
    mc = pd.to_numeric(df['MonthlyCharges'], errors='coerce').mean() if 'MonthlyCharges' in df else 0
    render_kpi("Avg Monthly", f"${mc:.2f}", "#EC4899")
with k6:
    tn = df['tenure'].mean() if 'tenure' in df else 0
    render_kpi("Avg Tenure", f"{tn:.1f} m", "#8B5CF6")

st.markdown("---")

# Feature Selectors
col_s1, col_s2 = st.columns(2)
with col_s1:
    selected_num = st.selectbox("Select Numerical Feature", [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges'] if c in df.columns] or num_cols)
with col_s2:
    selected_cat = st.selectbox("Select Categorical Feature", [c for c in ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'PaperlessBilling'] if c in df.columns] or cat_cols)

st.markdown("---")

# Numerical & Categorical Analysis
c_n, c_c = st.columns(2)
with c_n:
    st.markdown(f"### 🔢 Numerical Analysis: `{selected_num}`")
    fig_hist = generate_histogram(df, selected_num, hue="Churn" if "Churn" in df else None)
    st.plotly_chart(fig_hist, use_container_width=True)

with c_c:
    st.markdown(f"### 🏷️ Categorical Analysis: `{selected_cat}`")
    fig_bar = generate_bar_chart(df, selected_cat, hue="Churn" if "Churn" in df else None)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# Correlation Matrix Heatmap & Target Breakdown
col_corr, col_targ = st.columns([1, 1])
with col_corr:
    st.markdown("### 🌡️ Correlation Heatmap")
    fig_corr = generate_correlation_heatmap(df)
    st.plotly_chart(fig_corr, use_container_width=True)
    
with col_targ:
    st.markdown("### 🎯 Target Churn vs Contract Type")
    if "Contract" in df.columns and "Churn" in df.columns:
        fig_contract = generate_bar_chart(df, "Contract", hue="Churn")
        st.plotly_chart(fig_contract, use_container_width=True)
    else:
        st.info("Contract feature not present in dataset.")

st.markdown("---")

# Outlier Detection
render_outlier_analysis(df)

st.markdown("---")

# Business Insights Section
insights = render_business_insights_section(df)

st.markdown("---")

# Export Report
st.markdown("### 📥 Export EDA Report")
csv_rep, json_rep = export_eda_reports(df, insights)
cd1, cd2 = st.columns(2)
with cd1:
    st.download_button("📥 Download Summary Statistics (CSV)", data=csv_rep, file_name="eda_summary_statistics.csv", mime="text/csv")
with cd2:
    st.download_button("📥 Download Business Insights (JSON)", data=json_rep, file_name="eda_insights.json", mime="application/json")

# Navigation
render_nav_buttons(
    prev_page="pages/3_Overview.py",
    next_page="pages/5_Preprocessing.py",
    prev_label="Dataset Overview",
    next_label="Data Preprocessing"
)
