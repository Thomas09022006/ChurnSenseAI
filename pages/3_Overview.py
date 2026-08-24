"""
Prompt 3: Dataset Overview & Data Profiling
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ui import apply_custom_theme, render_stepper, render_kpi, render_nav_buttons
from utils.profiling_helpers import calculate_quality_score, generate_insights, generate_recommendations
from modules.profiling import render_column_explorer, render_quality_gauge
from modules.statistics import render_datatype_analysis, render_missing_value_analysis
from modules.quality import export_summary_data

st.set_page_config(
    page_title="Dataset Overview - ChurnSenseAI",
    page_icon="📋",
    layout="wide"
)

apply_custom_theme()
render_stepper(2)

st.title("📋 Dataset Overview & Data Profiling")
st.markdown("Automatically analyze and profile your customer dataset structure, statistics, and quality.")

df = st.session_state.get('df', None)

if df is None:
    st.warning("⚠️ No dataset uploaded yet. Please go back to the Upload page to load a dataset.")
    if st.button("⬅️ Go to Upload Dataset"):
        st.switch_page("pages/2_Upload.py")
    st.stop()

st.markdown("---")

# Data Quality Score Gauge & Summary
q_col1, q_col2 = st.columns([1, 2])
score_info = calculate_quality_score(df)
insights = generate_insights(df)
recs = generate_recommendations(df)

with q_col1:
    st.markdown("### 🏆 Data Quality Score")
    render_quality_gauge(score_info)
    
with q_col2:
    st.markdown("### 📊 Dataset KPI Summary")
    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi("Total Rows", f"{len(df):,}", "#6366F1")
        render_kpi("Missing Cells", f"{score_info['missing_pct']}%", "#F59E0B")
    with k2:
        render_kpi("Total Columns", f"{len(df.columns)}", "#10B981")
        render_kpi("Duplicates", f"{score_info['dup_pct']}%", "#EF4444")
    with k3:
        render_kpi("Target Column", "Churn", "#38BDF8")
        render_kpi("Quality Rating", score_info['rating'], score_info['color'])

st.markdown("---")

# Data Preview Tabs
st.markdown("### 🔍 Dataset Preview")
tab_head, tab_tail, tab_sample = st.tabs(["First 10 Rows", "Last 10 Rows", "Random Sample (10)"])
with tab_head:
    st.dataframe(df.head(10), use_container_width=True)
with tab_tail:
    st.dataframe(df.tail(10), use_container_width=True)
with tab_sample:
    st.dataframe(df.sample(min(10, len(df))), use_container_width=True)

st.markdown("---")

# Automatic Quick Insights & Recommendations
col_ins, col_rec = st.columns(2)
with col_ins:
    st.markdown("### 💡 Quick Data Science Insights")
    for ins in insights:
        st.markdown(f"- {ins}")
        
with col_rec:
    st.markdown("### 🛠️ Preprocessing Recommendations")
    for rec in recs:
        st.markdown(f"- {rec}")

st.markdown("---")

# Data Type Breakdown & Missing Value Analysis
col_dt, col_mv = st.columns(2)
with col_dt:
    render_datatype_analysis(df)
with col_mv:
    render_missing_value_analysis(df)

st.markdown("---")

# Column Explorer
render_column_explorer(df)

st.markdown("---")

# Summary Download
st.markdown("### 📥 Export Profiling Summary")
csv_bytes, json_str = export_summary_data(df, score_info, insights)
c_d1, c_d2 = st.columns(2)
with c_d1:
    st.download_button("📥 Download Summary (CSV)", data=csv_bytes, file_name="dataset_summary.csv", mime="text/csv")
with c_d2:
    st.download_button("📥 Download Insights (JSON)", data=json_str, file_name="dataset_insights.json", mime="application/json")

# Navigation
render_nav_buttons(
    prev_page="pages/2_Upload.py",
    next_page="pages/4_EDA.py",
    prev_label="Upload Dataset",
    next_label="EDA Dashboard"
)
