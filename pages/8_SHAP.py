"""
Prompt 8: SHAP Explainability & Final Polish
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ui import apply_custom_theme, render_stepper, render_kpi, render_nav_buttons
from modules.model_loader import load_trained_best_model
from utils.shap_helpers import compute_local_feature_contributions, generate_shap_rule_explanation
from modules.shap_analysis import render_local_shap_waterfall, render_global_shap_importance
from modules.explain import render_contributor_cards
from modules.report_generator import generate_shap_report_files

st.set_page_config(
    page_title="SHAP Explainability - ChurnSenseAI",
    page_icon="💡",
    layout="wide"
)

apply_custom_theme()
render_stepper(7)

st.title("💡 SHAP Explainability & Model Interpretability")
st.markdown("Deconstruct machine learning predictions using Explainable AI (SHAP) feature attributions.")

pred_res = st.session_state.get('last_prediction', None)
model_bundle = load_trained_best_model()

if model_bundle is None or pred_res is None:
    st.warning("⚠️ Prediction data not found. Please execute a Customer Churn Prediction first.")
    if st.button("⬅️ Go to Prediction Page"):
        st.switch_page("pages/7_Prediction.py")
    st.stop()

st.markdown("---")

# Prediction Summary Card
st.markdown("### 🔮 Subject Customer Prediction")
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi("Prediction", pred_res["prediction_label"], pred_res["color"])
with k2:
    render_kpi("Probability", f"{pred_res['churn_prob']*100:.1f}%", "#38BDF8")
with k3:
    render_kpi("Risk Level", pred_res["risk_level"], pred_res["color"])
with k4:
    render_kpi("Confidence", pred_res["confidence"], "#F59E0B")

st.markdown("---")

# SHAP Overview Expander
with st.expander("ℹ️ How SHAP (SHapley Additive exPlanations) Works"):
    st.markdown("""
        SHAP measures how much each customer attribute pushes the model output away from the base average prediction.
        - **Positive SHAP Values (+)** increase customer churn likelihood.
        - **Negative SHAP Values (-)** decrease churn likelihood (promote retention).
    """)

st.markdown("---")

# Compute SHAP Values
model = model_bundle["model"]
X_cust = pred_res["X_cust"]
feature_names = model_bundle["feature_names"]

with st.spinner("⏳ Calculating SHAP feature attribution values..."):
    df_contrib = compute_local_feature_contributions(model, X_cust, feature_names)

shap_info = generate_shap_rule_explanation(df_contrib, pred_res["inputs"])

# Contributor Cards & Narrative
render_contributor_cards(shap_info["pos_bullets"], shap_info["neg_bullets"], shap_info["narrative"])

st.markdown("---")

# Local & Global SHAP Charts
col_w, col_g = st.columns([1, 1])
with col_w:
    render_local_shap_waterfall(df_contrib)
with col_g:
    render_global_shap_importance(df_contrib)

st.markdown("---")

# Download Reports
st.markdown("### 📥 Download SHAP Explanation Report")
csv_bytes, json_bytes = generate_shap_report_files(df_contrib, shap_info["narrative"], pred_res)
cd1, cd2 = st.columns(2)
with cd1:
    st.download_button("📥 SHAP Feature Attributions (CSV)", data=csv_bytes, file_name="shap_feature_attributions.csv", mime="text/csv")
with cd2:
    st.download_button("📥 Complete XAI Explanation (JSON)", data=json_bytes, file_name="shap_explanation_report.json", mime="application/json")

st.markdown("---")

# Final Project Summary KPI Cards
st.markdown("### 🎉 Project Completion Summary")
s1, s2, s3, s4 = st.columns(4)
with s1:
    render_kpi("Models Trained", "4 Algorithms", "#6366F1")
with s2:
    render_kpi("Best Accuracy", f"{model_bundle.get('accuracy', 0)*100:.2f}%", "#10B981")
with s3:
    render_kpi("Prediction Engine", "Live Inference", "#38BDF8")
with s4:
    render_kpi("Status", "Deployment Ready", "#F59E0B")

st.success("🎉 Customer Churn Prediction Project Successfully Completed!")

# Navigation
st.markdown("---")
if st.button("🏁 Finish Project & Return to Home", type="primary", use_container_width=True):
    st.switch_page("pages/1_Home.py")
