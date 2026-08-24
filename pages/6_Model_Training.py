"""
Prompt 6: Model Training & Model Comparison
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ui import apply_custom_theme, render_stepper, render_kpi, render_nav_buttons
from modules.training import train_selected_models
from modules.comparison import render_comparison_table, render_feature_importance_charts
from modules.evaluation import render_roc_curves, render_confusion_matrices
from modules.metrics import generate_metrics_download_files
from modules.save_model import get_saved_model_bytes

st.set_page_config(
    page_title="Model Training - ChurnSenseAI",
    page_icon="🤖",
    layout="wide"
)

apply_custom_theme()
render_stepper(5)

st.title("🤖 Model Training & Model Comparison")
st.markdown("Train multiple classification models, evaluate metrics, compare performance, and export the best model.")

X_train = st.session_state.get('X_train', None)
X_test = st.session_state.get('X_test', None)
y_train = st.session_state.get('y_train', None)
y_test = st.session_state.get('y_test', None)

if X_train is None:
    st.warning("⚠️ Processed dataset not found. Please complete the Data Preprocessing module first.")
    if st.button("⬅️ Go to Data Preprocessing"):
        st.switch_page("pages/5_Preprocessing.py")
    st.stop()

st.markdown("---")

# Preprocessed Dataset Ready KPI
st.markdown("### 📋 Training Dataset Specs")
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi("Training Samples", f"{len(X_train):,}", "#6366F1")
with k2:
    render_kpi("Testing Samples", f"{len(X_test):,}", "#10B981")
with k3:
    render_kpi("Feature Count", f"{X_train.shape[1]}", "#38BDF8")
with k4:
    render_kpi("Target Classes", "2 (Binary)", "#F59E0B")

st.markdown("---")

# Model Selection Checkboxes & Config
col_m_sel, col_m_cfg = st.columns([2, 1])

with col_m_sel:
    st.markdown("### ⚙️ Select Models to Train")
    cb_lr = st.checkbox("Logistic Regression", value=True)
    cb_dt = st.checkbox("Decision Tree", value=True)
    cb_rf = st.checkbox("Random Forest", value=True)
    cb_xgb = st.checkbox("XGBoost", value=True)
    
    selected_models = []
    if cb_lr: selected_models.append("Logistic Regression")
    if cb_dt: selected_models.append("Decision Tree")
    if cb_rf: selected_models.append("Random Forest")
    if cb_xgb: selected_models.append("XGBoost")

with col_m_cfg:
    st.markdown("### ⚙️ Training Parameters")
    st.info("""
        - **Train / Test Ratio**: 80% / 20%
        - **Random Seed**: 42
        - **CV Strategy**: 5-Fold Stratified
        - **Primary Metric**: Accuracy / ROC-AUC
    """)

st.markdown("---")

# Train Button
if st.button("🚀 Train Selected Models", type="primary", use_container_width=True):
    if not selected_models:
        st.error("Please select at least one algorithm to train.")
    else:
        results = train_selected_models(selected_models, X_train, y_train, X_test, y_test)
        st.success("✔ Model training and evaluation successfully executed!")

results = st.session_state.get('trained_models', None)

if results is not None:
    st.markdown("---")
    
    # Best Model Banner Card
    best_name = st.session_state.get('best_model_name', '')
    best_res = results.get(best_name, {})
    
    st.markdown(f"""
        <div class="custom-card" style="border-left: 4px solid #10B981; background: rgba(16, 185, 129, 0.1);">
            <h3 style="color: #34D399; margin: 0;">🏆 Best Performing Model: {best_name}</h3>
            <p style="margin-top: 8px; color: #FAFAFA;">
                Accuracy: <b>{best_res.get('accuracy', 0)*100:.2f}%</b> | 
                ROC-AUC: <b>{best_res.get('roc_auc', 0):.4f}</b> | 
                Precision: <b>{best_res.get('precision', 0):.4f}</b> | 
                Recall: <b>{best_res.get('recall', 0):.4f}</b>
            </p>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0;">Automatically saved to <code>saved_models/best_model.joblib</code> for real-time customer predictions.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Comparison Table & Charts
    comp_df = render_comparison_table(results)
    
    st.markdown("---")
    
    # ROC Curve & Confusion Matrices
    col_roc, col_cm = st.columns([1, 1])
    with col_roc:
        render_roc_curves(results)
    with col_cm:
        render_confusion_matrices(results)
        
    st.markdown("---")
    
    # Feature Importance Charts
    render_feature_importance_charts(results)
    
    st.markdown("---")
    
    # Downloads Section
    st.markdown("### 📥 Download Trained Artifacts & Metrics")
    csv_bytes, json_bytes = generate_metrics_download_files(results)
    joblib_bytes = get_saved_model_bytes()
    
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("📥 Performance Metrics (CSV)", data=csv_bytes, file_name="model_performance_metrics.csv", mime="text/csv")
    with d2:
        st.download_button("📥 Performance Metrics (JSON)", data=json_bytes, file_name="model_performance_metrics.json", mime="application/json")
    with d3:
        if joblib_bytes:
            st.download_button("📥 Trained Model Bundle (.joblib)", data=joblib_bytes, file_name="best_model.joblib", mime="application/octet-stream")

else:
    st.info("👆 Click 'Train Selected Models' above to start algorithm training and comparison.")

# Navigation
render_nav_buttons(
    prev_page="pages/5_Preprocessing.py",
    next_page="pages/7_Prediction.py",
    prev_label="Data Preprocessing",
    next_label="Customer Prediction",
    next_disabled=results is None
)
