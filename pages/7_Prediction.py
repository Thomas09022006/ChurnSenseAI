"""
Prompt 7: Customer Churn Prediction
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ui import apply_custom_theme, render_stepper, render_kpi, render_nav_buttons
from modules.model_loader import load_trained_best_model
from modules.inference import run_customer_inference
from modules.prediction import render_probability_gauge
from modules.report import generate_prediction_report_files

st.set_page_config(
    page_title="Prediction - ChurnSenseAI",
    page_icon="🔮",
    layout="wide"
)

apply_custom_theme()
render_stepper(6)

st.title("🔮 Customer Churn Prediction")
st.markdown("Predict individual customer churn likelihood using your saved machine learning model.")

# Load saved best model
model_bundle = load_trained_best_model()

if model_bundle is None:
    st.warning("⚠️ Saved model bundle not found. Please complete the Model Training step first.")
    if st.button("⬅️ Go to Model Training"):
        st.switch_page("pages/6_Model_Training.py")
    st.stop()

st.markdown("---")

# Best Model Card
st.markdown("### 🏆 Active Inference Model")
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi("Model Name", model_bundle.get("best_model_name", "Best Model"), "#6366F1")
with k2:
    render_kpi("Accuracy", f"{model_bundle.get('accuracy', 0)*100:.2f}%", "#10B981")
with k3:
    render_kpi("ROC-AUC", f"{model_bundle.get('roc_auc', 0):.4f}", "#38BDF8")
with k4:
    render_kpi("Features", f"{len(model_bundle.get('feature_names', []))}", "#F59E0B")

st.markdown("---")

# Customer Input Form
st.markdown("### 📝 Enter Customer Profile Data")

with st.form("customer_input_form"):
    col_p, col_s = st.columns(2)
    
    with col_p:
        st.markdown("#### 👤 Personal Information")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        
        st.markdown("#### 📄 Contract & Billing")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    with col_s:
        st.markdown("#### 🌐 Service Subscriptions")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown("#### 📊 Customer Usage & Financials")
    u1, u2, u3 = st.columns(3)
    with u1:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
    with u2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=65.0)
    with u3:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=20000.0, value=780.0)

    submit_btn = st.form_submit_button("🚀 Predict Customer Churn", type="primary", use_container_width=True)

if submit_btn:
    inputs = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    with st.spinner("⏳ Preprocessing customer inputs and generating model inference..."):
        pred_res = run_customer_inference(inputs, model_bundle)
    st.success("✔ Prediction generated successfully!")

pred_res = st.session_state.get('last_prediction', None)

if pred_res is not None:
    st.markdown("---")
    
    # Large Prediction Result Banner
    is_churn = pred_res["pred_class"] == 1
    banner_color = "#EF4444" if is_churn else "#10B981"
    banner_bg = "rgba(239, 68, 68, 0.15)" if is_churn else "rgba(16, 185, 129, 0.15)"
    
    st.markdown(f"""
        <div class="custom-card" style="border-left: 6px solid {banner_color}; background: {banner_bg}; text-align: center;">
            <h2 style="color: {banner_color}; margin: 0;">{pred_res['prediction_label']}</h2>
            <p style="font-size: 1.2rem; color: #FAFAFA; margin-top: 10px;">
                Churn Probability: <b>{pred_res['churn_prob']*100:.1f}%</b> | 
                Risk Level: <span class="badge {pred_res['risk_badge']}">{pred_res['risk_level']}</span> | 
                Confidence: <b>{pred_res['confidence']}</b>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Probability Gauge & Risk Meter
    col_g, col_sum = st.columns([1, 1])
    with col_g:
        st.markdown("### ⏱️ Probability Gauge")
        render_probability_gauge(pred_res['churn_prob'], pred_res['color'])
        
    with col_sum:
        st.markdown("### 📝 Inference Summary")
        for b in pred_res["summary"]:
            st.markdown(f"- {b}")
            
    st.markdown("---")
    
    # Customer Entered Inputs Summary
    st.markdown("### 📋 Entered Customer Profile")
    st.dataframe(pd.DataFrame([pred_res["inputs"]]), use_container_width=True)
    
    st.markdown("---")
    
    # Downloads
    st.markdown("### 📥 Export Prediction Report")
    csv_bytes, json_bytes = generate_prediction_report_files(pred_res)
    cd1, cd2 = st.columns(2)
    with cd1:
        st.download_button("📥 Download Prediction (CSV)", data=csv_bytes, file_name="customer_prediction.csv", mime="text/csv")
    with cd2:
        st.download_button("📥 Download Prediction (JSON)", data=json_bytes, file_name="customer_prediction.json", mime="application/json")

# Navigation
render_nav_buttons(
    prev_page="pages/6_Model_Training.py",
    next_page="pages/8_SHAP.py",
    prev_label="Model Training",
    next_label="SHAP Explainability",
    next_disabled=pred_res is None
)
