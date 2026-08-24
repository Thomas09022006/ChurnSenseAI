"""
Prompt 1: Project Setup & Home Dashboard
"""

import streamlit as st
import os
import sys

# Ensure root directory in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ui import apply_custom_theme, render_stepper, render_kpi, render_nav_buttons

st.set_page_config(
    page_title="Customer Churn Prediction - Home",
    page_icon="🏠",
    layout="wide"
)

apply_custom_theme()

# Step Progress
render_stepper(0)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=64)
    st.title("ChurnSenseAI")
    st.markdown("**Navigation**")
    st.markdown("""
        - 🏠 Home
        - 📂 Upload Dataset
        - 📋 Dataset Overview
        - 📊 Exploratory Data Analysis
        - ⚙️ Data Preprocessing
        - 🤖 Model Training
        - 📈 Model Evaluation
        - 🔮 Prediction
        - 💡 SHAP Explainability
    """)
    st.caption("v1.0.0 | Portfolio Prototype")

# Hero Section
st.markdown("""
    <div class="custom-card" style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(15, 23, 42, 0.9));">
        <h1 style="font-size: 2.8rem; font-weight: 800; color: #FAFAFA; margin-bottom: 10px;">Customer Churn Prediction</h1>
        <p style="font-size: 1.2rem; color: #818CF8; margin-bottom: 25px;">Machine Learning-Based Customer Churn Prediction Dashboard</p>
        <p style="max-width: 800px; margin: 0 auto 30px auto; color: #94A3B8; font-size: 1rem; line-height: 1.6;">
            A portfolio-quality machine learning prototype demonstrating an end-to-end data science pipeline. 
            Upload datasets, profile customer behavior, train classification algorithms, compare models, and explain predictions using SHAP Explainable AI.
        </p>
    </div>
""", unsafe_allow_html=True)

col_hero_btn1, col_hero_btn2, col_hero_btn3 = st.columns([1, 1, 1])
with col_hero_btn2:
    if st.button("🚀 Get Started - Upload Dataset", type="primary", use_container_width=True):
        st.switch_page("pages/2_Upload.py")

st.markdown("---")

# Project Metrics KPIs
st.markdown("### 📊 Project Metrics")
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi("Algorithms", "4 Models", "#6366F1")
with k2:
    render_kpi("Classification", "Binary", "#10B981")
with k3:
    render_kpi("Explainability", "SHAP XAI", "#F59E0B")
with k4:
    render_kpi("Deployment", "Streamlit", "#38BDF8")

st.markdown("---")

# ML Workflow Timeline
st.markdown("### 🔄 Machine Learning Workflow")
st.markdown("""
    <div class="stepper-container" style="justify-content: space-around; padding: 20px;">
        <div style="text-align:center;">📂<br><b>1. Upload</b></div> ›
        <div style="text-align:center;">📋<br><b>2. Profiling</b></div> ›
        <div style="text-align:center;">📊<br><b>3. EDA</b></div> ›
        <div style="text-align:center;">⚙️<br><b>4. Preprocess</b></div> ›
        <div style="text-align:center;">🤖<br><b>5. Training</b></div> ›
        <div style="text-align:center;">📈<br><b>6. Evaluation</b></div> ›
        <div style="text-align:center;">🔮<br><b>7. Prediction</b></div> ›
        <div style="text-align:center;">💡<br><b>8. SHAP</b></div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Key Features
st.markdown("### ✨ Platform Features")
f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("""
        <div class="custom-card">
            <h4>📂 Dataset Upload & Validation</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">Upload CSV or Excel customer datasets up to 100MB with automated column validation.</p>
        </div>
        <div class="custom-card">
            <h4>📊 Exploratory Data Analysis</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">Interactive Plotly histograms, correlation heatmaps, box plots, and automated business insights.</p>
        </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
        <div class="custom-card">
            <h4>⚙️ Preprocessing & Feature Engineering</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">Automated missing value imputation, tenure binning, charge categorization, encoding, and scaling.</p>
        </div>
        <div class="custom-card">
            <h4>🤖 ML Model Training</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">Train Logistic Regression, Decision Tree, Random Forest, and XGBoost with ROC curve comparisons.</p>
        </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
        <div class="custom-card">
            <h4>🔮 Customer Prediction</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">Interactive form to infer churn probability, risk meters, and confidence levels for individual customers.</p>
        </div>
        <div class="custom-card">
            <h4>💡 SHAP Explainability</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">Local waterfall plots and global feature contributions powered by SHAP XAI.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Machine Learning Models Cards
st.markdown("### 🤖 Supported Classification Algorithms")
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
        <div class="custom-card">
            <h4 style="color: #818CF8;">Logistic Regression</h4>
            <p style="font-size:0.85rem; color:#94A3B8;">Linear classifier serving as a statistical baseline model.</p>
            <span class="badge badge-info">Baseline</span>
        </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
        <div class="custom-card">
            <h4 style="color: #34D399;">Decision Tree</h4>
            <p style="font-size:0.85rem; color:#94A3B8;">Tree-based rules providing high interpretability and feature splits.</p>
            <span class="badge badge-success">Interpretable</span>
        </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
        <div class="custom-card">
            <h4 style="color: #FBBF24;">Random Forest</h4>
            <p style="font-size:0.85rem; color:#94A3B8;">Ensemble of decision trees mitigating overfitting variance.</p>
            <span class="badge badge-warning">Ensemble</span>
        </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
        <div class="custom-card">
            <h4 style="color: #F87171;">XGBoost</h4>
            <p style="font-size:0.85rem; color:#94A3B8;">Gradient boosted trees optimized for top predictive accuracy.</p>
            <span class="badge badge-danger">High Performance</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Recommended Dataset Info
st.markdown("### 📋 Recommended Dataset Specs")
st.markdown("""
    <div class="custom-card">
        <b>IBM Telco Customer Churn Dataset</b><br>
        <span style="color:#94A3B8;">7,043 Records | 21 Features | Binary Target: <code>Churn</code> ('Yes' / 'No')</span>
    </div>
""", unsafe_allow_html=True)

# Navigation
render_nav_buttons(
    prev_page=None,
    next_page="pages/2_Upload.py",
    next_label="Upload Dataset"
)
