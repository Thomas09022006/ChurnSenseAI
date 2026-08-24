"""
Main Entry Point for ChurnSenseAI Streamlit Application.
"""

import streamlit as st

st.set_page_config(
    page_title="ChurnSenseAI - Customer Churn Prediction",
    page_icon="🤖",
    layout="wide"
)

# Redirect to Home page
st.switch_page("pages/1_Home.py")
