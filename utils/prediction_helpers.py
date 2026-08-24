"""
Prediction Helper functions for Prompt 7.
"""

import pandas as pd
import numpy as np

def prepare_customer_input_df(inputs: dict) -> pd.DataFrame:
    """Construct single-row raw customer dataframe matching IBM Telco schema."""
    customer_dict = {
        'customerID': 'CUST-PRED-001',
        'gender': inputs['gender'],
        'SeniorCitizen': int(inputs['SeniorCitizen']),
        'Partner': inputs['Partner'],
        'Dependents': inputs['Dependents'],
        'tenure': int(inputs['tenure']),
        'PhoneService': inputs['PhoneService'],
        'MultipleLines': inputs['MultipleLines'],
        'InternetService': inputs['InternetService'],
        'OnlineSecurity': inputs['OnlineSecurity'],
        'OnlineBackup': inputs['OnlineBackup'],
        'DeviceProtection': inputs['DeviceProtection'],
        'TechSupport': inputs['TechSupport'],
        'StreamingTV': inputs['StreamingTV'],
        'StreamingMovies': inputs['StreamingMovies'],
        'Contract': inputs['Contract'],
        'PaperlessBilling': inputs['PaperlessBilling'],
        'PaymentMethod': inputs['PaymentMethod'],
        'MonthlyCharges': float(inputs['MonthlyCharges']),
        'TotalCharges': float(inputs['TotalCharges']),
        'Churn': 'No' # Placeholder
    }
    return pd.DataFrame([customer_dict])

def preprocess_single_customer(df_row: pd.DataFrame, feature_names: list, scaler) -> pd.DataFrame:
    """Preprocess single customer row using saved feature pipeline."""
    proc_df = df_row.copy()
    if 'customerID' in proc_df.columns:
        proc_df = proc_df.drop(columns=['customerID'])
    if 'Churn' in proc_df.columns:
        proc_df = proc_df.drop(columns=['Churn'])
        
    # Feature Engineering
    bins = [-1, 12, 24, 48, 120]
    labels = ['0-12 Months', '13-24 Months', '25-48 Months', '48+ Months']
    proc_df['tenure_group'] = pd.cut(proc_df['tenure'], bins=bins, labels=labels)
    
    mc_bins = [0, 35, 75, 500]
    mc_labels = ['Low', 'Medium', 'High']
    proc_df['charge_category'] = pd.cut(proc_df['MonthlyCharges'], bins=mc_bins, labels=mc_labels)
    
    proc_df["avg_monthly_revenue"] = proc_df["TotalCharges"] / proc_df["tenure"].replace(0, 1)
    
    # One-hot encoding matching training feature space
    cat_cols = proc_df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = proc_df.select_dtypes(include=['number']).columns.tolist()
    
    encoded_df = pd.get_dummies(proc_df, columns=cat_cols, drop_first=True)
    
    # Reindex columns to match training feature_names exactly (fill missing dummies with 0)
    final_df = pd.DataFrame(0, index=[0], columns=feature_names)
    for col in encoded_df.columns:
        if col in final_df.columns:
            final_df[col] = encoded_df[col]
            
    # Apply scaler
    if scaler is not None:
        scaled_cols = [c for c in num_cols if c in final_df.columns]
        if scaled_cols:
            final_df[scaled_cols] = scaler.transform(final_df[scaled_cols])
            
    return final_df

def calculate_confidence_level(prob: float) -> tuple:
    """
    Calculate confidence level & risk level:
    Confidence: High (>90%), Medium (70-90%), Low (<70%)
    Risk Level: Low (0-30%), Medium (31-60%), High (61-100%)
    """
    churn_prob_pct = prob * 100
    
    # Confidence
    if churn_prob_pct >= 90 or churn_prob_pct <= 10:
        confidence = "High"
    elif churn_prob_pct >= 70 or churn_prob_pct <= 30:
        confidence = "Medium"
    else:
        confidence = "Low"
        
    # Risk Level
    if churn_prob_pct <= 30:
        risk_level = "Low Risk"
        risk_badge = "badge-success"
        color = "#10B981"
    elif churn_prob_pct <= 60:
        risk_level = "Medium Risk"
        risk_badge = "badge-warning"
        color = "#F59E0B"
    else:
        risk_level = "High Risk"
        risk_badge = "badge-danger"
        color = "#EF4444"
        
    return confidence, risk_level, risk_badge, color

def generate_prediction_summary(inputs: dict, prob: float, risk_level: str) -> list:
    """Generate rule-based prediction summary without LLM/AI."""
    prob_pct = prob * 100
    summary = []
    
    if prob_pct >= 50:
        summary.append(f"🔴 **High Churn Risk**: Customer has a **{prob_pct:.1f}%** probability of churning.")
    else:
        summary.append(f"🟢 **Customer Retention**: Customer has a **{(100 - prob_pct):.1f}%** probability of remaining active.")
        
    if inputs.get("Contract") == "Month-to-month":
        summary.append("📄 Month-to-month contract is a significant risk factor for customer churn.")
    elif inputs.get("Contract") in ["One year", "Two year"]:
        summary.append("✔ Long-term contract type encourages customer retention.")
        
    if float(inputs.get("MonthlyCharges", 0)) > 75:
        summary.append("💳 Higher monthly charges (>$75) elevate potential churn likelihood.")
        
    if int(inputs.get("tenure", 0)) < 12:
        summary.append("⏱️ Short tenure (<12 months) indicates customer is still in early risk stage.")
        
    return summary
