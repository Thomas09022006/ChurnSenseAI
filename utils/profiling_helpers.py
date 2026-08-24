"""
Profiling Helper Functions for Dataset Overview (Prompt 3).
"""

import pandas as pd
import numpy as np

def calculate_quality_score(df: pd.DataFrame) -> dict:
    """
    Calculate Data Quality Score out of 100 based on:
    - Missing Values
    - Duplicate Rows
    - Constant Columns
    - Invalid Data Types
    """
    if df is None or len(df) == 0:
        return {"score": 0, "rating": "Poor", "color": "#EF4444"}

    score = 100
    
    # Missing value penalty
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100
    score -= min(missing_pct * 1.5, 30)
    
    # Duplicate rows penalty
    dup_pct = (df.duplicated().sum() / len(df)) * 100
    score -= min(dup_pct * 2, 20)
    
    # Constant columns penalty
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    score -= len(constant_cols) * 10
    
    score = max(0, round(score, 1))
    
    if score >= 95:
        rating = "Excellent"
        color = "#10B981"
    elif score >= 80:
        rating = "Good"
        color = "#3B82F6"
    elif score >= 60:
        rating = "Needs Cleaning"
        color = "#F59E0B"
    else:
        rating = "Poor"
        color = "#EF4444"
        
    return {
        "score": score,
        "rating": rating,
        "color": color,
        "missing_pct": round(missing_pct, 2),
        "dup_pct": round(dup_pct, 2),
        "constant_cols": constant_cols
    }

def generate_insights(df: pd.DataFrame) -> list:
    """Generate 8-10 rule-based data science insights automatically."""
    insights = []
    num_rows, num_cols = df.shape
    insights.append(f"📊 Dataset contains **{num_rows:,}** customer records and **{num_cols}** features.")
    
    if "Churn" in df.columns:
        counts = df["Churn"].value_counts(normalize=True) * 100
        churn_yes = float(counts.loc["Yes"]) if "Yes" in counts.index else (float(counts.loc[1]) if 1 in counts.index else 0.0)
        insights.append(f"📉 Churn Rate is **{churn_yes:.1f}%** across the dataset.")
        if churn_yes < 30:
            insights.append("⚖️ Target variable shows moderate class imbalance (majority retained).")
            
    if "tenure" in df.columns:
        avg_tenure = df["tenure"].mean()
        insights.append(f"⏱️ Average customer tenure is **{avg_tenure:.1f} months**.")
        
    if "MonthlyCharges" in df.columns:
        avg_charge = pd.to_numeric(df["MonthlyCharges"], errors='coerce').mean()
        insights.append(f"💳 Average monthly charge per customer is **${avg_charge:.2f}**.")
        
    if "TotalCharges" in df.columns:
        tot_missing = pd.to_numeric(df["TotalCharges"], errors='coerce').isnull().sum()
        if tot_missing > 0:
            insights.append(f"⚠️ TotalCharges contains **{tot_missing}** missing or blank entries requiring numeric conversion.")
        else:
            insights.append("✔ TotalCharges is fully populated.")
            
    if "Contract" in df.columns:
        top_contract = df["Contract"].mode()[0]
        insights.append(f"📄 Most common contract type is **{top_contract}**.")
        
    if "InternetService" in df.columns:
        top_internet = df["InternetService"].mode()[0]
        insights.append(f"🌐 Primary internet service type is **{top_internet}**.")
        
    dup_cnt = df.duplicated().sum()
    if dup_cnt > 0:
        insights.append(f"⚠️ Detected **{dup_cnt}** duplicate row(s) in dataset.")
    else:
        insights.append("✔ No duplicate rows found in dataset.")
        
    return insights

def generate_recommendations(df: pd.DataFrame) -> list:
    """Generate automatic recommendations for data cleaning & preprocessing."""
    recs = []
    if "TotalCharges" in df.columns:
        recs.append("🔧 Convert `TotalCharges` from string/object to numeric float type and impute missing values with median.")
    recs.append("🏷️ Apply One-Hot Encoding for multi-category categorical features (e.g. Contract, PaymentMethod, InternetService).")
    recs.append("🔢 Normalize/Scale numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using StandardScaler.")
    recs.append("💡 Perform Feature Engineering to derive tenure buckets and monthly charge categories.")
    recs.append("⚖️ Use SMOTE or class weighting during model training to handle potential churn class imbalance.")
    return recs
