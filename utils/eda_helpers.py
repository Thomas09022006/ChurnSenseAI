"""
EDA helper functions for Prompt 4.
"""

import pandas as pd
import numpy as np

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> tuple:
    """Detect outliers in numerical column using IQR method."""
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return 0, 0.0, None, None, None
        
    s = df[column].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = s[(s < lower_bound) | (s > upper_bound)]
    num_outliers = len(outliers)
    pct_outliers = (num_outliers / len(s)) * 100 if len(s) > 0 else 0.0
    
    return num_outliers, round(pct_outliers, 2), lower_bound, upper_bound, outliers

def generate_business_insights(df: pd.DataFrame) -> list:
    """Generate business insights using rule-based Python logic."""
    insights = []
    
    if "Contract" in df.columns and "Churn" in df.columns:
        month_to_month_churn = df[df["Contract"] == "Month-to-month"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
        two_year_churn = df[df["Contract"] == "Two year"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
        insights.append(f"📌 **Contract Type**: Month-to-month customers have a **{month_to_month_churn:.1f}%** churn rate compared to **{two_year_churn:.1f}%** for two-year contracts.")
        
    if "InternetService" in df.columns and "Churn" in df.columns:
        fiber_churn = df[df["InternetService"] == "Fiber optic"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
        dsl_churn = df[df["InternetService"] == "DSL"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
        insights.append(f"🌐 **Internet Service**: Fiber Optic customers exhibit higher churn (**{fiber_churn:.1f}%**) than DSL users (**{dsl_churn:.1f}%**).")
        
    if "tenure" in df.columns and "Churn" in df.columns:
        avg_tenure_churn = df[df["Churn"] == "Yes"]["tenure"].mean()
        avg_tenure_retain = df[df["Churn"] == "No"]["tenure"].mean()
        insights.append(f"⏱️ **Customer Tenure**: Churned customers have an average tenure of **{avg_tenure_churn:.1f} months**, whereas retained customers average **{avg_tenure_retain:.1f} months**.")
        
    if "MonthlyCharges" in df.columns and "Churn" in df.columns:
        mc = pd.to_numeric(df["MonthlyCharges"], errors='coerce')
        temp_df = df.copy()
        temp_df["MonthlyCharges_num"] = mc
        mc_churn = temp_df[temp_df["Churn"] == "Yes"]["MonthlyCharges_num"].mean()
        mc_retain = temp_df[temp_df["Churn"] == "No"]["MonthlyCharges_num"].mean()
        insights.append(f"💳 **Monthly Charges**: Customers who churn paid a higher average monthly charge (**${mc_churn:.2f}**) vs retained customers (**${mc_retain:.2f}**).")
        
    if "TechSupport" in df.columns and "Churn" in df.columns:
        no_support_churn = df[df["TechSupport"] == "No"]["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
        insights.append(f"🛠️ **Tech Support**: Customers without Tech Support have a churn rate of **{no_support_churn:.1f}%**.")
        
    return insights
