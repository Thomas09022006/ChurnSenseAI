"""
Preprocessing logic & pipelines for Prompt 5.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean dataset:
    - Remove duplicate rows
    - Convert TotalCharges to numeric float, filling blanks with NaN
    - Fill missing TotalCharges with median
    - Strip whitespace from object columns
    """
    clean_df = df.copy()
    initial_rows = len(clean_df)
    
    # 1. Remove duplicate rows
    clean_df = clean_df.drop_duplicates()
    dups_removed = initial_rows - len(clean_df)
    
    # 2. Strip whitespace in string columns
    for col in clean_df.select_dtypes(include=['object']).columns:
        clean_df[col] = clean_df[col].astype(str).str.strip()
        
    # 3. Handle TotalCharges numeric conversion & missing values
    missing_totalcharges = 0
    if "TotalCharges" in clean_df.columns:
        clean_df["TotalCharges"] = pd.to_numeric(clean_df["TotalCharges"], errors='coerce')
        missing_totalcharges = clean_df["TotalCharges"].isnull().sum()
        if missing_totalcharges > 0:
            median_val = clean_df["TotalCharges"].median()
            clean_df["TotalCharges"] = clean_df["TotalCharges"].fillna(median_val)
            
    cleaning_summary = {
        "initial_rows": initial_rows,
        "clean_rows": len(clean_df),
        "dups_removed": dups_removed,
        "missing_totalcharges_filled": missing_totalcharges
    }
    
    return clean_df, cleaning_summary

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering:
    - tenure_group: 0-12, 13-24, 25-48, 48+ Months
    - charge_category: Low, Medium, High
    - avg_monthly_revenue: TotalCharges / max(tenure, 1)
    """
    fe_df = df.copy()
    
    # Tenure Group
    if "tenure" in fe_df.columns:
        bins = [-1, 12, 24, 48, 120]
        labels = ['0-12 Months', '13-24 Months', '25-48 Months', '48+ Months']
        fe_df['tenure_group'] = pd.cut(fe_df['tenure'], bins=bins, labels=labels)
        
    # Charge Category
    if "MonthlyCharges" in fe_df.columns:
        fe_df["MonthlyCharges"] = pd.to_numeric(fe_df["MonthlyCharges"], errors='coerce')
        mc_bins = [0, 35, 75, 500]
        mc_labels = ['Low', 'Medium', 'High']
        fe_df['charge_category'] = pd.cut(fe_df['MonthlyCharges'], bins=mc_bins, labels=mc_labels)
        
    # Avg Monthly Revenue
    if "TotalCharges" in fe_df.columns and "tenure" in fe_df.columns:
        fe_df["avg_monthly_revenue"] = fe_df["TotalCharges"] / fe_df["tenure"].replace(0, 1)
        
    return fe_df

def encode_and_scale(
    df: pd.DataFrame, 
    target_col: str = "Churn"
) -> Tuple[pd.DataFrame, pd.Series, Any, Any, list, Dict[str, Any]]:
    """
    Encode categorical features, scale numerical features, map target variable.
    Returns:
    X_processed, y_processed, scaler, encoders_dict, feature_names, summary
    """
    proc_df = df.copy()
    
    # Exclude identifier columns if present
    if "customerID" in proc_df.columns:
        proc_df = proc_df.drop(columns=["customerID"])
        
    # Map target
    if target_col in proc_df.columns:
        target_s = proc_df[target_col].astype(str).str.lower().str.strip()
        y = target_s.map({'yes': 1, '1': 1, 'true': 1, 'no': 0, '0': 0, 'false': 0}).fillna(0).astype(int)
        proc_df = proc_df.drop(columns=[target_col])
    else:
        y = pd.Series([0] * len(proc_df))
        
    # Separate categorical and numerical features
    cat_cols = proc_df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = proc_df.select_dtypes(include=['number']).columns.tolist()
    
    # One-Hot Encoding
    encoded_df = pd.get_dummies(proc_df, columns=cat_cols, drop_first=True)
    feature_names = encoded_df.columns.tolist()
    
    # Scale numerical features
    scaler = StandardScaler()
    if num_cols:
        encoded_df[num_cols] = scaler.fit_transform(encoded_df[num_cols])
        
    summary = {
        "original_features": len(df.columns),
        "processed_features": len(feature_names),
        "num_features_scaled": len(num_cols),
        "cat_features_encoded": len(cat_cols)
    }
    
    return encoded_df, y, scaler, cat_cols, feature_names, summary

def perform_train_test_split(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform stratified train test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    return X_train, X_test, y_train, y_test
