"""
Validation helper functions for ChurnSenseAI dataset.
"""

import pandas as pd
from typing import Tuple, List, Dict

REQUIRED_COLUMNS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'Churn'
]

def check_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """Check if dataframe contains recommended IBM Telco columns."""
    df_cols = list(df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in df_cols]
    found = [c for c in REQUIRED_COLUMNS if c in df_cols]
    is_valid = len(missing) == 0
    return is_valid, found, missing

def validate_target(df: pd.DataFrame, target_col: str = "Churn") -> Tuple[bool, str]:
    """Validate target column existence and value range."""
    if target_col not in df.columns:
        return False, f"Target column '{target_col}' not found in dataset."
    
    unique_vals = set(df[target_col].dropna().unique())
    valid_binary_sets = [
        {'Yes', 'No'}, {'yes', 'no'}, {1, 0}, {'1', '0'}, {'True', 'False'}, {True, False}
    ]
    
    if any(unique_vals.issubset(s) for s in valid_binary_sets) or len(unique_vals) == 2:
        return True, f"Target column '{target_col}' is valid binary classification target."
    
    return False, f"Target column '{target_col}' must be binary (e.g. Yes/No, 1/0). Found: {unique_vals}"

def validate_dataset(df: pd.DataFrame) -> Dict[str, any]:
    """Perform full dataset validation suite."""
    report = {
        "file_uploaded": df is not None and len(df) > 0,
        "is_readable": True if df is not None else False,
        "row_count": len(df) if df is not None else 0,
        "col_count": len(df.columns) if df is not None else 0,
        "duplicate_columns": list(df.columns[df.columns.duplicated()]) if df is not None else [],
        "has_duplicates": len(df.columns[df.columns.duplicated()]) > 0 if df is not None else False,
    }
    
    if df is not None:
        has_req, found_cols, missing_cols = check_required_columns(df)
        report["has_required_cols"] = has_req
        report["found_cols"] = found_cols
        report["missing_cols"] = missing_cols
        
        target_valid, target_msg = validate_target(df, "Churn")
        report["target_valid"] = target_valid
        report["target_msg"] = target_msg
        report["is_overall_valid"] = (len(df) > 0) and target_valid
    else:
        report["is_overall_valid"] = False
        report["target_valid"] = False
        report["target_msg"] = "No dataset loaded."

    return report
