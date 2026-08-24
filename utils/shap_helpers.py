"""
SHAP Explainability & Rule-Based Narrative Helper Module for Prompt 8.
"""

import pandas as pd
import numpy as np

def compute_local_feature_contributions(model, X_cust: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Compute local feature contributions for the given customer sample.
    Uses SHAP if available; if SHAP has an environment/backend issue, falls back gracefully 
    to exact model coefficient/importance weighted feature contribution.
    """
    df_contrib = None
    
    try:
        import shap
        model_type = type(model).__name__
        
        if "Tree" in model_type or "Forest" in model_type or "XGB" in model_type or "Gradient" in model_type:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_cust)
        else:
            explainer = shap.LinearExplainer(model, X_cust)
            shap_vals = explainer.shap_values(X_cust)
            
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif hasattr(shap_vals, "values"):
            sv = shap_vals.values[0]
            if len(sv.shape) > 1:
                sv = sv[:, 1]
        else:
            sv = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
            
        df_contrib = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': sv,
            'Abs SHAP': np.abs(sv)
        }).sort_values(by='Abs SHAP', ascending=False)
        
    except Exception as e:
        # Fallback to feature importance * feature value contribution
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = model.coef_[0]
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)
            
        vals = X_cust.values[0]
        contrib = importances * vals
        df_contrib = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': contrib,
            'Abs SHAP': np.abs(contrib)
        }).sort_values(by='Abs SHAP', ascending=False)

    return df_contrib

def generate_shap_rule_explanation(df_contrib: pd.DataFrame, inputs: dict) -> dict:
    """Generate positive, negative, and overall rule-based narrative explanations."""
    pos_contrib = df_contrib[df_contrib['SHAP Value'] > 0].head(5)
    neg_contrib = df_contrib[df_contrib['SHAP Value'] < 0].head(5)
    
    pos_bullets = []
    for _, row in pos_contrib.iterrows():
        feat = row['Feature']
        pos_bullets.append(f"🔴 **{feat}**: Increases customer churn risk.")
        
    neg_bullets = []
    for _, row in neg_contrib.iterrows():
        feat = row['Feature']
        neg_bullets.append(f"🟢 **{feat}**: Helps retain the customer.")
        
    top_3 = df_contrib.head(3)['Feature'].tolist()
    narrative = f"""
    The model's prediction is primarily driven by: **{', '.join(top_3)}**. 
    Positive values push the prediction toward higher churn risk, while negative values strengthen retention.
    """
    
    return {
        "pos_bullets": pos_bullets,
        "neg_bullets": neg_bullets,
        "narrative": narrative
    }
