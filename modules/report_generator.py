"""
SHAP Report Generator Module for Prompt 8.
"""

import pandas as pd
import json

def generate_shap_report_files(df_contrib: pd.DataFrame, narrative: str, pred_res: dict) -> tuple:
    """Generate downloadable CSV & JSON reports for SHAP Explainability."""
    csv_bytes = df_contrib[['Feature', 'SHAP Value', 'Abs SHAP']].to_csv(index=False).encode('utf-8')
    
    report_dict = {
        "prediction_summary": {
            "prediction": pred_res.get("prediction_label", "N/A"),
            "probability": pred_res.get("churn_prob", 0.0),
            "risk_level": pred_res.get("risk_level", "N/A"),
            "confidence": pred_res.get("confidence", "N/A")
        },
        "interpretation": narrative,
        "feature_contributions": df_contrib.to_dict(orient="records")
    }
    json_bytes = json.dumps(report_dict, indent=4).encode('utf-8')
    
    return csv_bytes, json_bytes
