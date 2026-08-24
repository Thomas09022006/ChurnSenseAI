"""
Prediction Report Generator for Prompt 7.
"""

import pandas as pd
import json

def generate_prediction_report_files(pred_res: dict) -> tuple:
    """Generate CSV & JSON download data for prediction report."""
    input_df = pd.DataFrame([pred_res["inputs"]])
    input_df["Prediction"] = pred_res["prediction_label"]
    input_df["Churn_Probability_%"] = round(pred_res["churn_prob"] * 100, 2)
    input_df["Risk_Level"] = pred_res["risk_level"]
    input_df["Confidence"] = pred_res["confidence"]
    input_df["Timestamp"] = pred_res["timestamp"]
    
    csv_bytes = input_df.to_csv(index=False).encode('utf-8')
    
    json_data = {
        "prediction": pred_res["prediction_label"],
        "churn_probability": pred_res["churn_prob"],
        "risk_level": pred_res["risk_level"],
        "confidence": pred_res["confidence"],
        "summary": pred_res["summary"],
        "customer_inputs": pred_res["inputs"],
        "timestamp": pred_res["timestamp"]
    }
    json_bytes = json.dumps(json_data, indent=4).encode('utf-8')
    
    return csv_bytes, json_bytes
