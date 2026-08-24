"""
Metrics summary export functions for Prompt 6.
"""

import pandas as pd
import json

def generate_metrics_download_files(results: dict) -> tuple:
    """Generate CSV & JSON download data for model comparison."""
    rows = []
    for model_name, res in results.items():
        rows.append({
            "Model": model_name,
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1_Score": res["f1"],
            "ROC_AUC": res["roc_auc"],
            "Training_Time_Sec": res["training_time"]
        })
        
    comp_df = pd.DataFrame(rows)
    csv_bytes = comp_df.to_csv(index=False).encode('utf-8')
    
    json_bytes = json.dumps(rows, indent=4).encode('utf-8')
    
    return csv_bytes, json_bytes
