"""
Model Comparison Table & Feature Importance Module for Prompt 6.
"""

import pandas as pd
import streamlit as st
import plotly.express as px

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif")
)

def render_comparison_table(results: dict):
    """Render comparison table of trained models sorted by Accuracy."""
    table_data = []
    for model_name, res in results.items():
        table_data.append({
            "Model": model_name,
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1 Score": res["f1"],
            "ROC-AUC": res["roc_auc"],
            "Training Time (s)": res["training_time"]
        })
        
    df_comp = pd.DataFrame(table_data).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    st.markdown("### 📊 Model Performance Comparison")
    st.dataframe(df_comp, use_container_width=True)
    
    # Accuracy Comparison Bar Chart
    fig = px.bar(
        df_comp,
        x="Model",
        y="Accuracy",
        color="Model",
        text="Accuracy",
        title="Model Accuracy Comparison",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(**DARK_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    
    return df_comp

def render_feature_importance_charts(results: dict):
    """Render Top 15 Feature Importances for trained tree/linear models."""
    st.markdown("### 📈 Feature Importance Analysis")
    tabs = st.tabs(list(results.keys()))
    
    for tab, (model_name, res) in zip(tabs, results.items()):
        with tab:
            feat_df = res["feature_importance"].head(15)
            fig = px.bar(
                feat_df,
                x="Importance",
                y="Feature",
                orientation='h',
                title=f"Top 15 Important Features ({model_name})",
                color="Importance",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), **DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
