

"""
Plotly Chart Builders for EDA Module.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif")
)

def generate_histogram(df: pd.DataFrame, col: str, hue: str = None):
    """Generate interactive histogram for numerical column."""
    fig = px.histogram(
        df, 
        x=col, 
        color=hue, 
        marginal="box",
        title=f"Distribution of {col}" + (f" by {hue}" if hue else ""),
        color_discrete_sequence=['#6366F1', '#10B981', '#EF4444'],
        barmode="overlay",
        opacity=0.75
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig

def generate_boxplot(df: pd.DataFrame, col: str, hue: str = None):
    """Generate boxplot for numerical column."""
    fig = px.box(
        df, 
        x=hue if hue else None, 
        y=col, 
        color=hue if hue else None,
        title=f"Box Plot of {col}" + (f" by {hue}" if hue else ""),
        color_discrete_sequence=['#6366F1', '#EF4444']
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig

def generate_bar_chart(df: pd.DataFrame, col: str, hue: str = None):
    """Generate bar chart for categorical column."""
    if hue:
        counts = df.groupby([col, hue]).size().reset_index(name='Count')
        fig = px.bar(
            counts, 
            x=col, 
            y='Count', 
            color=hue, 
            barmode='group',
            title=f"{col} Breakdown by {hue}",
            color_discrete_sequence=['#10B981', '#EF4444']
        )
    else:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, 'Count']
        fig = px.bar(
            counts, 
            x=col, 
            y='Count', 
            title=f"Frequency Count of {col}",
            color='Count',
            color_continuous_scale='Viridis'
        )
    fig.update_layout(**DARK_LAYOUT)
    return fig

def generate_correlation_heatmap(df: pd.DataFrame):
    """Generate correlation heatmap for numerical features."""
    num_df = df.select_dtypes(include=['number']).copy()
    if "TotalCharges" in df.columns and "TotalCharges" not in num_df.columns:
        num_df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
        
    corr = num_df.corr()
    
    fig = px.imshow(
        corr,
        text_auto=".2f",
        title="Numerical Features Correlation Heatmap",
        color_continuous_scale='RdBu_r',
        aspect="auto"
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig
