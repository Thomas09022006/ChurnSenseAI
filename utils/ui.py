"""
UI Customizations, CSS styling, Progress Stepper, and Navigation helpers for ChurnSenseAI.
"""

import streamlit as st

def apply_custom_theme():
    """Apply modern sleek dark theme styling to Streamlit app."""
    st.markdown("""
        <style>
        /* Modern Dark Theme Styling */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .main {
            background-color: #0E1117;
            color: #FAFAFA;
        }

        /* Glassmorphic Rounded Cards */
        .custom-card {
            background: rgba(22, 27, 34, 0.75);
            border-radius: 14px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            backdrop-filter: blur(8px);
            margin-bottom: 20px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .custom-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
            border-color: rgba(99, 102, 241, 0.4);
        }

        .kpi-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
            border-radius: 12px;
            padding: 18px 22px;
            border-left: 4px solid #6366F1;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            margin-bottom: 16px;
        }

        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #6366F1;
            margin-top: 4px;
        }

        .kpi-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94A3B8;
        }

        /* Stepper Navigation */
        .stepper-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(22, 27, 34, 0.8);
            padding: 14px 20px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 25px;
            overflow-x: auto;
        }

        .step-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
            font-weight: 500;
            white-space: nowrap;
        }

        .step-active {
            color: #6366F1;
            font-weight: 700;
        }

        .step-completed {
            color: #10B981;
        }

        .step-pending {
            color: #64748B;
        }

        .step-divider {
            color: #334155;
            margin: 0 4px;
        }

        /* Badge Styling */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .badge-success { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-warning { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
        .badge-danger { background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }
        .badge-info { background: rgba(99, 102, 241, 0.15); color: #818CF8; border: 1px solid rgba(99, 102, 241, 0.3); }

        /* Streamlit Button Overrides */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        
        </style>
    """, unsafe_allow_html=True)

def render_stepper(current_step_index: int):
    """
    Render progress stepper across 8 workflow steps.
    Steps:
    0: Home
    1: Upload
    2: Overview
    3: EDA
    4: Preprocessing
    5: Model Training
    6: Prediction
    7: SHAP Explainability
    """
    steps = [
        ("🏠", "Home"),
        ("📂", "Upload"),
        ("📋", "Overview"),
        ("📊", "EDA"),
        ("⚙️", "Preprocessing"),
        ("🤖", "Training"),
        ("🔮", "Prediction"),
        ("💡", "SHAP")
    ]
    
    html = '<div class="stepper-container">'
    for idx, (icon, name) in enumerate(steps):
        if idx < current_step_index:
            status_class = "step-completed"
            symbol = "✅"
        elif idx == current_step_index:
            status_class = "step-active"
            symbol = "🟢"
        else:
            status_class = "step-pending"
            symbol = "⚪"
            
        html += f'<div class="step-item {status_class}"><span>{symbol}</span> {icon} {name}</div>'
        if idx < len(steps) - 1:
            html += '<div class="step-divider">›</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_kpi(label: str, value: str, border_color: str = "#6366F1"):
    """Render a KPI metric card."""
    st.markdown(f"""
        <div class="kpi-card" style="border-left-color: {border_color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color: {border_color};">{value}</div>
        </div>
    """, unsafe_allow_html=True)

def render_nav_buttons(
    prev_page: str = None, 
    next_page: str = None, 
    prev_label: str = "Previous", 
    next_label: str = "Next",
    next_disabled: bool = False
):
    """Render Next / Previous navigation buttons at bottom of page."""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if prev_page:
            if st.button(f"⬅️ {prev_label}", key=f"btn_prev_{prev_page}"):
                st.switch_page(prev_page)
                
    with col3:
        if next_page:
            if st.button(f"{next_label} ➡️", key=f"btn_next_{next_page}", disabled=next_disabled, type="primary"):
                st.switch_page(next_page)
