"""
Model Interpretation & Contributors display for Prompt 8.
"""

import streamlit as st

def render_contributor_cards(pos_bullets: list, neg_bullets: list, narrative: str):
    """Render Positive & Negative SHAP contributor cards and rule-based narrative."""
    st.markdown("### 🔍 Feature Contributors")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="custom-card" style="border-left: 4px solid #EF4444;">
                <h4 style="color: #EF4444; margin-top:0;">🔴 Top Positive Contributors (Increase Churn Risk)</h4>
        """, unsafe_allow_html=True)
        if pos_bullets:
            for b in pos_bullets:
                st.markdown(b)
        else:
            st.write("No strong positive churn factors.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div class="custom-card" style="border-left: 4px solid #10B981;">
                <h4 style="color: #10B981; margin-top:0;">🟢 Top Negative Contributors (Reduce Churn Risk)</h4>
        """, unsafe_allow_html=True)
        if neg_bullets:
            for b in neg_bullets:
                st.markdown(b)
        else:
            st.write("No strong negative churn factors.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("### 📝 Model Interpretation")
    st.info(narrative)
