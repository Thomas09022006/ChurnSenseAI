"""
Dataset Upload Business Logic Module.
"""

import pandas as pd
import streamlit as st
import datetime
from utils.file_helpers import load_dataset_from_bytes, calculate_memory_usage
from utils.validation_helpers import validate_dataset

def process_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read uploaded file, validate and update Streamlit session state."""
    file_bytes = uploaded_file.getvalue()
    df = load_dataset_from_bytes(file_bytes, uploaded_file.name)
    
    validation_report = validate_dataset(df)
    
    st.session_state['df'] = df
    st.session_state['dataset_name'] = uploaded_file.name
    st.session_state['upload_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state['validation_status'] = validation_report['is_overall_valid']
    st.session_state['validation_report'] = validation_report
    
    return df
