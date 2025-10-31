import streamlit as st
import os

# --- Import the refactored dashboard functions ---
# Import the ROUTER functions, which contain the sidebar logic
try:
    from ui.amlsim_pages import draw_amlsim_router
    from ui.bitcoin_pages import draw_bitcoin_router
except ImportError as e:
    st.error(f"FATAL ERROR: Could not find project modules. Ensure you have run file renaming commands.")
    st.error(f"Error details: {e}")
    st.stop()
    
# --- Application Setup ---

st.set_page_config(layout="wide", page_title="Inferno GNN Analyzer")

st.sidebar.title("🔥 Inferno Project Selection")
st.sidebar.markdown("### GNN Financial Crime Analyzer")

# Check if data exists for either project (for status message)
amlsim_data_exists = os.path.exists('outputs/suspicion_scores.csv')
bitcoin_data_exists = os.path.exists('outputs/bitcoin_predictions.csv')

# --- Main Selection Logic ---

project_options = {
    "Select Project": "landing",
    "1. AMLSim (Edge Classification)": "amlsim",
    "2. Elliptic (Node Classification)": "bitcoin"
}

selection = st.sidebar.radio("Choose Project Model Type:", list(project_options.keys()))
project_mode = project_options[selection]

if project_mode == "landing":
    st.title("🔥 Inferno Project Portfolio")
    st.markdown("### Unified GNN Financial Crime Detection Dashboard")
    st.markdown("Welcome! Please select a project from the sidebar to begin analysis.")
    
    st.markdown("---")
    st.subheader("Project Status")
    
    st.metric("AMLSim Status (Edge Classification)", "Ready" if amlsim_data_exists else "Data Missing", 
              delta="Run train/predict scripts" if not amlsim_data_exists else None)
    st.metric("Elliptic Status (Node Classification)", "Ready" if bitcoin_data_exists else "Data Missing",
              delta="Run train/predict scripts" if not bitcoin_data_exists else None)

elif project_mode == "amlsim":
    if amlsim_data_exists:
        draw_amlsim_router() # Calls the secondary menu for AMLSim
    else:
        st.error("AMLSim data not found. Please run the data_preprocessing.py, train.py, and predict.py scripts first.")

elif project_mode == "bitcoin":
    if bitcoin_data_exists:
        draw_bitcoin_router() # Calls the secondary menu for Bitcoin
    else:
        st.error("Elliptic Bitcoin data not found. Please run the data_preprocessing_bitcoin.py, train_bitcoin.py, and predict_bitcoin.py scripts first.")
