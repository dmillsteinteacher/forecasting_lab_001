import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="3-State Forecaster Dashboard")

# --- INITIAL SESSION STATE ---
# We use this to keep the "memory" of the probabilities alive between clicks
if 'current_probs' not in st.session_state:
    st.session_state.current_probs = np.array([0.333, 0.333, 0.334])
    st.session_state.history = pd.DataFrame([st.session_state.current_probs], 
                                            columns=['State 1', 'State 2', 'State 3'])

st.title("🎲 3-State Forecaster Dashboard")
st.markdown("_A tool for tracking hidden states through time and evidence._")

# --- TOP METRICS ---
# Instant numerical feedback for the students
m1, m2, m3 = st.columns(3)
m1.metric("State 1 Confidence", f"{st.session_state.current_probs[0]:.1%}")
m2.metric("State 2 Confidence", f"{st.session_state.current_probs[1]:.1%}")
m3.metric("State 3 Confidence", f"{st.session_state.current_probs[2]:.1%}")

st.divider()

# --- 1. MODEL PARAMETERS ---
st.subheader("1. The Model Parameters")
top_col1, top_col2, top_col3 = st.columns([1.5, 1.5, 1])

with top_col1:
    st.write("**Transition Matrix (A)**")
    a_df = pd.DataFrame(
        [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]],
        columns=["To S1", "To S2", "To S3"],
        index=["From S1", "From S2", "From S3"]
    )
    A = st.data_editor(a_df, key="trans_matrix")

with top_col2:
    st.write("**Emissions Matrix (B)**")
    b_df = pd.DataFrame(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
        columns=["Obs: Low", "Obs: Med", "Obs: High"],
        index=["If S1", "If S2", "If S3"]
    )
    B = st.data_editor(b_df, key="emiss_matrix")

with top_col3:
    st.write("**The Starting Line ($\pi$)**")
    start_s1 = st.number_input("Init S1", 0.0, 1.0, 0.333)
    start_s2 = st.number_input("Init S2", 0.0, 1.0, 0.333)
    # Ensure probabilities sum to 1 by calculating S3
    start_s3 = 1.0 - (start_s1 + start_s2)
    st.write(f"Init S3 (auto): **{max(0, start_s3):.3f}**")
    
    if st.button("Set Initial & Reset"):
        st.session_state.current_probs = np.array([start_s1, start_s2, start_s3])
        st.session_state.history = pd.DataFrame([st.session_state.current_probs], 
                                                columns=['State 1', 'State 2', 'State 3'])
        # Clear the Bayes
