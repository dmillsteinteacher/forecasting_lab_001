import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="3-State Forecaster Dashboard")

# --- INITIAL SESSION STATE ---
if 'current_probs' not in st.session_state:
    st.session_state.current_probs = np.array([0.333, 0.333, 0.334])
    st.session_state.history = pd.DataFrame([st.session_state.current_probs], 
                                            columns=['State 1', 'State 2', 'State 3'])

st.title("🎲 3-State Forecaster Dashboard")
st.markdown("_A tool for tracking hidden states through time and evidence._")

# --- TOP METRICS ---
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
    start_s3 = 1.0 - (start_s1 + start_s2)
    st.write(f"Init S3 (auto): **{max(0, start_s3):.3f}**")
    
    if st.button("Set Initial & Reset"):
        st.session_state.current_probs = np.array([start_s1, start_s2, start_s3])
        st.session_state.history = pd.DataFrame([st.session_state.current_probs], 
                                                columns=['State 1', 'State 2', 'State 3'])
        if 'last_update' in st.session_state:
            del st.session_state.last_update
        st.rerun()

# --- VALIDATION ENGINE (Informational Only) ---
def validate_matrix(df, label):
    sums = df.sum(axis=1)
    # Check if rows sum to ~1.0. If not, show warning but don't break UI.
    if not np.allclose(sums, 1.0, atol=1e-2):
        st.warning(f"⚠️ {label} rows should sum to 1.0. Current: {list(sums.round(2))}")

validate_matrix(A, "Transition Matrix")
validate_matrix(B, "Emissions Matrix")

st.divider()

# --- 2. FORECASTING ACTIONS ---
st.subheader("2. Forecasting Actions")
mid_col1, mid_col2, mid_col3 = st.columns([1, 2, 1])

with mid_col1:
    if st.button("⏳ Advance Time (Drift)"):
        # Markov Step: Probabilities * Transition Matrix
        st.session_state.current_probs = np.dot(st.session_state.current_probs, A.values)
        new_row = pd.DataFrame([st.session_state.current_probs], columns=['State 1', 'State 2', 'State 3'])
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
    
    st.write("---")
    obs_choice = st.selectbox("Observe Data:", ["Obs: Low", "Obs: Med", "Obs: High"])
    if st.button("👁️ Update (Bayes Zap)"):
        obs_idx = ["Obs: Low", "Obs: Med", "Obs: High"].index(obs_choice)
        likelihoods = B.values[:, obs_idx]
        
        priors = st.session_state.current_probs
        unnorm = priors * likelihoods
        
        # Safety check for division by zero
        total = np.sum(unnorm)
        if total > 0:
            st.session_state.current_probs = unnorm / total
        
        new_row = pd.DataFrame([st.session_state.current_probs], columns=['State 1', 'State 2', 'State 3'])
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
        st.session_state.last_update = {"Prior": priors, "Likelihood": likelihoods, "Unnorm": unnorm}

with mid_col2:
    if 'last_update' in st.session_state:
        st.write("**The Bayes Box Math**")
        bbox_df = pd.DataFrame({
            "State": ["S1", "S2", "S3"],
            "Prior": st.session_state.last_update["Prior"],
            "Likelihood": st.session_state.last_update["Likelihood"],
            "Unnorm": st.session_state.last_update["Unnorm"],
            "Posterior": st.session_state.current_probs
        })
        
        # Clean numeric formatting for the table
        numeric_cols = ["Prior", "Likelihood", "Unnorm", "Posterior"]
        bbox_df[numeric_cols] = bbox_df[numeric_cols].astype(float).round(3)
        st.table(bbox_df)
    else:
        st.info("The Bayes Box calculations will appear here after you click 'Update'.")

with mid_col3:
    st.write("**Current Belief Mass**")
    st.bar_chart(st.session_state.current_probs)

# --- 3. TREND ---
st.subheader("3. Forecasting History")
st.line_chart(st.session_state.history)

if st.button("Hard Reset Everything"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
