import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="3-State Forecaster Dashboard")

# --- INITIAL SESSION STATE ---
if 'history' not in st.session_state:
    # Initialize with a uniform prior
    init_v = np.array([0.333, 0.333, 0.334])
    st.session_state.current_probs = init_v
    st.session_state.history = pd.DataFrame([init_v], columns=['State 1', 'State 2', 'State 3'])

st.title("🎲 3-State Forecaster Dashboard")
st.markdown("_Tracking hidden states through time and evidence._")

# --- TOP ROW: THE LEVERS (A, B) ---
st.subheader("1. The Model Levers")
top_col1, top_col2 = st.columns(2)

with top_col1:
    st.write("**Transition Matrix (A)** - _How the world changes over time_")
    a_df = pd.DataFrame(
        [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]],
        columns=["To S1", "To S2", "To S3"],
        index=["From S1", "From S2", "From S3"]
    )
    A = st.data_editor(a_df, key="trans_matrix")

with top_col2:
    st.write("**Emissions Matrix (B)** - _How evidence relates to states_")
    b_df = pd.DataFrame(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
        columns=["Obs: Low", "Obs: Med", "Obs: High"],
        index=["If S1", "If S2", "If S3"]
    )
    B = st.data_editor(b_df, key="emiss_matrix")

st.divider()

# --- MIDDLE ROW: THE BAYES BOX ---
st.subheader("2. Current Beliefs & The Bayes Box")
mid_col1, mid_col2, mid_col3 = st.columns([1, 2, 1])

with mid_col1:
    st.write("**Forecaster Actions**")
    if st.button("⏳ Advance Time (Drift)"):
        # Markov Step
        st.session_state.current_probs = np.dot(st.session_state.current_probs, A.values)
        new_row = pd.DataFrame([st.session_state.current_probs], columns=['State 1', 'State 2', 'State 3'])
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
    
    st.write("---")
    obs_choice = st.selectbox("Select New Evidence:", ["Obs: Low", "Obs: Med", "Obs: High"])
    if st.button("👁️ Update with Evidence (Zap)"):
        obs_idx = ["Obs: Low", "Obs: Med", "Obs: High"].index(obs_choice)
        likelihoods = B.values[:, obs_idx]
        
        # Calculate Bayes Box
        priors = st.session_state.current_probs
        unnorm = priors * likelihoods
        st.session_state.current_probs = unnorm / np.sum(unnorm)
        
        # Log History
        new_row = pd.DataFrame([st.session_state.current_probs], columns=['State 1', 'State 2', 'State 3'])
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
        
        # Store for display in mid_col2
        st.session_state.last_update = {
            "Prior": priors,
            "Likelihood": likelihoods,
            "Unnormalized": unnorm
        }

with mid_col2:
    if 'last_update' in st.session_state:
        st.write("**The Bayes Box Math**")
        bbox_df = pd.DataFrame({
            "State": ["S1", "S2", "S3"],
            "Prior": st.session_state.last_update["Prior"],
            "Likelihood": st.session_state.last_update["Likelihood"],
            "Unnorm": st.session_state.last_update["Unnormalized"],
            "Posterior": st.session_state.current_probs
        })
        st.table(bbox_df.style.format("{:.3f}"))
    else:
        st.info("Click 'Update with Evidence' to see the Bayes Box math.")

with mid_col3:
    st.write("**Probability Mass**")
    st.bar_chart(st.session_state.current_probs)

# --- BOTTOM ROW: THE TREND ---
st.subheader("3. Forecasting History")
st.line_chart(st.session_state.history)

if st.button("Reset Simulation"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()
