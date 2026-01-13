import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="3-State Forecaster Dashboard")

# --- INITIAL SESSION STATE ---
if 'locked' not in st.session_state:
    st.session_state.locked = False
    st.session_state.current_probs = np.array([0.333, 0.333, 0.334])
    st.session_state.time_step = 0
    st.session_state.history = pd.DataFrame()

st.title("🎲 3-State Forecaster Dashboard")
st.markdown("_A tool for tracking hidden states through time and evidence._")

# --- TOP METRICS ---
if st.session_state.locked:
    m1, m2, m3 = st.columns(3)
    # Using delta to show how much confidence has changed since the last step
    m1.metric("State 1 Confidence", f"{st.session_state.current_probs[0]:.1%}")
    m2.metric("State 2 Confidence", f"{st.session_state.current_probs[1]:.1%}")
    m3.metric("State 3 Confidence", f"{st.session_state.current_probs[2]:.1%}")
    st.divider()

# --- 1. MODEL PARAMETERS ---
st.subheader("1. Setup & Model Configuration")
top_col1, top_col2, top_col3 = st.columns([1.5, 1.5, 1])

# Initial matrix defaults
a_init = pd.DataFrame([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]], 
                      columns=["To S1", "To S2", "To S3"], index=["From S1", "From S2", "From S3"])
b_init = pd.DataFrame([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], 
                      columns=["Obs: Low", "Obs: Med", "Obs: High"], index=["If S1", "If S2", "If S3"])

with top_col1:
    st.write("**Transition Matrix (A)**")
    A = st.data_editor(a_init, key="trans_matrix", disabled=st.session_state.locked)

with top_col2:
    st.write("**Emissions Matrix (B)**")
    B = st.data_editor(b_init, key="emiss_matrix", disabled=st.session_state.locked)

with top_col3:
    st.write("**Initial State Vector ($\pi$)**")
    s1 = st.number_input("Init S1", 0.0, 1.0, 0.333, disabled=st.session_state.locked)
    s2 = st.number_input("Init S2", 0.0, 1.0, 0.333, disabled=st.session_state.locked)
    s3 = 1.0 - (s1 + s2)
    st.write(f"Init S3 (auto): **{max(0, s3):.3f}**")
    
    if not st.session_state.locked:
        if st.button("🚀 Initialize & Lock Model", use_container_width=True):
            a_sums = A.sum(axis=1)
            b_sums = B.sum(axis=1)
            if np.allclose(a_sums, 1.0, atol=1e-2) and np.allclose(b_sums, 1.0, atol=1e-2) and s3 >= -0.001:
                st.session_state.locked = True
                st.session_state.current_probs = np.array([s1, s2, s3])
                st.session_state.history = pd.DataFrame([{
                    "Time": 0, "State 1": s1, "State 2": s2, "State 3": max(0, s3), "Event": "Start"
                }])
                st.rerun()
            else:
                st.error("Row sums must equal 1.0 (approx) before locking!")
    else:
        if st.button("🔓 Hard Reset (Unlock)", use_container_width=True):
            st.session_state.locked = False
            st.session_state.time_step = 0
            st.session_state.history = pd.DataFrame()
            if 'last_update' in st.session_state: del st.session_state.last_update
            st.rerun()

# --- 2. FORECASTING ACTIONS ---
if st.session_state.locked:
    st.divider()
    st.subheader("2. Forecasting Field Operations")
    mid_col1, mid_col2, mid_col3 = st.columns([1, 2, 1])

    with mid_col1:
        st.write("**Option A: Passage of Time**")
        n_steps = st.number_input("Steps to Drift", 1, 100, 1)
        if st.button(f"⏳ Advance Time"):
            new_rows = []
            for _ in range(n_steps):
                st.session_state.time_step += 1
                st.session_state.current_probs = np.dot(st.session_state.current_probs, A.values)
                new_rows.append({"Time": st.session_state.time_step, "State 1": st.session_state.current_probs[0], 
                                 "State 2": st.session_state.current_probs[1], "State 3": st.session_state.current_probs[2], "Event": "Drift"})
            st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame(new_rows)], ignore_index=True)
        
        st.write("---")
        st.write("**Option B: New Evidence**")
        obs_choice = st.selectbox("Select Observation:", ["Obs: Low", "Obs: Med", "Obs: High"])
        if st.button("👁️ Observation Received"):
            obs_idx = ["Obs: Low", "Obs: Med", "Obs: High"].index(obs_choice)
            likelihoods = B.values[:, obs_idx]
            priors = st.session_state.current_probs
            unnorm = priors * likelihoods
            if np.sum(unnorm) > 0:
                st.session_state.current_probs = unnorm / np.sum(unnorm)
            
            new_row = pd.DataFrame([{"Time": st.session_state.time_step, "State 1": st.session_state.current_probs[0], 
                                     "State 2": st.session_state.current_probs[1], "State 3": st.session_state.current_probs[2], "Event": f"Obs: {obs_choice}"}])
            st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
            st.session_state.last_update = {"Prior": priors, "Likelihood": likelihoods, "Unnorm": unnorm, "Label": obs_choice}

    with mid_col2:
        if 'last_update' in st.session_state:
            st.write(f"**Bayesian Inference for {st.session_state.last_update['Label']}**")
            bbox_df = pd.DataFrame({
                "State": ["S1", "S2", "S3"], 
                "Prior (P(H))": st.session_state.last_update["Prior"],
                "Likelihood (P(E|H))": st.session_state.last_update["Likelihood"],
                "Unnormalized": st.session_state.last_update["Unnorm"],
                "Posterior (P(H|E))": st.session_state.current_probs
            }).round(4)
            st.table(bbox_df)
            st.caption("Posterior = (Prior × Likelihood) / Sum of Unnormalized")

    with mid_col3:
        st.write("**Current Belief Mass**")
        st.bar_chart(st.session_state.current_probs)

    # --- 3. TREND PLOT ---
    st.subheader("3. Forecasting History")
    st.line_chart(st.session_state.history, x="Time", y=["State 1", "State 2", "State 3"])
    
    with st.expander("View Full Event Log"):
        st.dataframe(st.session_state.history)
else:
    st.info("Set your parameters above and hit 'Initialize & Lock' to begin the forecasting simulation.")
