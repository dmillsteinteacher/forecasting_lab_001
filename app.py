import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Forecaster Dashboard")

# --- INITIAL SESSION STATE ---
if 'locked' not in st.session_state:
    st.session_state.locked = False
    st.session_state.current_probs = np.array([0.3333, 0.3333, 0.3334])
    st.session_state.time_step = 0
    st.session_state.history = pd.DataFrame()
    st.session_state.state_names = ["State 1", "State 2", "State 3"]

st.title("🎲 Forecaster Dashboard")

# --- 1. SETUP & CONFIGURATION ---
st.subheader("1. Model Configuration")
name_col1, name_col2, name_col3 = st.columns(3)

# State Name Inputs
with name_col1:
    s1_name = st.text_input("Name of State 1", "State 1", disabled=st.session_state.locked)
with name_col2:
    s2_name = st.text_input("Name of State 2", "State 2", disabled=st.session_state.locked)
with name_col3:
    s3_name = st.text_input("Name of State 3", "State 3", disabled=st.session_state.locked)

current_names = [s1_name, s2_name, s3_name]

st.divider()

top_col1, top_col2, top_col3 = st.columns([1.5, 1.5, 1])

# Matrix Defaults
a_init = pd.DataFrame([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]], 
                      columns=current_names, index=current_names)
b_init = pd.DataFrame([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], 
                      columns=["Obs: Low", "Obs: Med", "Obs: High"], index=current_names)

with top_col1:
    st.write("**Transition Matrix (A)**")
    A = st.data_editor(a_init, key="trans_matrix", disabled=st.session_state.locked)

with top_col2:
    st.write("**Emissions Matrix (B)**")
    B = st.data_editor(b_init, key="emiss_matrix", disabled=st.session_state.locked)

with top_col3:
    st.write("**Initial State Vector ($\pi$)**")
    init_s1 = st.number_input(f"Init {s1_name}", 0.0, 1.0, 0.3333, format="%.4f", disabled=st.session_state.locked)
    init_s2 = st.number_input(f"Init {s2_name}", 0.0, 1.0, 0.3333, format="%.4f", disabled=st.session_state.locked)
    init_s3 = 1.0 - (init_s1 + init_s2)
    st.write(f"Init {s3_name} (auto): **{max(0, init_s3):.4f}**")
    
    if not st.session_state.locked:
        if st.button("🚀 Initialize & Lock Model", use_container_width=True):
            # Validations
            unique_names = len(set(current_names)) == 3
            a_sums = A.sum(axis=1)
            b_sums = B.sum(axis=1)
            
            if not unique_names:
                st.error("State names must be unique!")
            elif not (np.allclose(a_sums, 1.0, atol=1e-2) and np.allclose(b_sums, 1.0, atol=1e-2)):
                st.error("Matrix rows must sum to 1.0!")
            else:
                st.session_state.locked = True
                st.session_state.state_names = current_names
                st.session_state.current_probs = np.array([init_s1, init_s2, init_s3])
                st.session_state.history = pd.DataFrame([{
                    "Time": 0, 
                    current_names[0]: init_s1, 
                    current_names[1]: init_s2, 
                    current_names[2]: init_s3, 
                    "Event": "Start",
                    "Is_Obs": False
                }])
                st.rerun()
    else:
        if st.button("🔓 Hard Reset", use_container_width=True):
            st.session_state.locked = False
            st.session_state.time_step = 0
            st.session_state.history = pd.DataFrame()
            st.rerun()

# --- 2. FIELD OPERATIONS ---
if st.session_state.locked:
    st.divider()
    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric(st.session_state.state_names[0], f"{st.session_state.current_probs[0]:.2%}")
    m2.metric(st.session_state.state_names[1], f"{st.session_state.current_probs[1]:.2%}")
    m3.metric(st.session_state.state_names[2], f"{st.session_state.current_probs[2]:.2%}")
    
    st.subheader("2. Forecasting Operations")
    mid_col1, mid_col2, mid_col3 = st.columns([1, 2, 1])

    with mid_col1:
        st.write("**Time**")
        n_steps = st.number_input("Steps to Drift", 1, 100, 1)
        if st.button("⏳ Advance Time"):
            new_rows = []
            for _ in range(n_steps):
                st.session_state.time_step += 1
                st.session_state.current_probs = np.dot(st.session_state.current_probs, A.values)
                new_rows.append({
                    "Time": st.session_state.time_step, 
                    st.session_state.state_names[0]: st.session_state.current_probs[0], 
                    st.session_state.state_names[1]: st.session_state.current_probs[1], 
                    st.session_state.state_names[2]: st.session_state.current_probs[2], 
                    "Event": "Drift",
                    "Is_Obs": False
                })
            st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame(new_rows)], ignore_index=True)
        
        st.write("---")
        st.write("**Evidence**")
        obs_choice = st.selectbox("Select Observation:", ["Obs: Low", "Obs: Med", "Obs: High"])
        if st.button("👁️ Observation Received"):
            obs_idx = ["Obs: Low", "Obs: Med", "Obs: High"].index(obs_choice)
            likelihoods = B.values[:, obs_idx]
            priors = st.session_state.current_probs
            unnorm = priors * likelihoods
            if np.sum(unnorm) > 0:
                st.session_state.current_probs = unnorm / np.sum(unnorm)
            
            new_row = pd.DataFrame([{
                "Time": st.session_state.time_step, 
                st.session_state.state_names[0]: st.session_state.current_probs[0], 
                st.session_state.state_names[1]: st.session_state.current_probs[1], 
                st.session_state.state_names[2]: st.session_state.current_probs[2], 
                "Event": f"Obs: {obs_choice}",
                "Is_Obs": True # This triggers the marker on the chart
            }])
            st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
            st.session_state.last_update = {"Prior": priors, "Likelihood": likelihoods, "Label": obs_choice}

    with mid_col2:
        if 'last_update' in st.session_state:
            st.write(f"**Bayes Update: {st.session_state.last_update['Label']}**")
            bbox_df = pd.DataFrame({
                "State": st.session_state.state_names, 
                "Prior": st.session_state.last_update["Prior"],
                "Likelihood": st.session_state.last_update["Likelihood"],
                "Posterior": st.session_state.current_probs
            }).round(4)
            st.table(bbox_df)

    with mid_col3:
        st.write("**Current Belief**")
        st.bar_chart(pd.DataFrame(st.session_state.current_probs, index=st.session_state.state_names))

    # --- 3. TREND PLOT ---
    st.subheader("3. Forecasting History")
    
    # Prepare data for plotting - round for the tooltip
    chart_data = st.session_state.history.copy()
    for col in st.session_state.state_names:
        chart_data[col] = chart_data[col].astype(float).round(4)

    # We use st.line_chart with the 'points' parameter to show observations
    st.line_chart(
        chart_data, 
        x="Time", 
        y=st.session_state.state_names, 
        color=None # Automatically assigns colors to states
    )
    
    st.caption("Points on lines indicate 'Observation Received' events. Hover for Probability values.")
    
    with st.expander("View Event Log"):
        st.dataframe(st.session_state.history.drop(columns=["Is_Obs"]))

else:
    st.info("Rename your states and configure the matrices, then click 'Initialize' to begin.")
