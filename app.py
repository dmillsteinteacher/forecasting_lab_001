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
    st.session_state.obs_names = ["Obs: Low", "Obs: Med", "Obs: High"]

st.title("🎲 Precision Forecaster Dashboard")

# --- 1. SETUP & CONFIGURATION ---
st.subheader("1. Define System Names")
n_col1, n_col2 = st.columns(2)

with n_col1:
    st.write("**State Names**")
    sn1 = st.text_input("State 1", "State 1", disabled=st.session_state.locked)
    sn2 = st.text_input("State 2", "State 2", disabled=st.session_state.locked)
    sn3 = st.text_input("State 3", "State 3", disabled=st.session_state.locked)
    current_states = [sn1, sn2, sn3]

with n_col2:
    st.write("**Observation Names**")
    on1 = st.text_input("Obs 1", "Obs: Low", disabled=st.session_state.locked)
    on2 = st.text_input("Obs 2", "Obs: Med", disabled=st.session_state.locked)
    on3 = st.text_input("Obs 3", "Obs: High", disabled=st.session_state.locked)
    current_obs = [on1, on2, on3]

st.divider()

top_col1, top_col2, top_col3 = st.columns([1.5, 1.5, 1])

# Dynamic DataFrames
a_init = pd.DataFrame([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]], 
                      columns=current_states, index=current_states)
b_init = pd.DataFrame([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], 
                      columns=current_obs, index=current_states)

with top_col1:
    st.write("**Transition Matrix (A)**")
    A = st.data_editor(a_init, key="trans_matrix", disabled=st.session_state.locked)

with top_col2:
    st.write("**Emissions Matrix (B)**")
    B = st.data_editor(b_init, key="emiss_matrix", disabled=st.session_state.locked)

with top_col3:
    st.write("**Initial State Vector**")
    init_s1 = st.number_input(f"Init {sn1}", 0.0, 1.0, 0.3333, format="%.4f", disabled=st.session_state.locked)
    init_s2 = st.number_input(f"Init {sn2}", 0.0, 1.0, 0.3333, format="%.4f", disabled=st.session_state.locked)
    init_s3 = 1.0 - (init_s1 + init_s2)
    st.write(f"Init {sn3}: **{max(0, init_s3):.4f}**")
    
    if not st.session_state.locked:
        if st.button("🚀 Initialize & Lock Model", use_container_width=True):
            # Detailed Validation
            errors = []
            if len(set(current_states)) < 3: errors.append("State names must be unique.")
            if len(set(current_obs)) < 3: errors.append("Observation names must be unique.")
            
            # Row check for A
            for i, row_sum in enumerate(A.sum(axis=1)):
                if not np.isclose(row_sum, 1.0, atol=1e-2):
                    errors.append(f"Transition Matrix: Row '{current_states[i]}' sums to {row_sum:.3f} (must be 1.0)")
            
            # Row check for B
            for i, row_sum in enumerate(B.sum(axis=1)):
                if not np.isclose(row_sum, 1.0, atol=1e-2):
                    errors.append(f"Emissions Matrix: Row '{current_states[i]}' sums to {row_sum:.3f} (must be 1.0)")
            
            if errors:
                for err in errors: st.error(err)
            else:
                st.session_state.locked = True
                st.session_state.state_names = current_states
                st.session_state.obs_names = current_obs
                st.session_state.current_probs = np.array([init_s1, init_s2, init_s3])
                st.session_state.history = pd.DataFrame([{
                    "Time": 0, "State": current_states[0], "Probability": float(init_s1), "Event": "Start"
                }, {
                    "Time": 0, "State": current_states[1], "Probability": float(init_s2), "Event": "Start"
                }, {
                    "Time": 0, "State": current_states[2], "Probability": float(init_s3), "Event": "Start"
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
    st.subheader("2. Forecasting Operations")
    mid_col1, mid_col2, mid_col3 = st.columns([1, 2, 1])

    with mid_col1:
        n_steps = st.number_input("Steps to Drift", 1, 100, 1)
        if st.button("⏳ Advance Time"):
            for _ in range(n_steps):
                st.session_state.time_step += 1
                st.session_state.current_probs = np.dot(st.session_state.current_probs, A.values)
                for i in range(3):
                    new_row = pd.DataFrame([{
                        "Time": st.session_state.time_step, 
                        "State": st.session_state.state_names[i], 
                        "Probability": float(st.session_state.current_probs[i]), 
                        "Event": "Drift"
                    }])
                    st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

        st.write("---")
        obs_choice = st.selectbox("Select Observation:", st.session_state.obs_names)
        if st.button("👁️ Observation Received"):
            obs_idx = st.session_state.obs_names.index(obs_choice)
            likelihoods = B.values[:, obs_idx]
            priors = st.session_state.current_probs
            unnorm = priors * likelihoods
            if np.sum(unnorm) > 0:
                st.session_state.current_probs = unnorm / np.sum(unnorm)
            
            for i in range(3):
                new_row = pd.DataFrame([{
                    "Time": st.session_state.time_step, 
                    "State": st.session_state.state_names[i], 
                    "Probability": float(st.session_state.current_probs[i]), 
                    "Event": f"Obs: {obs_choice}"
                }])
                st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
            st.session_state.last_update = {"Prior": priors, "Likelihood": likelihoods, "Label": obs_choice}

    with mid_col2:
        if 'last_update' in st.session_state:
            st.write(f"**Bayes Update Table: {st.session_state.last_update['Label']}**")
            bbox_df = pd.DataFrame({
                "State": st.session_state.state_names, 
                "Prior Prob": st.session_state.last_update["Prior"],
                "Likelihood": st.session_state.last_update["Likelihood"],
                "Posterior Prob": st.session_state.current_probs
            }).round(4)
            st.table(bbox_df)

    with mid_col3:
        st.write("**Current Belief**")
        bar_data = pd.DataFrame(st.session_state.current_probs, index=st.session_state.state_names, columns=["Probability"])
        st.bar_chart(bar_data)

    # --- 3. TREND PLOT (VEGA-LITE FOR TOOLTIP CONTROL) ---
    st.subheader("3. Forecasting History")
    
    chart_data = st.session_state.history.copy()
    chart_data["Probability"] = chart_data["Probability"].astype(float).round(4)

    # Creating a Vega-Lite chart to control Tooltip Labels ("Probability" and "State")
    st.vega_lite_chart(chart_data, {
        'mark': {'type': 'line', 'point': True},
        'encoding': {
            'x': {'field': 'Time', 'type': 'quantitative'},
            'y': {'field': 'Probability', 'type': 'quantitative', 'scale': {'domain': [0, 1]}},
            'color': {'field': 'State', 'type': 'nominal'},
            'tooltip': [
                {'field': 'Time', 'type': 'quantitative'},
                {'field': 'State', 'type': 'nominal'},
                {'field': 'Probability', 'type': 'quantitative', 'format': '.4f'}
            ]
        }
    }, use_container_width=True)
    
    with st.expander("Detailed Event Log"):
        st.dataframe(st.session_state.history)
else:
    st.info("Step 1: Configure names and matrices. Step 2: Initialize & Lock Model.")
