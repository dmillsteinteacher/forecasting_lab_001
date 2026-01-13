import streamlit as st
import pandas as pd

st.set_page_config(page_title="Bayes Box Engine", page_icon="🎲")

st.title("📦 The Moving Bayes Box")

# --- INITIALIZATION ---
# This block only runs once when the app first loads
if 'probs' not in st.session_state:
    st.session_state.probs = [0.5, 0.5]  # [Stable, Unstable]
    st.session_state.history = []

# --- THE MATH LOGIC ---
def apply_drift():
    """The Markov Step: Move probability through the transition matrix."""
    curr_s, curr_u = st.session_state.probs
    # Example Matrix: Stable->Stable (0.9), Unstable->Unstable (0.8)
    new_s = (curr_s * 0.9) + (curr_u * 0.2)
    new_u = (curr_s * 0.1) + (curr_u * 0.8)
    st.session_state.probs = [new_s, new_u]
    st.session_state.history.append("Time Drifted")

def apply_update(obs_type):
    """The Bayes Step: Update based on a 'Sense' (Observation)."""
    curr_s, curr_u = st.session_state.probs
    
    # Likelihoods: P(High Temp | Stable) = 0.2, P(High Temp | Unstable) = 0.8
    if obs_type == "High Temp":
        likelihoods = [0.2, 0.8]
    else: # Low Temp
        likelihoods = [0.8, 0.2]
        
    # Bayes Box Calculation: Prior * Likelihood
    unnorm = [curr_s * likelihoods[0], curr_u * likelihoods[1]]
    total = sum(unnorm)
    st.session_state.probs = [x / total for x in unnorm]
    st.session_state.history.append(f"Sensed {obs_type}")

# --- UI LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Controls")
    if st.button("⏳ Advance 1 Day (Drift)"):
        apply_drift()
        
    st.divider()
    
    st.write("Current Observation:")
    if st.button("🔥 Sense: High Temp"):
        apply_update("High Temp")
    if st.button("❄️ Sense: Low Temp"):
        apply_update("Low Temp")
        
    if st.button("🔄 Reset Machine"):
        st.session_state.probs = [0.5, 0.5]
        st.session_state.history = []
        st.rerun()

with col2:
    st.subheader("Current Beliefs")
    # Display as a bar chart
    chart_data = pd.DataFrame({
        'State': ['Stable', 'Unstable'],
        'Probability': st.session_state.probs
    })
    st.bar_chart(chart_data, x='State', y='Probability')
    
    st.metric("Stable Confidence", f"{st.session_state.probs[0]:.1%}")
    st.metric("Unstable Confidence", f"{st.session_state.probs[1]:.1%}")

# Show the log of events
with st.expander("View Event Log"):
    st.write(st.session_state.history)
