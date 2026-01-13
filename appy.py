import streamlit as st

# 1. Setup the browser tab title and icon
st.set_page_config(page_title="Forecasting Lab 001", page_icon="📈")

# 2. Main Page Content
st.title("🚀 Forecaster's Hello World")

st.write("""
Welcome to the **Applied Statistics** Forecasting tool. 
This app is currently in 'Phase 1: Deployment Test'.
""")

# 3. Interactive 'Vibe' Check
name = st.text_input("Enter your Forecaster Name:", "Student")
vibe_score = st.slider("What is the current 'Market Vibe'?", 0, 100, 50)

if st.button("Submit Initial Forecast"):
    st.success(f"Log received for {name}!")
    st.write(f"Current Vibe Level: **{vibe_score}%**")
    st.balloons()