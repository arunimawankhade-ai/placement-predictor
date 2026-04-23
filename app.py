import streamlit as st
import pickle
import numpy as np

# Page settings
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎯",
    layout="centered"
)

# Custom CSS styling
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

h1 {
    color: #ff4b4b;
    text-align: center;
    font-family: 'Poppins', sans-serif;
}

.stNumberInput input {
    background-color: #1c1f26;
    color: white;
    border-radius: 10px;
}

.stButton button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))  

# Title
st.markdown("<h1>🚀 Placement Predictor</h1>", unsafe_allow_html=True)

# Inputs
st.write("### Enter your details 👇")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input("📘 CGPA", min_value=0.0, max_value=10.0)

with col2:
    iq = st.number_input("🧠 IQ", min_value=50.0, max_value=200.0)

# Prediction
if st.button("Predict"):
    input_data = np.array([[cgpa, iq]])
    input_scaled = scaler.transform(input_data)

    proba = model.predict_proba(input_scaled)[0][1]

    if proba > 0.7:
        st.markdown(f"<h3 style='color:lightgreen;'>✅ Placed ({proba*100:.2f}%)</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:red;'>❌ Not Placed ({proba*100:.2f}%)</h3>", unsafe_allow_html=True)