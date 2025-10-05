# frontend_clover.py
# CLOVER Frontend - Streamlit UI with Dashboard
# Run: streamlit run frontend_clover.py

import streamlit as st
import pandas as pd
import time
from backend_clover import clover_inference

st.set_page_config(page_title="CLOVER Demo", page_icon="🌱", layout="wide")

# --- Header ---
st.title("🌱 CLOVER: Carbon-Aware Inference System")
st.caption("Dynamic model switching based on simulated carbon intensity levels.")

# --- Input Section ---
st.markdown("### 💬 Try it yourself:")
user_input = st.text_area("Enter your text query:", "This project is revolutionary and exciting!")

# --- Inference ---
if st.button("⚡ Run Inference"):
    with st.spinner("Running model inference..."):
        time.sleep(1)  # just to make UI feel responsive
        model_used, result, carbon_level = clover_inference(user_input)

    st.success("✅ Inference Completed!")
    st.subheader("🔎 Inference Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Carbon Intensity", carbon_level)
    with col2:
        st.metric("Model Used", model_used)
    with col3:
        st.metric("Model Output", result)

    # --- Carbon Level Gauge ---
    st.markdown("### 🌍 Carbon Intensity Dashboard")

    carbon_colors = {"LOW": "#32CD32", "MEDIUM": "#FFD700", "HIGH": "#FF6347"}
    color = carbon_colors.get(carbon_level, "#999999")
    intensity_value = {"LOW": 30, "MEDIUM": 60, "HIGH": 90}[carbon_level]

    st.progress(intensity_value / 100)
    st.markdown(f"<h5 style='color:{color};text-align:center;'>Current Carbon Level: {carbon_level}</h5>", unsafe_allow_html=True)

    # --- Model Summary Table ---
    data = {
        "Model Variant": ["BERT", "DistilBERT", "Quantized DistilBERT"],
        "Carbon Usage": ["High", "Medium", "Low"],
        "Accuracy": ["High", "Medium-High", "Moderate"],
        "Use Case": ["Full performance", "Balanced mode", "Eco mode"]
    }

    st.markdown("### 🧠 Model Selection Summary")
    df = pd.DataFrame(data)
    st.table(df)

# --- Footer ---
st.markdown("---")
st.caption("Developed by Abhijit Rai | Major Project: Carbon Reduction through Adaptive Inference")
