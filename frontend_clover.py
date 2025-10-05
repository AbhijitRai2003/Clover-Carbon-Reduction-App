# frontend_clover.py
# CLOVER Frontend - Interactive Streamlit UI
# Run: streamlit run frontend_clover.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from backend_clover import clover_inference

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="CLOVER Demo",
    page_icon="🌱",
    layout="wide"
)

# -------------------- HEADER --------------------
st.title("🌱 CLOVER: Carbon-Aware Inference System")
st.caption("Dynamic model switching based on simulated carbon intensity levels.")

# -------------------- USER INPUT SECTION --------------------
st.markdown("### 💬 Try it yourself:")
user_input = st.text_area("Enter your text query:", "This project is revolutionary and exciting!")

# -------------------- INFERENCE SECTION --------------------
if st.button("⚡ Run Inference"):
    with st.spinner("Running model inference..."):
        time.sleep(1.2)  # just to make the UI feel more natural
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

    # -------------------- VISUAL DASHBOARD --------------------
    st.markdown("### 🌍 Carbon Intensity Dashboard")

    # --- Gauge Chart ---
    carbon_colors = {"LOW": "#32CD32", "MEDIUM": "#FFD700", "HIGH": "#FF6347"}
    intensity_values = {"LOW": 30, "MEDIUM": 60, "HIGH": 90}
    color = carbon_colors.get(carbon_level, "#999999")
    intensity = intensity_values.get(carbon_level, 50)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=intensity,
        number={'suffix': " %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "#32CD32"},
                {'range': [40, 70], 'color': "#FFD700"},
                {'range': [70, 100], 'color': "#FF6347"}
            ],
        },
        title={'text': "Carbon Intensity Level"}
    ))
    st.plotly_chart(gauge, use_container_width=True)

    st.markdown(
        f"<h5 style='color:{color};text-align:center;'>Current Carbon Level: {carbon_level}</h5>",
        unsafe_allow_html=True
    )

    # --- Carbon Trend Simulation Chart ---
    trend_data = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Carbon Intensity (%)": [85, 70, 65, 50, 45, intensity]
    })
    fig = px.bar(
        trend_data,
        x="Month",
        y="Carbon Intensity (%)",
        color="Carbon Intensity (%)",
        color_continuous_scale="RdYlGn_r",
        title="📊 Carbon Intensity Trend Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------- MODEL SUMMARY TABLE --------------------
    st.markdown("### 🧠 Model Selection Summary")
    data = {
        "Model Variant": ["BERT", "DistilBERT", "Quantized DistilBERT"],
        "Carbon Usage": ["High", "Medium", "Low"],
        "Accuracy": ["High", "Medium-High", "Moderate"],
        "Use Case": ["Full performance", "Balanced mode", "Eco mode"]
    }
    df = pd.DataFrame(data)
    st.table(df)

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Developed by Abhijit Rai | Major Project: Carbon Reduction through Adaptive Inference")
