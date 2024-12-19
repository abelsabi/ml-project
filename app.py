import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import requests
import time

# Set Streamlit Page Configurations
st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

# Custom CSS for Styling
st.markdown("""
    <style>
    .main {
        background-color: #121212;
        color: #ffffff;
    }
    .stSlider {
        color: #ff6b6b;
    }
    .css-2trqyj {
        color: #00ff00;
    }
    .css-1cpxqw2 {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title Section
st.title("🩺 Breast Cancer Predictor")
st.write(
    "This project predicts whether a breast tumor is **Benign** or **Malignant** "
    "using Machine Learning (ML). Adjust the tumor parameters to explore the model's predictions."
)

# Sidebar for Features Input
st.sidebar.header("🔧 Input Tumor Features")
st.sidebar.write("Adjust the sliders to simulate different tumor characteristics.")

# Feature Inputs Using Sliders
feature_dict = {
    "Radius Mean": st.sidebar.slider("Radius (mean)", 0.0, 30.0, 10.0),
    "Texture Mean": st.sidebar.slider("Texture (mean)", 0.0, 40.0, 15.0),
    "Perimeter Mean": st.sidebar.slider("Perimeter (mean)", 0.0, 200.0, 80.0),
    "Area Mean": st.sidebar.slider("Area (mean)", 0.0, 2500.0, 1000.0),
    "Smoothness Mean": st.sidebar.slider("Smoothness (mean)", 0.0, 1.0, 0.5),
    "Concavity Mean": st.sidebar.slider("Concavity (mean)", 0.0, 1.0, 0.2),
    "Symmetry Mean": st.sidebar.slider("Symmetry (mean)", 0.0, 1.0, 0.3),
}

# Convert Features to DataFrame
input_data = pd.DataFrame([feature_dict])

# Feature Visualization Section
st.subheader("📊 Tumor Feature Visualization")
col1, col2 = st.columns([2, 1])

# Radar Chart for Features
fig = go.Figure()

categories = list(feature_dict.keys())
values = list(feature_dict.values())
values += values[:1]  # Repeat first value to close the radar chart

fig.add_trace(go.Scatterpolar(
    r=values,
    theta=categories + [categories[0]],
    fill='toself',
    name='Input Features',
    line_color='#FF6B6B'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, max(values) + 1])
    ),
    showlegend=False,
    template="plotly_dark"
)

col1.plotly_chart(fig, use_container_width=True)

# Feature Importance Visualization
st.subheader("🔥 Feature Importance (Example)")
feature_importance = pd.Series([0.2, 0.15, 0.1, 0.25, 0.1, 0.1, 0.1],
                               index=categories)
fig_bar = px.bar(feature_importance, x=feature_importance.values, y=feature_importance.index,
                 orientation='h', color=feature_importance.values, color_continuous_scale='reds',
                 title="Feature Importance")
st.plotly_chart(fig_bar, use_container_width=True)

# Prediction Results Simulation
st.subheader("🔍 Prediction Results")

# Placeholder for Results
result_placeholder = st.empty()

# Simulate Loading Time
with st.spinner("🔎 Analyzing Tumor Parameters..."):
    time.sleep(2)

# Simulated Prediction and Probabilities
prediction = np.random.choice(["Benign", "Malignant"], p=[0.7, 0.3])
benign_prob = round(np.random.uniform(0.6, 0.9), 2)
malignant_prob = round(1 - benign_prob, 2)

# Display Prediction
if prediction == "Benign":
    result_placeholder.success("✅ The tumor is predicted to be **Benign**.")
else:
    result_placeholder.error("⚠️ The tumor is predicted to be **Malignant**.")

# Progress Indicators for Probabilities
st.write("### 📈 Prediction Probabilities")
st.progress(int(benign_prob * 100), text=f"Probability of being Benign: {benign_prob * 100}%")
st.progress(int(malignant_prob * 100), text=f"Probability of being Malignant: {malignant_prob * 100}%")

# Footer
st.markdown("---")
st.caption("🧬 This Breast Cancer Predictor was built using **Streamlit** and **Python**. Explore and simulate tumor parameters for analysis.")
