import pandas as pd
import numpy as np
import joblib
import folium
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import folium_static
from tf_keras.models import load_model
import traceback

# Page Configuration
st.set_page_config(page_title="🚗 Digital Twin Risk Predictor", layout="wide")

# Load Model & Preprocessors (FIXED FILENAMES)
try:
    model = load_model("model.h5")
    scaler = joblib.load("scaler_no_interaction.pkl")
    le_risk = joblib.load("le_risk_no_interaction.pkl")
except FileNotFoundError as e:
    st.error(f"Model or encoders are missing: {e}")
    st.warning("Make sure you run train_model.py first!")
    st.stop()

# Header
st.markdown("""
    <style>
        .big-title { font-size: 32px; font-weight: bold; color: #2E8B57; text-align: center; }
        .stButton>button { background-color: #FF5733; color: white; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🚗 Digital Twin for Vehicle Insurance</p>', unsafe_allow_html=True)
st.subheader("Predict your **driving risk level** and explore accident-prone zones.")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Risk Prediction", "🗺️ Accident Risk Map", "📈 Model Performance"])

# 🚗 **Risk Prediction Tab**
with tab1:
    st.header("📌 Enter Driving Data")

    # FIXED INPUTS TO MATCH TRAINING DATA (9 Features)
    col1, col2, col3 = st.columns(3)
    with col1:
        speed = st.number_input("Speed (km/h)", min_value=0.0, max_value=250.0, value=60.0)
        braking = st.number_input("Braking Severity (0-5)", min_value=0.0, max_value=5.0, value=1.0)
        acceleration = st.number_input("Acceleration Severity (0-5)", min_value=0.0, max_value=5.0, value=1.0)
    with col2:
        engine_rpm = st.number_input("Engine RPM", min_value=0.0, max_value=10000.0, value=2500.0)
        throttle_position = st.number_input("Throttle Position (%)", min_value=0.0, max_value=100.0, value=50.0)
        fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=0.0, max_value=30.0, value=8.0)
    with col3:
        engine_load = st.number_input("Engine Load (%)", min_value=0.0, max_value=100.0, value=40.0)
        mileage = st.number_input("Mileage (Total km)", min_value=0.0, max_value=500000.0, value=50000.0)
        past_accidents = st.number_input("Past Accidents", min_value=0, max_value=20, value=0)

    if st.button("🚀 Predict Risk Level"):
        try:
            # Prepare input in the EXACT order as training
            user_input = np.array([[speed, braking, acceleration, engine_rpm, throttle_position, 
                                    fuel_consumption, engine_load, mileage, past_accidents]])
            
            user_input_scaled = scaler.transform(user_input)

            # Predict Risk
            risk_pred_proba = model.predict(user_input_scaled)
            risk_index = np.argmax(risk_pred_proba, axis=1)[0]
            
            # Apply threshold to force High-risk predictions (Same logic as training code)
            if risk_pred_proba[0][2] > 0.3:
                risk_index = 2

            risk_prediction = le_risk.inverse_transform([risk_index])[0]

            # Risk Level Colors (Updated to match 'Moderate' instead of 'Medium')
            risk_levels = {"Low": "🟢 Low", "Moderate": "🟠 Moderate", "High": "🔴 High"}
            risk_meter = {"Low": 30, "Moderate": 60, "High": 90}

            # Display Risk Level
            st.markdown(f"### 🚨 **Predicted Risk Level: {risk_levels.get(risk_prediction, risk_prediction)}**")
            st.progress(risk_meter.get(risk_prediction, 50) / 100)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.text(traceback.format_exc())

# 🗺️ **Accident Risk Map Tab**
with tab2:
    st.header("🌍 Interactive Accident Risk Map")
    
    # Generate sample accident locations
    accident_data = pd.DataFrame({
        "latitude": np.random.uniform(12.90, 13.10, 50),
        "longitude": np.random.uniform(77.50, 77.70, 50),
        "risk": np.random.choice(["Low", "Moderate", "High"], 50)
    })

    # Map Initialization
    m = folium.Map(location=[13.0, 77.6], zoom_start=12)
    
    for _, row in accident_data.iterrows():
        color = {"Low": "green", "Moderate": "orange", "High": "red"}.get(row["risk"], "blue")
        folium.Marker([row["latitude"], row["longitude"]],
                      popup=f"Risk: {row['risk']}",
                      icon=folium.Icon(color=color)).add_to(m)

    folium_static(m)

# 📈 **Model Performance Tab**
with tab3:
    st.header("📊 Model Performance Metrics")
    st.write("This tab displays static placeholder data for visual presentation.")

    # Display Confusion Matrix (Simulated for UI)
    labels = ["Low", "Moderate", "High"]
    cm = np.array([[50, 5, 2], [4, 40, 6], [3, 8, 42]])  
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

# Sidebar AI Chat
st.sidebar.title("🤖 AI Assistant")
st.sidebar.markdown("Ask me about the risk model!")

chat_input = st.sidebar.text_input("Type a question:")
if chat_input:
    response = {
        "What does the model predict?": "It predicts driving risk as Low, Moderate, or High.",
        "How does it work?": "It analyzes speed, braking, acceleration, engine metrics, and past accidents.",
        "How accurate is it?": "The model's accuracy depends on the underlying training data distribution."
    }.get(chat_input, "I'm not sure, but you can try rephrasing!")
    st.sidebar.write(response)
