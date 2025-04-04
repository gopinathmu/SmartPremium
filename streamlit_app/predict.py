import streamlit as st
import joblib
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "xgb_model.pkl")

try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}")
    st.stop()

st.title("SmartPremium: Insurance Cost Prediction")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
income = st.number_input("Annual Income", min_value=1000, max_value=200000, value=30000)
dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
health_score = st.number_input("Health Score", min_value=0.0, max_value=100.0, value=50.0)
claims = st.number_input("Previous Claims", min_value=0, max_value=10, value=1)
vehicle_age = st.number_input("Vehicle Age", min_value=0, max_value=20, value=5)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
insurance_duration = st.number_input("Insurance Duration (years)", min_value=1, max_value=10, value=5)
smoking_status = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])

gender = 1 if gender == "Male" else 0
smoking_status = 1 if smoking_status == "Smoker" else 0

extra_features = np.zeros(18)  # Add placeholder values for missing features

input_features = np.concatenate(([age, gender, income, dependents, health_score, claims,
                                  vehicle_age, credit_score, insurance_duration, smoking_status], extra_features))

input_features = input_features.reshape(1, -1)  # Ensure 2D array shape

if st.button("Predict Premium"):
    prediction = model.predict(input_features)
    st.success(f"Predicted Insurance Premium: ₹{prediction[0]:,.2f}")
