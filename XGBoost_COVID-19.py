import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoost.pkl')

# Define feature names
feature_names = [
    "Glucose", "Urea", "Creatinie", "Sodium", "Postassium",
    "TB", "DB", "ALT", "AST",
    "ALP", "Hemoglobin", "WBC", "Platelet", "CRP", "D-Dimer"
]

# Streamlit user interface
st.title("COVID-19 Predictor")

# Glucose: numerical input
Glucose = st.number_input("Glucose (mg/dL):", min_value=40.0, max_value=500.0, value=150.0)

# Urea: numerical input
Urea = st.number_input("Urea (mg/dL):", min_value=10.0, max_value=350.0, value=50.0)

# Creatinie: numerical input
Creatinie = st.number_input("Creatinie (mg/dL):", min_value=0.3, max_value=20.0, value=1.2)

# Sodium: numerical input
Sodium = st.number_input("Sodium (mmol/L):", min_value=110.0, max_value=190.0, value=140.0)

# Potassium: numerical input
Potassium = st.number_input("Potassium (mmol/L):", min_value=2.0, max_value=6.5, value=4.0)

# TB: numerical input
TB = st.number_input("TB (μmol/L):", min_value=0.2, max_value=17.5, value=1.0)

# DB: numerical input
DB = st.number_input("DB (μmol/L):", min_value=0.1, max_value=11.0, value=0.4)

# ALT: numerical input
ALT = st.number_input("ALT (U/L):", min_value=10.0, max_value=460.0, value=50.0)

# AST: numerical input
AST = st.number_input("AST (U/L):", min_value=1.0, max_value=480.0, value=50.0)

# ALP: numerical input
ALP = st.number_input("ALP (U/L):", min_value=15.0, max_value=500.0, value=100.0)

# Hemoglobin: numerical input
Hemoglobin = st.number_input("Hemoglobin (g/dL):", min_value=0.0, max_value=20.0, value=10.0)

# WBC: numerical input
WBC = st.number_input("WBC (cells/μL):", min_value=1500.0, max_value=35000.0, value=10000.0)

# Platelet: numerical input
Platelet = st.number_input("Platelet (platelets/μL):", min_value=10000.0, max_value=640000.0, value=23000.0)

# CRP: numerical input
CRP = st.number_input("CRP (mg/L):", min_value=0.0, max_value=250.0, value=50.0)

# D-Dimer: numerical input
D_Dimer = st.number_input("D-Dimer (mg/L):", min_value=0.0, max_value=10.0, value=1.0)

# Process inputs and make predictions
feature_values = [Glucose, Urea, Creatinie, Sodium, Potassium, TB, DB, ALT, AST, ALP, Hemoglobin, WBC, Platelet, CRP, D_Dimer]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of COVID-19. "
            f"The model predicts that your probability of having COVID-19 is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you Further test to determine if you have the disease and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of COVID-19. "
            f"The model predicts that your probability of not having COVID-19 is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")