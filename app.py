import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import os
from streamlit_lottie import st_lottie

# Function to load Lottie animations
def load_lottie(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return None
        return response.json()
    except requests.exceptions.RequestException:
        return None

# Define model directory
model_dir = os.path.join(os.path.dirname(__file__), "Models")

# Load Models
models = {
    "Diabetes": joblib.load(os.path.join(model_dir, "diabetes_rf_model.pkl")),
    "Heart Disease": joblib.load(os.path.join(model_dir, "heart_disease_rf_model.pkl")),
    "Hypothyroid": joblib.load(os.path.join(model_dir, "hypothyroid_rf_model.pkl")),
    "Lung Cancer": joblib.load(os.path.join(model_dir, "lung_cancer_svc_model.pkl")),
    "Parkinson's": joblib.load(os.path.join(model_dir, "parkinsons_random_forest.pkl")),
}

# Define expected columns for each disease
expected_columns = {
    "Diabetes": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
    "Heart Disease": ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FBS', 'RestECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'Slope', 'Ca', 'Thal'],
    "Hypothyroid": ['TSH', 'T3', 'TT4', 'T4U', 'FTI'],
    "Lung Cancer": ['Age', 'Smokes', 'Coughing', 'ShortBreath', 'Wheezing', 'SwallowingDiff', 'ChestPain', 'FrequentCold', 'Fatigue', 'WeightLoss', 'Hoarseness', 'Pollution', 'Asbestos', 'FamilyHistory', 'Pneumonia', 'LungNodules'],
    "Parkinson's": ['Fo', 'Fhi', 'Flo', 'Jitter', 'Shimmer', 'HNR']
}

# Load animations
medical_animation = load_lottie('https://lottie.host/9b12e3a5-1f3b-4e0e-b9db-2b053cbd01f8/JzZXJrBXvQ.json')
success_animation = load_lottie('https://lottie.host/8a6b409b-93c8-4a20-9e89-7b6d8e82fa1b/rsp92Fq0mA.json')
error_animation = load_lottie('https://lottie.host/646a8c39-2d80-41fc-9c90-40f3cdde94b7/XMciZvgdD8.json')

# Page config
st.set_page_config(page_title="Disease AI", page_icon="🧬", layout="centered")

# Sidebar
with st.sidebar:
    st.title("🧠 Health Predictor")
    st.write("Predict 5 Major Diseases")
    if medical_animation:
        st_lottie(medical_animation, height=200)

# Disease selection
disease = st.selectbox("Select Disease", list(models.keys()))
model = models[disease]
columns = expected_columns[disease]

# Collect inputs
st.subheader(f"Enter Details for {disease}")
user_input = []
for col in columns:
    val = st.number_input(f"{col}", value=0.0 if '.' in str(col) else 0)
    user_input.append(0 if val is None else val)

# Predict button
if st.button("Predict"):
    if None in user_input or len(user_input) != len(columns):
        st.warning("Please complete all fields.")
    else:
        try:
            user_input = [float(val) for val in user_input]
            input_df = pd.DataFrame([user_input], columns=columns)
            prediction = model.predict(input_df)[0]
            if prediction == 1:
                if success_animation:
                    st_lottie(success_animation, height=150)
                st.error(f"⚠️ You may have {disease}. Consult a doctor.")
            else:
                if error_animation:
                    st_lottie(error_animation, height=150)
                st.success(f"✅ You do NOT have {disease}. Stay healthy!")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
