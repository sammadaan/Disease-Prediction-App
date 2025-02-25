import streamlit as st
import joblib
import numpy as np
import requests
import json
from streamlit_lottie import st_lottie

# Function to Load Lottie Animations
def load_lottie(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error: Unable to fetch Lottie animation. Status code: {response.status_code}")
            return None
        return response.json()  # Ensure JSON response is properly returned
    except requests.exceptions.RequestException as e:
        print(f"Request Exception: {e}")
        return None
import joblib
import os

# Define model directory (use relative paths)
model_dir = os.path.join(os.path.dirname(__file__), "Models")

# Load Models
models = {
    "Diabetes": joblib.load(os.path.join(model_dir, "diabetes_rf_model.pkl")),
    "Heart Disease": joblib.load(os.path.join(model_dir, "heart_disease_rf_model.pkl")),
    "Hypothyroid": joblib.load(os.path.join(model_dir, "hypothyroid_rf_model.pkl")),
    "Lung Cancer": joblib.load(os.path.join(model_dir, "lung_cancer_svc_model.pkl")),
    "Parkinson’s": joblib.load(os.path.join(model_dir, "parkinsons_random_forest.pkl")),
}


# Load Animations
loading_animation = load_lottie('https://lottie.host/6f5fbbd6-eda4-48c7-a0d2-52a307f635ee/V5MbdClBCy.json')
medical_animation = load_lottie('https://lottie.host/9b12e3a5-1f3b-4e0e-b9db-2b053cbd01f8/JzZXJrBXvQ.json')
success_animation = load_lottie('https://lottie.host/8a6b409b-93c8-4a20-9e89-7b6d8e82fa1b/rsp92Fq0mA.json')
error_animation = load_lottie('https://lottie.host/646a8c39-2d80-41fc-9c90-40f3cdde94b7/XMciZvgdD8.json')

# Set Page Config
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    body {
        background-image: url('https://wallpaperaccess.com/full/3170189.jpg');
        background-size: cover;
        font-family: Arial, sans-serif;
    }
    .main-container {
        background: rgba(0, 0, 0, 0.8);
        padding: 20px;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #00FFB3;
        text-align: center;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with animation
st.sidebar.title("⚕️ Disease Prediction AI")
st.sidebar.write("This app predicts **5 major diseases** using ML models.")
if medical_animation:
    st_lottie(medical_animation, height=200, key="medical")

# Disease Selection
disease = st.sidebar.selectbox("Select a Disease", list(models.keys()))

# Input Fields
st.write(f"### Enter details for {disease}")

inputs = []
if disease == "Diabetes":
    inputs = [
        st.number_input("Pregnancies", 0, 20),
        st.number_input("Glucose", 0, 200),
        st.number_input("Blood Pressure", 0, 150),
        st.number_input("Skin Thickness", 0, 100),
        st.number_input("Insulin", 0, 900),
        st.number_input("BMI", 0.0, 70.0),
        st.number_input("Diabetes Pedigree", 0.0, 2.5),
        st.number_input("Age", 0, 120)
    ]
elif disease == "Heart Disease":
    inputs = [
        st.number_input("Age", 0, 120),
        st.number_input("Sex (0=Female, 1=Male)", 0, 1),
        st.number_input("Chest Pain Type (0-3)", 0, 3),
        st.number_input("Resting Blood Pressure", 80, 200),
        st.number_input("Cholesterol", 100, 600),
        st.number_input("Fasting Blood Sugar (0/1)", 0, 1),
        st.number_input("Resting ECG (0-2)", 0, 2),
        st.number_input("Max Heart Rate", 60, 220),
        st.number_input("Exercise Angina (0/1)", 0, 1),
        st.number_input("Oldpeak", -2.0, 6.0),
        st.number_input("Slope (0-2)", 0, 2),
        st.number_input("Ca (0-4)", 0, 4),
        st.number_input("Thal (0-3)", 0, 3)
    ]
elif disease == "Hypothyroid":
    inputs = [
        st.number_input("TSH", 0.0, 10.0),
        st.number_input("T3", 0.0, 5.0),
        st.number_input("TT4", 0.0, 200.0),
        st.number_input("T4U", 0.0, 2.0),
        st.number_input("FTI", 0.0, 200.0)
    ]
elif disease == "Lung Cancer":
    inputs = [
        st.number_input("Age", 0, 100),
        st.number_input("Smokes (0/1)", 0, 1),
        st.number_input("Coughing (0/1)", 0, 1),
        st.number_input("Shortness of Breath (0/1)", 0, 1),
        st.number_input("Wheezing (0/1)", 0, 1),
        st.number_input("Swallowing Difficulty (0/1)", 0, 1),
        st.number_input("Chest Pain (0/1)", 0, 1),
        st.number_input("Frequent Cold (0/1)", 0, 1),
        st.number_input("Fatigue (0/1)", 0, 1),
        st.number_input("Weight Loss (0/1)", 0, 1),
        st.number_input("Hoarseness (0/1)", 0, 1),
        st.number_input("Exposure to Pollution (0/1)", 0, 1),
        st.number_input("Exposure to Asbestos (0/1)", 0, 1),
        st.number_input("Family History (0/1)", 0, 1),
        st.number_input("Frequent Pneumonia (0/1)", 0, 1),
        st.number_input("Lung Nodules (0/1)", 0, 1)
    ]
elif disease == "Parkinson’s":
    inputs = [
        st.number_input("MDVP:Fo(Hz)", 50.0, 300.0),
        st.number_input("MDVP:Fhi(Hz)", 50.0, 300.0),
        st.number_input("MDVP:Flo(Hz)", 50.0, 300.0),
        st.number_input("Jitter(%)", 0.0, 0.1),
        st.number_input("Shimmer(dB)", 0.0, 1.0),
        st.number_input("HNR", 0.0, 50.0)
    ]

# Predict Button
if st.button("Predict"):
    if loading_animation:
        st_lottie(loading_animation, height=150, key="loading")
    
    model = models[disease]
    prediction = model.predict(np.array(inputs).reshape(1, -1))

    if prediction[0] == 1:
        if success_animation:
            st_lottie(success_animation, height=200, key="success")
        st.success(f"The model predicts that you **may have {disease}**. Please consult a doctor.")
    else:
        if error_animation:
            st_lottie(error_animation, height=200, key="error")
        st.success(f"The model predicts that you **do NOT have {disease}**. Stay healthy!")
