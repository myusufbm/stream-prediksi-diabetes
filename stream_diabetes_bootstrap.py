# main.py
"""
Streamlit app for Diabetes Prediction using trained pipeline
Author : Yusuf Ghozali | NIM 2221400110

How to run locally:
    streamlit run main.py
"""

import pickle
import streamlit as st
import pandas as pd
from pathlib import Path

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------
MODEL_FILE = Path("diabetes_svm_pipeline.pkl")  # make sure the file is in the same folder
APP_TITLE = "Diabetes Prediction"
APP_SUBTITLE = "Early detection tool powered by Machine Learning"

# ------------------------------------------------------
# Load model once and cache it
# ------------------------------------------------------
@st.cache_resource(show_spinner="Loading ML model...")
def load_model(model_path: Path):
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()
    with model_path.open("rb") as f:
        model = pickle.load(f)
    return model

model = load_model(MODEL_FILE)

# ------------------------------------------------------
# Inject Bootstrap 5 CSS (CDN)
# ------------------------------------------------------
BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
st.markdown(
    f"<link href='{BOOTSTRAP_CSS}' rel='stylesheet'>",
    unsafe_allow_html=True,
)

# Custom CSS tweaks for Streamlit layout
CUSTOM_CSS = """
<style>
    .stApp {background-color: #f8f9fa;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------
# Page Header (Bootstrap Jumbotron style)
# ------------------------------------------------------
st.markdown(
    f"""
    <div class='py-5 text-center container'>
        <h1 class='display-4 fw-bold'>{APP_TITLE}</h1>
        <p class='lead mb-4'>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------
# Sidebar for user input
# ------------------------------------------------------
with st.sidebar:
    st.header("Patient Data Input")

    Pregnancies = st.number_input("Pregnancies", min_value=0, value=0, step=1)
    Glucose = st.number_input("Glucose", min_value=0.0, value=120.0)
    BloodPressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
    Insulin = st.number_input("Insulin", min_value=0.0, value=79.0)
    BMI = st.number_input("BMI", min_value=0.0, value=25.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
    Age = st.number_input("Age", min_value=1, value=33)

    submitted = st.button("Predict", type="primary")

# ------------------------------------------------------
# Prediction Logic
# ------------------------------------------------------
if submitted:
    # Prepare single-row DataFrame with correct column order
    input_data = pd.DataFrame({
        "Pregnancies": [Pregnancies],
        "Glucose": [Glucose],
        "BloodPressure": [BloodPressure],
        "SkinThickness": [SkinThickness],
        "Insulin": [Insulin],
        "BMI": [BMI],
        "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
        "Age": [Age],
    })

    # Predict using pipeline (handles preprocessing internally)
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    # Show result in Bootstrap alert box
    if pred == 1:
        st.markdown(
            f"""
            <div class='alert alert-danger text-center fw-bold' role='alert'>
                Positive for Diabetes<br>
                <span class='fw-normal'>Risk Probability: {prob:.2%}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='alert alert-success text-center fw-bold' role='alert'>
                Negative for Diabetes<br>
                <span class='fw-normal'>Confidence: {1-prob:.2% if prob is not None else 'N/A'}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ------------------------------------------------------
# Footer with Name & NIM
# ------------------------------------------------------
st.markdown(
    """
    <footer class='mt-5 py-3 bg-dark'>
        <div class='container text-center'>
            <span class='text-muted'>Created by Yusuf Ghozali &nbsp;|&nbsp; NIM 2221400110</span>
        </div>
    </footer>
    """,
    unsafe_allow_html=True,
)
