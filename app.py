import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("sugarsense_svm_model.pkl")

# Page configuration
st.set_page_config(
    page_title="SugarSense",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 SugarSense")
st.subheader("Early Diabetes Risk Screening")

st.write(
    "Enter the patient's information below to estimate diabetes risk. "
    "This tool is intended for educational and preliminary screening purposes "
    "and is not a medical diagnosis."
)

# Patient inputs
pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0.0,
    max_value=300.0,
    value=120.0
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

insulin = st.number_input(
    "Insulin",
    min_value=0.0,
    max_value=1000.0,
    value=80.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

# Engineered features
obesity_flag = int(bmi >= 30)
age_risk_flag = int(age >= 40)
glucose_bmi = glucose * bmi

# Create input dataframe
input_data = pd.DataFrame([{
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": diabetes_pedigree,
    "Age": age,
    "ObesityFlag": obesity_flag,
    "AgeRiskFlag": age_risk_flag,
    "Glucose_BMI": glucose_bmi
}])

# Prediction
if st.button("Check Diabetes Risk"):

    # Convert problematic zeros to missing values
    zero_as_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    input_data[zero_as_missing] = input_data[
        zero_as_missing
    ].replace(0, float("nan"))

    probability = model.predict_proba(input_data)[0][1]

    # Tuned threshold from the project
    threshold = 0.23

    if probability >= threshold:
        st.warning("Higher Diabetes Risk")
    else:
        st.success("Lower Diabetes Risk")

    st.write(f"Estimated probability: {probability:.2%}")

    st.caption(
        "This result is a machine-learning screening estimate and "
        "should not be used as a medical diagnosis."
    )