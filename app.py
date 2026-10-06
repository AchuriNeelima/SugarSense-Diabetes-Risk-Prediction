import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="SugarSense",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #e0f2fe, #f0fdf4);
        margin-bottom: 1.5rem;
        border: 1px solid #dbeafe;
    }

    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: 0.3rem;
        color: #0f172a;
    }

    .hero p {
        color: #475569;
        font-size: 1.05rem;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .section-text {
        color: #64748b;
        margin-bottom: 1.2rem;
    }

    .info-card {
        padding: 1.2rem;
        border-radius: 16px;
        background-color: white;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    .risk-card {
        padding: 1.5rem;
        border-radius: 18px;
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        margin-top: 1rem;
    }

    .small-text {
        color: #64748b;
        font-size: 0.9rem;
    }

    .disclaimer {
        padding: 1rem;
        border-radius: 12px;
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
        margin-top: 1.5rem;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model = joblib.load("sugarsense_svm_model.pkl")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🩺 SugarSense</h1>
    <p>
        Diabetes Risk Screening using Supervised Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("SugarSense")
st.sidebar.caption("ML-based diabetes risk screening")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Risk Screening",
        "Data Insights",
        "Model Performance",
        "Feature Importance",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "For educational and preliminary screening purposes only."
)


# =========================================================
# OVERVIEW
# =========================================================
if page == "Overview":

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        'SugarSense uses supervised machine learning to estimate diabetes '
        'risk from patient measurements. The application is designed as '
        'a screening and educational aid, not as a medical diagnosis.'
        '</div>',
        unsafe_allow_html=True
    )

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Patient Records", "768")

    with col2:
        st.metric("Input Features", "8")

    with col3:
        st.metric("Final Accuracy", "74.03%")

    with col4:
        st.metric("ROC-AUC", "80.67%")

    st.markdown("---")

    # Dataset distribution
    st.subheader("Dataset Outcome Distribution")

    outcome_data = pd.DataFrame(
        {
            "Outcome": [
                "No Diabetes",
                "Diabetes"
            ],
            "Patients": [
                500,
                268
            ]
        }
    )

    st.bar_chart(
        outcome_data.set_index("Outcome")
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>Dataset</h4>
            <p>
            The project uses 768 patient records with 8 clinical input
            features and one target variable named Outcome.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>Machine Learning Workflow</h4>
            <p>
            Data audit → EDA → Feature engineering → Train/test split →
            Model comparison → Hyperparameter tuning → Evaluation →
            Threshold optimization.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# RISK SCREENING
# =========================================================
elif page == "Risk Screening":

    st.markdown(
        '<div class="section-title">Diabetes Risk Screening</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        'Enter the patient measurements below to generate an ML-based '
        'risk estimate.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Patient Measurements")

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0.0,
            max_value=300.0,
            value=120.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=0.0,
            max_value=100.0,
            value=20.0
        )

    with col2:

        st.subheader("Additional Attributes")

        insulin = st.number_input(
            "Insulin (μU/mL)",
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
            value=0.50
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    st.markdown("---")

    # Engineered features
    obesity_flag = int(bmi >= 30)
    age_risk_flag = int(age >= 40)
    glucose_bmi = glucose * bmi

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

    if st.button(
        "Check Diabetes Risk",
        type="primary",
        use_container_width=True
    ):

        zero_as_missing = [
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI"
        ]

        input_data[zero_as_missing] = input_data[
            zero_as_missing
        ].replace(0, np.nan)

        probability = model.predict_proba(
            input_data
        )[0][1]

        threshold = 0.23

        st.markdown("---")
        st.subheader("Risk Assessment")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Estimated Probability",
                f"{probability:.2%}"
            )

            st.progress(
                min(float(probability), 1.0)
            )

        with result_col2:

            if probability >= threshold:
                st.error("Higher Diabetes Risk")
            else:
                st.success("Lower Diabetes Risk")

            st.caption(
                f"Decision threshold: {threshold:.2f}"
            )

        st.markdown("""
        <div class="disclaimer">
        <strong>Important:</strong> This is a machine-learning screening
        estimate for educational and preliminary purposes. It is not a
        medical diagnosis and should not replace professional medical
        evaluation.
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# DATA INSIGHTS
# =========================================================
elif page == "Data Insights":

    st.markdown(
        '<div class="section-title">Data Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        'Key findings from the dataset audit and exploratory analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    # Problematic zero values
    st.subheader("Problematic Zero Values")

    zero_data = pd.DataFrame(
        {
            "Feature": [
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI"
            ],
            "Percentage": [
                0.65,
                4.56,
                29.56,
                48.70,
                1.43
            ]
        }
    )

    st.bar_chart(
        zero_data.set_index("Feature")
    )

    st.caption(
        "Zero values in these clinical measurements were treated as "
        "missing values and handled using median imputation inside the "
        "machine-learning pipeline."
    )

    st.subheader("Dataset Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Total Records", "768")

    with summary_col2:
        st.metric("No Diabetes", "500")

    with summary_col3:
        st.metric("Diabetes", "268")

    st.markdown("---")

    st.subheader("Feature Engineering")

    feature_data = pd.DataFrame(
        {
            "Engineered Feature": [
                "ObesityFlag",
                "AgeRiskFlag",
                "Glucose_BMI"
            ],
            "Description": [
                "BMI >= 30",
                "Age >= 40",
                "Glucose × BMI"
            ]
        }
    )

    st.dataframe(
        feature_data,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        'Final evaluation results for the selected SVM model.'
        '</div>',
        unsafe_allow_html=True
    )

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

    with metric_col1:
        st.metric("Accuracy", "74.03%")

    with metric_col2:
        st.metric("Precision", "60.61%")

    with metric_col3:
        st.metric("Recall", "74.07%")

    with metric_col4:
        st.metric("F1 Score", "66.67%")

    with metric_col5:
        st.metric("ROC-AUC", "80.67%")

    st.markdown("---")

    st.subheader("Final Model Metrics")

    metrics_data = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],
            "Score": [
                74.03,
                60.61,
                74.07,
                66.67,
                80.67
            ]
        }
    )

    st.bar_chart(
        metrics_data.set_index("Metric")
    )

    st.markdown("---")

    st.subheader("Threshold Optimization")

    threshold_data = pd.DataFrame(
        {
            "Metric": [
                "Precision",
                "Recall",
                "F1 Score"
            ],
            "Default 0.50": [
                63.64,
                51.85,
                57.14
            ],
            "Tuned 0.23": [
                53.49,
                85.19,
                65.71
            ]
        }
    )

    st.dataframe(
        threshold_data,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The threshold was reduced to approximately 0.23 to prioritize "
        "higher recall. This increased test recall from 51.85% to 85.19%, "
        "while precision decreased from 63.64% to 53.49%."
    )

    st.markdown("---")

    st.subheader("Selected Model")

    st.success(
        "SVM was the best tuned model based on cross-validation ROC-AUC."
    )

    st.write(
        "Best tuned cross-validation ROC-AUC: **0.8447**"
    )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
elif page == "Feature Importance":

    st.markdown(
        '<div class="section-title">Feature Importance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        'Permutation importance shows how much model ROC-AUC changes '
        'when individual features are shuffled.'
        '</div>',
        unsafe_allow_html=True
    )

    importance_data = pd.DataFrame(
        {
            "Feature": [
                "Glucose",
                "Pregnancies",
                "BMI",
                "DiabetesPedigreeFunction",
                "ObesityFlag",
                "Glucose_BMI",
                "SkinThickness",
                "Age",
                "AgeRiskFlag",
                "Insulin",
                "BloodPressure"
            ],
            "Importance": [
                0.172333,
                0.021778,
                0.017500,
                0.016241,
                0.011278,
                0.006074,
                0.000944,
                0.000722,
                0.000463,
                -0.000037,
                -0.003870
            ]
        }
    )

    st.bar_chart(
        importance_data.set_index("Feature")
    )

    st.subheader("Main Observation")

    st.info(
        "Glucose was the strongest feature in the final model. "
        "Pregnancies, BMI and DiabetesPedigreeFunction also contributed "
        "to the model's predictions."
    )

    st.caption(
        "A slightly negative permutation importance does not mean that "
        "a feature causes lower diabetes risk. It means that shuffling "
        "that feature slightly increased ROC-AUC in this experiment."
    )


# =========================================================
# ABOUT PROJECT
# =========================================================
elif page == "About Project":

    st.markdown(
        '<div class="section-title">About SugarSense</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### Project Goal

    SugarSense is a supervised machine-learning project designed to
    estimate diabetes risk using patient measurements.

    ### Machine Learning Workflow

    1. Data audit
    2. Exploratory data analysis
    3. Feature engineering
    4. Stratified train-test split
    5. Machine-learning pipelines
    6. Cross-validation
    7. Hyperparameter tuning
    8. Final model evaluation
    9. Threshold optimization
    10. Permutation feature importance

    ### Models Evaluated

    - Logistic Regression
    - KNN
    - Decision Tree
    - Random Forest
    - XGBoost
    - SVM

    ### Final Model

    The selected model was a tuned Support Vector Machine (SVM).

    ### Limitations

    The dataset represents a specific population and may not generalize
    equally to all populations. Therefore, the system should be treated
    as a screening and educational aid rather than a medical diagnostic
    system.
    """)

    st.markdown("""
    <div class="disclaimer">
    <strong>Medical Disclaimer:</strong> SugarSense is not a doctor,
    diagnostic device, or substitute for professional medical advice.
    Always consult a qualified healthcare professional for medical
    decisions.
    </div>
    """, unsafe_allow_html=True)
