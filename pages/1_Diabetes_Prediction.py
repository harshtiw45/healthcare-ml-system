import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("🩸 Diabetes Prediction")
st.write("Predict diabetes risk using medical parameters.")

# ---------------- LOAD MODEL ----------------

model = joblib.load("models/diabetes_model.pkl")

# Expected feature order (from training)
MODEL_FEATURES = [
"age",
"hypertension",
"heart_disease",
"bmi",
"HbA1c_level",
"blood_glucose_level",
"gender_Female",
"gender_Male",
"gender_Other",
"smoking_history_No Info",
"smoking_history_current",
"smoking_history_ever",
"smoking_history_former",
"smoking_history_never",
"smoking_history_not current"
]

# ---------------- INPUT UI ----------------

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    blood_glucose = st.number_input(
        "Blood Glucose Level",
        min_value=0.0,
        max_value=400.0,
        value=100.0
    )

    hba1c = st.number_input(
        "HbA1c Level",
        min_value=0.0,
        max_value=15.0,
        value=5.5
    )


with col2:

    hypertension = st.selectbox(
        "Hypertension",
        [0,1]
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0,1]
    )

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    smoking_history = st.selectbox(
        "Smoking History",
        [
            "never",
            "former",
            "current",
            "not current",
            "ever",
            "No Info"
        ]
    )

# ---------------- ENCODING ----------------

input_dict = {

"age": age,
"hypertension": hypertension,
"heart_disease": heart_disease,
"bmi": bmi,
"HbA1c_level": hba1c,
"blood_glucose_level": blood_glucose,

"gender_Female": 1 if gender=="Female" else 0,
"gender_Male": 1 if gender=="Male" else 0,
"gender_Other": 0,

"smoking_history_No Info": 1 if smoking_history=="No Info" else 0,
"smoking_history_current": 1 if smoking_history=="current" else 0,
"smoking_history_ever": 1 if smoking_history=="ever" else 0,
"smoking_history_former": 1 if smoking_history=="former" else 0,
"smoking_history_never": 1 if smoking_history=="never" else 0,
"smoking_history_not current": 1 if smoking_history=="not current" else 0

}

# Convert to dataframe
input_data = pd.DataFrame([input_dict])

# Ensure exact feature order
input_data = input_data.reindex(columns=MODEL_FEATURES, fill_value=0)

# ---------------- PREDICTION ----------------

if st.button("Predict Risk"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1] * 100

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠ High Risk of Diabetes")

    else:

        st.success("✔ Low Risk of Diabetes")

    # ---------------- GAUGE CHART ----------------

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': "Diabetes Risk Score"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "white"},
            'steps': [
                {'range': [0,30], 'color': "green"},
                {'range': [30,60], 'color': "orange"},
                {'range': [60,100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.write(f"AI Confidence: {round(probability,2)}%")