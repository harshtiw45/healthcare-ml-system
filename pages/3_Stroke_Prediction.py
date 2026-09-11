import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout="wide")
st.title("🧠 Stroke Prediction")

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/stroke_model.pkl")

# ---------------- INPUT UI ----------------
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 45)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    avg_glucose_level = st.number_input("Glucose Level", 50.0, 300.0, 100.0)

with col2:
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children"])
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

# ---------------- RAW INPUT ----------------
input_dict = {
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "gender": gender,
    "work_type": work_type,
    "smoking_status": smoking_status
}

input_df = pd.DataFrame([input_dict])

# ---------------- 🔥 FIX: ONE-HOT ENCODING ----------------
input_encoded = pd.get_dummies(input_df)

# ---------------- ALIGN WITH MODEL ----------------
if hasattr(model, "feature_names_in_"):
    expected_cols = model.feature_names_in_

    aligned_df = pd.DataFrame([[0]*len(expected_cols)], columns=expected_cols)

    for col in input_encoded.columns:
        if col in aligned_df.columns:
            aligned_df[col] = input_encoded[col]

else:
    aligned_df = input_encoded

# ---------------- PREDICT ----------------
if st.button("Predict Stroke Risk"):
    try:
        prediction = model.predict(aligned_df)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Stroke")
        else:
            st.success("✅ Low Risk of Stroke")

    except Exception as e:
        st.error("Prediction failed")
        st.write(e)