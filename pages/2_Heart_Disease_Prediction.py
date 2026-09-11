import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout="wide")
st.title("❤️ Heart Disease Prediction")

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/heart_model.pkl")

# ---------------- UI ----------------
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)

with col2:
    chol = st.number_input("Cholesterol", 100, 600, 200)
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)

# ---------------- ENCODE ----------------
sex_val = 1 if sex == "Male" else 0

input_dict = {
    "age": age,
    "sex": sex_val,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "thalach": thalach
}

input_df = pd.DataFrame([input_dict])

# ---------------- 🔥 FIX: ALIGN FEATURES ----------------
if hasattr(model, "feature_names_in_"):
    expected_cols = model.feature_names_in_

    # create 1 row with all zeros
    aligned_df = pd.DataFrame([[0]*len(expected_cols)], columns=expected_cols)

    # fill values where matching
    for col in input_df.columns:
        if col in aligned_df.columns:
            aligned_df.at[0, col] = input_df[col].values[0]
else:
    aligned_df = input_df

# ---------------- PREDICT ----------------
if st.button("Predict Heart Disease"):
    try:
        prediction = model.predict(aligned_df)[0]

        # probability (if supported)
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(aligned_df)[0][1]

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

        if prob is not None:
            st.info(f"Confidence: {prob*100:.2f}%")

    except Exception as e:
        st.error("Prediction failed")
        st.write(e)