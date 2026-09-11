import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Healthcare Intelligence", layout="wide")

# ---------------- GENERIC ACCURACY FUNCTION ----------------

@st.cache_data
def get_real_accuracy(model_path, dataset_path, target_options):
    try:
        model = joblib.load(model_path)
        df = pd.read_csv(dataset_path)

        # detect target column
        target = None
        for col in target_options:
            if col in df.columns:
                target = col
                break

        if target is None:
            return 0

        X = df.drop(target, axis=1)
        y = df[target]

        # fix cancer labels
        if y.dtype == "object":
            y = y.map({"B": 0, "M": 1})

        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # align features
        if hasattr(model, "feature_names_in_"):
            X_test = X_test.reindex(columns=model.feature_names_in_, fill_value=0)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100
        return round(acc, 2)

    except Exception as e:
        st.error(f"{model_path} error")
        st.write(e)
        return 0


# ---------------- HEART MODEL SPECIAL FIX ----------------

@st.cache_data
def get_heart_accuracy():
    try:
        df = pd.read_csv("datasets/heart.csv")

        target = "HeartDisease" if "HeartDisease" in df.columns else "target"

        X = df.drop(target, axis=1)
        y = df[target]

        # 🔥 IMPORTANT FIX: apply encoding
        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = joblib.load("models/heart_model.pkl")

        # align features
        X_test = X_test.reindex(columns=model.feature_names_in_, fill_value=0)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100
        return round(acc, 2)

    except Exception as e:
        st.error("Heart model error")
        st.write(e)
        return 0


# ---------------- LOAD ACCURACIES ----------------

diabetes_acc = get_real_accuracy(
    "models/diabetes_model.pkl",
    "datasets/diabetes.csv",
    ["Outcome", "diabetes"]
)

heart_acc = get_heart_accuracy()

stroke_acc = get_real_accuracy(
    "models/stroke_model.pkl",
    "datasets/stroke.csv",
    ["stroke"]
)

parkinson_acc = get_real_accuracy(
    "models/parkinsons_model.pkl",
    "datasets/parkinsons.csv",
    ["status"]
)

cancer_acc = get_real_accuracy(
    "models/cancer_model.pkl",
    "datasets/breast_cancer.csv",
    ["diagnosis"]
)

accuracies = [diabetes_acc, heart_acc, stroke_acc, parkinson_acc, cancer_acc]

# ---------------- HERO ----------------

st.markdown("""
<div style='text-align:center;padding:30px;
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;font-size:40px;font-weight:bold;border-radius:10px;'>
🧬 AI Healthcare Intelligence
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- STATS ----------------

st.markdown("## ⚡ Platform Statistics")

c1, c2, c3 = st.columns(3)

c1.metric("ML Models", 5)
c2.metric("Diseases Covered", 5)
c3.metric("Best Accuracy", f"{max(accuracies):.2f}%")

st.caption("Accuracy estimated using train-test split (not full dataset).")

st.write("")
st.write("")

# ---------------- TABLE ----------------

st.markdown("## 🧠 Machine Learning Models Used")

model_df = pd.DataFrame({
    "Disease": ["Diabetes", "Heart Disease", "Stroke", "Parkinson", "Breast Cancer"],
    "Algorithm": ["Random Forest", "Random Forest", "XGBoost", "XGBoost", "Random Forest"],
    "Accuracy (%)": accuracies
})

st.dataframe(model_df, use_container_width=True)

st.write("")
st.write("")

# ---------------- CHART ----------------

st.markdown("## 📊 Model Accuracy Comparison")

fig = px.bar(
    model_df,
    x="Disease",
    y="Accuracy (%)",
    color="Disease",
    text="Accuracy (%)"
)

fig.update_traces(textposition="outside")

st.plotly_chart(fig, use_container_width=True)

st.write("")
st.write("")

# ---------------- NAVIGATION ----------------

st.markdown("## 🧪 AI Diagnostic Modules")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🩸 Diabetes"):
        st.switch_page("pages/1_Diabetes_Prediction.py")

with col2:
    if st.button("❤️ Heart"):
        st.switch_page("pages/2_Heart_Disease_Prediction.py")

with col3:
    if st.button("🧠 Stroke"):
        st.switch_page("pages/3_Stroke_Prediction.py")

st.write("")

col4, col5, col6 = st.columns(3)

with col4:
    if st.button("🎗 Cancer"):
        st.switch_page("pages/5_Breast_Cancer_Prediction.py")

with col5:
    if st.button("🎙 Parkinson"):
        st.switch_page("pages/4_Parkinsons_Prediction.py")

with col6:
    if st.button("📊 Dashboard"):
        st.switch_page("pages/6_Health_Dashboard.py")

st.write("")
st.success("Select any module to start AI diagnosis.")