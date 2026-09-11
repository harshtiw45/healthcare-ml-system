import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(layout="wide")
st.title("📊 AI Healthcare Analytics Dashboard")

# ---------------- GENERIC ACCURACY ----------------

@st.cache_data
def get_real_accuracy(model_path, dataset_path, target_options):
    try:
        model = joblib.load(model_path)
        df = pd.read_csv(dataset_path)

        # detect target
        target = None
        for col in target_options:
            if col in df.columns:
                target = col
                break

        if target is None:
            return 0

        X = df.drop(target, axis=1)
        y = df[target]

        # cancer label fix
        if y.dtype == "object":
            y = y.map({"B": 0, "M": 1})

        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # align
        if hasattr(model, "feature_names_in_"):
            X_test = X_test.reindex(columns=model.feature_names_in_, fill_value=0)

        y_pred = model.predict(X_test)

        return round(accuracy_score(y_test, y_pred) * 100, 2)

    except:
        return 0


# ---------------- HEART FIX ----------------

@st.cache_data
def get_heart_accuracy():
    try:
        df = pd.read_csv("datasets/heart.csv")

        target = "HeartDisease" if "HeartDisease" in df.columns else "target"

        X = df.drop(target, axis=1)
        y = df[target]

        # 🔥 apply encoding (important)
        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = joblib.load("models/heart_model.pkl")

        X_test = X_test.reindex(columns=model.feature_names_in_, fill_value=0)

        y_pred = model.predict(X_test)

        return round(accuracy_score(y_test, y_pred) * 100, 2)

    except:
        return 0


# ---------------- LOAD ACCURACIES ----------------

diabetes_acc = get_real_accuracy(
    "models/diabetes_model.pkl", "datasets/diabetes.csv", ["Outcome", "diabetes"]
)

heart_acc = get_heart_accuracy()

stroke_acc = get_real_accuracy(
    "models/stroke_model.pkl", "datasets/stroke.csv", ["stroke"]
)

parkinson_acc = get_real_accuracy(
    "models/parkinsons_model.pkl", "datasets/parkinsons.csv", ["status"]
)

cancer_acc = get_real_accuracy(
    "models/cancer_model.pkl", "datasets/breast_cancer.csv", ["diagnosis"]
)

accuracies = [diabetes_acc, heart_acc, stroke_acc, parkinson_acc, cancer_acc]

# ---------------- STATS ----------------

st.subheader("System Statistics")

c1, c2, c3 = st.columns(3)
c1.metric("ML Models", 5)
c2.metric("Diseases Covered", 5)
c3.metric("Best Accuracy", f"{max(accuracies):.2f}%")

st.caption("Accuracy estimated using train-test split (not full dataset).")

st.write("")

# ---------------- MODEL TABLE ----------------

st.subheader("Machine Learning Models Used")

model_df = pd.DataFrame({
    "Disease": ["Diabetes", "Heart Disease", "Stroke", "Parkinson", "Breast Cancer"],
    "Algorithm": ["Random Forest", "Random Forest", "XGBoost", "XGBoost", "Random Forest"],
    "Accuracy (%)": accuracies
})

st.dataframe(model_df, use_container_width=True)

st.write("")

# ---------------- CHART ----------------

st.subheader("Model Performance")

fig = px.bar(
    model_df,
    x="Disease",
    y="Accuracy (%)",
    color="Disease",
    text="Accuracy (%)"
)

fig.update_traces(textposition="outside")

st.plotly_chart(fig, use_container_width=True)

# ---------------- PDF REPORT ----------------

st.subheader("📄 Generate Report")

def generate_report():
    doc = SimpleDocTemplate("health_report.pdf")
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("AI Healthcare System Report", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Model Accuracies:", styles["Heading2"]))
    content.append(Spacer(1, 10))

    for disease, acc in zip(model_df["Disease"], model_df["Accuracy (%)"]):
        content.append(Paragraph(f"{disease}: {acc}%", styles["Normal"]))
        content.append(Spacer(1, 8))

    content.append(Spacer(1, 20))
    content.append(Paragraph(f"Best Accuracy: {max(accuracies):.2f}%", styles["Normal"]))

    doc.build(content)


if st.button("Generate PDF Report"):
    generate_report()
    st.success("Report generated successfully! Check your project folder.")

# ---------------- NOTE ----------------

st.caption("All values are dynamically computed using model evaluation logic.")