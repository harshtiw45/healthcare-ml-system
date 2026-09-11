import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ---------- LOAD CSS ----------

def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------- PAGE TITLE ----------

st.markdown('<div class="title">🎗 Breast Cancer Detection</div>', unsafe_allow_html=True)

st.write("Predict breast cancer risk using tumor measurement features.")

# ---------- LOAD MODEL + DATASET ----------

model = joblib.load("models/cancer_model.pkl")

df = pd.read_csv("datasets/breast_cancer.csv")

# remove unwanted columns
for col in ["id","Unnamed: 32"]:
    if col in df.columns:
        df = df.drop(col,axis=1)

features = df.drop("diagnosis",axis=1).columns

# ---------- INPUT FORM ----------

st.markdown('<div class="section-title">Tumor Measurements</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

inputs = {}

col1,col2 = st.columns(2)

for i,f in enumerate(features):

    default = float(df[f].mean())

    with (col1 if i%2==0 else col2):

        inputs[f] = st.number_input(
            f,
            value=default
        )

st.markdown('</div>', unsafe_allow_html=True)

# ---------- PREDICTION ----------

if st.button("Predict Cancer Risk"):

    input_df = pd.DataFrame([inputs])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    # ---------- GAUGE CHART ----------

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability*100,
        title={'text':"Cancer Risk"},
        gauge={
            'axis':{'range':[0,100]},
            'steps':[
                {'range':[0,40],'color':'green'},
                {'range':[40,70],'color':'orange'},
                {'range':[70,100],'color':'red'}
            ]
        }
    ))

    st.plotly_chart(fig,use_container_width=True)

    # ---------- RESULT CARD ----------

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if prediction == 1:
        st.error("⚠ Malignant Tumor Detected")
    else:
        st.success("✔ Tumor Appears Benign")

    st.markdown('</div>', unsafe_allow_html=True)