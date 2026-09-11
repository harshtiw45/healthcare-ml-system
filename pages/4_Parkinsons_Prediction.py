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

st.markdown('<div class="title">🎙 Parkinson Disease Detection</div>', unsafe_allow_html=True)

st.write("Detect Parkinson disease using voice signal measurements.")

# ---------- LOAD MODEL + DATASET ----------

model = joblib.load("models/parkinsons_model.pkl")

df = pd.read_csv("datasets/parkinsons.csv")

# remove identifier
if "name" in df.columns:
    df = df.drop("name",axis=1)

features = df.drop("status",axis=1).columns

# ---------- EXAMPLE BUTTONS ----------

colA,colB = st.columns(2)

if colA.button("Load Healthy Example"):
    example = df[df["status"]==0].iloc[0]
    st.session_state["example"]=example

if colB.button("Load Parkinson Example"):
    example = df[df["status"]==1].iloc[0]
    st.session_state["example"]=example

# ---------- INPUT FORM ----------

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

inputs = {}

col1,col2 = st.columns(2)

for i,f in enumerate(features):

    default = float(df[f].mean())

    if "example" in st.session_state:
        default = float(st.session_state["example"][f])

    with (col1 if i%2==0 else col2):

        inputs[f] = st.number_input(
            f,
            value=default,
            step=0.000001,
            format="%.6f"
        )

st.markdown('</div>', unsafe_allow_html=True)

# ---------- PREDICTION ----------

if st.button("Predict Parkinson Risk"):

    input_df = pd.DataFrame([inputs])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    # ---------- GAUGE CHART ----------

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability*100,
        title={'text':"Parkinson Risk"},
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

    if prediction==1:
        st.error("⚠ Parkinson Detected")
    else:
        st.success("✔ No Parkinson Detected")

    st.markdown('</div>', unsafe_allow_html=True)