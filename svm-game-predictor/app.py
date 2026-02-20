import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Video Game Sales Predictor",
    page_icon="🎮",
    layout="centered"
)

st.title("🎮 Video Game Sales Class Predictor")
st.markdown("Predict whether a game will have **Low, Medium, or High** global sales.")

# load model
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.subheader("Enter Regional Sales")

na_sales = st.number_input("NA Sales", min_value=0.0, value=0.5)
eu_sales = st.number_input("EU Sales", min_value=0.0, value=0.3)
jp_sales = st.number_input("JP Sales", min_value=0.0, value=0.1)
other_sales = st.number_input("Other Sales", min_value=0.0, value=0.05)

if st.button("Predict Sales Class"):

    features = np.array([[na_sales, eu_sales, jp_sales, other_sales]])
    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)[0]

    labels = {
        0: "Low Sales 📉",
        1: "Medium Sales 📊",
        2: "High Sales 🚀"
    }

    st.success(f"Prediction: **{labels[pred]}**")