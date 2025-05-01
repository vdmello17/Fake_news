import streamlit as st
import torch
from inference import predict_fake_news

st.title("Fake News Detector")

statement = st.text_area("Enter News Statement:")
job = st.number_input("Job ID:", min_value=0, step=1)
party = st.number_input("Party ID:", min_value=0, step=1)
context = st.number_input("Context ID:", min_value=0, step=1)

if st.button("Check Authenticity"):
    prediction, probabilities = predict_fake_news(statement, job, party, context)
    label = 'Fake' if prediction == 1 else 'Real'
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: {probabilities}")
