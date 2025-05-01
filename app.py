import streamlit as st
import torch

# Safely try to import the prediction function
try:
    from inference import predict_fake_news
except ImportError as e:
    st.error(f"Failed to import inference module: {e}")
    st.stop()

st.title("📰 Fake News Detector")

statement = st.text_area("Enter News Statement:")
job = st.number_input("Job ID:", min_value=0, max_value=13, step=1)
party = st.number_input("Party ID:", min_value=0, max_value=5, step=1)
context = st.number_input("Context ID:", min_value=0, max_value=13, step=1)

if st.button("Check Authenticity"):
    if statement.strip() == "":
        st.warning("Please enter a statement before prediction.")
    else:
        try:
            prediction, probabilities = predict_fake_news(statement, job, party, context)
            label = '🟥 Fake' if prediction == 1 else '🟩 Real'
            confidence = float(torch.max(torch.tensor(probabilities)))
            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: **{confidence:.2f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

