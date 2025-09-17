# %%
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("churn_model.pkl")

st.title("📊 Telecom Customer Churn Prediction")

st.write("Enter customer details to predict churn:")

# Inputs from user
account_length = st.number_input("Account Length", min_value=0)
day_mins = st.number_input("Day Minutes", min_value=0.0)
evening_mins = st.number_input("Evening Minutes", min_value=0.0)
night_mins = st.number_input("Night Minutes", min_value=0.0)
intl_mins = st.number_input("International Minutes", min_value=0.0)
cust_calls = st.number_input("Customer Service Calls", min_value=0)

if st.button("Predict Churn"):
    features = [[account_length, day_mins, evening_mins, night_mins, intl_mins, cust_calls]]
    prediction = model.predict(features)
    st.success("Prediction: *Churn" if prediction[0] == 1 else "Prediction: **Not Churn*")


