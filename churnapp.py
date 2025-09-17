import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
df = pd.read_csv("telecommunications_churn.csv")

# Load trained model
model = joblib.load("churn_model.pkl")

# Sidebar for navigation
st.sidebar.title("📊 Telecom Churn App")
app_mode = st.sidebar.selectbox("Choose the App Mode", ["Dashboard", "Churn Prediction"])

# -------------------- DASHBOARD --------------------
if app_mode == "Dashboard":
    st.title("📊 Telecom Churn Dashboard")

    # Show dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Churn distribution
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="churn", data=df, palette="Set2", ax=ax)
    st.pyplot(fig)

    # International Plan vs Churn
    st.subheader("International Plan vs Churn")
    fig, ax = plt.subplots()
    sns.countplot(x="international_plan", hue="churn", data=df, palette="coolwarm", ax=ax)
    st.pyplot(fig)

    # Customer service calls vs churn
    st.subheader("Customer Service Calls vs Churn")
    fig, ax = plt.subplots()
    sns.histplot(df[df["churn"] == 1]["customer_service_calls"], kde=False, color="red", label="Churn")
    sns.histplot(df[df["churn"] == 0]["customer_service_calls"], kde=False, color="green", label="No Churn")
    plt.legend()
    st.pyplot(fig)

    # Numerical correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -------------------- CHURN PREDICTION --------------------
elif app_mode == "Churn Prediction":
    st.title("📈 Churn Prediction Tool")

    # Input form
    account_length = st.number_input("Account Length", min_value=0)
    voice_mail_plan = st.selectbox("Voice Mail Plan", ["no", "yes"])
    voice_mail_messages = st.number_input("Voice Mail Messages", min_value=0)
    day_mins = st.number_input("Day Minutes", min_value=0.0)
    evening_mins = st.number_input("Evening Minutes", min_value=0.0)
    night_mins = st.number_input("Night Minutes", min_value=0.0)
    international_mins = st.number_input("International Minutes", min_value=0.0)
    customer_service_calls = st.number_input("Customer Service Calls", min_value=0)
    international_plan = st.selectbox("International Plan", ["no", "yes"])

    # Derived features (like charges if not in dataset)
    day_charge = day_mins * 0.17
    evening_charge = evening_mins * 0.085
    night_charge = night_mins * 0.045
    international_charge = international_mins * 0.27
    total_charge = day_charge + evening_charge + night_charge + international_charge

    # Convert categorical
    voice_mail_plan_val = 1 if voice_mail_plan == "yes" else 0
    international_plan_val = 1 if international_plan == "yes" else 0

    # Prediction button
    if st.button("Predict Churn"):
        input_data = pd.DataFrame({
            "account_length": [account_length],
            "voice_mail_plan": [voice_mail_plan_val],
            "voice_mail_messages": [voice_mail_messages],
            "day_mins": [day_mins],
            "evening_mins": [evening_mins],
            "night_mins": [night_mins],
            "international_mins": [international_mins],
            "customer_service_calls": [customer_service_calls],
            "international_plan": [international_plan_val],
            "day_charge": [day_charge],
            "evening_charge": [evening_charge],
            "night_charge": [night_charge],
            "international_charge": [international_charge],
            "total_charge": [total_charge]
        })

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠ This customer is likely to CHURN.")
        else:
            st.success("✅ This customer is NOT likely to churn.")
