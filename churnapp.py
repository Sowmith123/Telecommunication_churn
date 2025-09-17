import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
df = pd.read_csv("telecommunication_churn.csv")

# Load trained model
model = joblib.load("churn_model.pkl")

# Configure page
st.set_page_config(page_title="Telecom Churn App", layout="wide")

# Sidebar
st.sidebar.title("📊 Telecom Churn App")
app_mode = st.sidebar.radio("Choose the App Mode", ["Dashboard", "Churn Prediction"])

# -------------------- DASHBOARD --------------------
if app_mode == "Dashboard":
    st.title("📊 Telecom Churn Dashboard")

    # ---- KPIs ----
    total_customers = len(df)
    churned_customers = df["churn"].sum()
    churn_rate = churned_customers / total_customers * 100
    avg_day_mins = df["day_mins"].mean()
    avg_total_charge = df["total_charge"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Customers", total_customers)
    col2.metric("⚠ Churned Customers", churned_customers)
    col3.metric("📉 Churn Rate (%)", f"{churn_rate:.2f}")
    col4.metric("💰 Avg. Total Charge", f"${avg_total_charge:.2f}")

    st.markdown("---")

    # ---- Interactive Filters ----
    state_filter = st.selectbox("Filter by State", ["All"] + sorted(df["state"].unique().tolist()))
    intl_filter = st.selectbox("Filter by International Plan", ["All", "yes", "no"])

    filtered_df = df.copy()
    if state_filter != "All":
        filtered_df = filtered_df[filtered_df["state"] == state_filter]
    if intl_filter != "All":
        filtered_df = filtered_df[filtered_df["international_plan"] == intl_filter]

    # ---- Tabs for Insights ----
    tab1, tab2, tab3 = st.tabs(["📊 Churn Distribution", "📉 Calls & Minutes", "🌍 State Insights"])

    # Tab 1: Churn Distribution
    with tab1:
        st.subheader("Churn vs Non-Churn Customers")
        fig, ax = plt.subplots()
        sns.countplot(x="churn", data=filtered_df, palette="Set2", ax=ax)
        st.pyplot(fig)

        st.subheader("International Plan vs Churn")
        fig, ax = plt.subplots()
        sns.countplot(x="international_plan", hue="churn", data=filtered_df, palette="coolwarm", ax=ax)
        st.pyplot(fig)

    # Tab 2: Calls & Minutes Usage
    with tab2:
        st.subheader("Average Call Minutes by Churn Status")
        fig, ax = plt.subplots(figsize=(8, 5))
        churn_group = filtered_df.groupby("churn")[["day_mins", "evening_mins", "night_mins", "international_mins"]].mean()
        churn_group.plot(kind="bar", ax=ax)
        plt.ylabel("Average Minutes")
        st.pyplot(fig)

        st.subheader("Customer Service Calls vs Churn")
        fig, ax = plt.subplots()
        sns.boxplot(x="churn", y="customer_service_calls", data=filtered_df, palette="Set1", ax=ax)
        st.pyplot(fig)

    # Tab 3: State Insights
    with tab3:
        st.subheader("Churn Rate by State")
        churn_state = filtered_df.groupby("state")["churn"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        churn_state.plot(kind="bar", color="orange", ax=ax)
        plt.ylabel("Churn Rate")
        st.pyplot(fig)

        st.subheader("Top 10 States by Avg. Charges")
        charge_state = filtered_df.groupby("state")["total_charge"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        charge_state.plot(kind="bar", color="blue", ax=ax)
        plt.ylabel("Avg. Charge ($)")
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

    # Derived features
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
