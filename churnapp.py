import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Load dataset
df = pd.read_csv("telecommunication_churn.csv")

# Load trained model
model = joblib.load("churn_model.pkl")

# Sidebar
st.sidebar.title("📊 Telecom Churn App")
app_mode = st.sidebar.radio("Choose the App Mode", ["Dashboard", "Churn Prediction"])

# -------------------- DASHBOARD --------------------
if app_mode == "Dashboard":
    st.title("📊 Telecom Churn Dashboard")
    st.markdown("A detailed view of churn trends, customer behaviors, and insights.")

    # ---- Dataset Preview
    with st.expander("📂 Dataset Preview"):
        st.dataframe(df.head())

    # ---- Churn Distribution
    st.subheader("Churn Distribution")
    churn_counts = df["churn"].value_counts().reset_index()
    fig = px.pie(churn_counts, names="index", values="churn", color="index",
                 color_discrete_map={"yes":"red","no":"green"},
                 title="Churn vs Non-Churn Customers")
    st.plotly_chart(fig)

    # ---- International Plan vs Churn
    st.subheader("International Plan vs Churn")
    fig = px.histogram(df, x="international_plan", color="churn", barmode="group",
                       title="Impact of International Plan on Churn")
    st.plotly_chart(fig)

    # ---- Customer Service Calls vs Churn
    st.subheader("Customer Service Calls vs Churn")
    fig = px.box(df, x="churn", y="customer_service_calls", color="churn",
                 title="Customer Service Calls Distribution")
    st.plotly_chart(fig)

    # ---- Average Usage Insights
    st.subheader("Average Usage (Day, Evening, Night, International)")
    avg_usage = df.groupby("churn")[["day_mins","evening_mins","night_mins","international_mins"]].mean().reset_index()
    fig = px.bar(avg_usage, x="churn", y=["day_mins","evening_mins","night_mins","international_mins"],
                 barmode="group", title="Average Call Minutes by Churn Status")
    st.plotly_chart(fig)

    # ---- Correlation Heatmap
    st.subheader("Correlation Heatmap (Numerical Features)")
    numeric_df = df.select_dtypes(include=["float64","int64"])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap")
    st.plotly_chart(fig)

    # ---- State-wise Churn (if you have state column)
    if "state" in df.columns:
        st.subheader("State-wise Churn Rate")
        churn_rate = df.groupby("state")["churn"].mean().reset_index()
        fig = px.choropleth(churn_rate, locations="state", locationmode="USA-states",
                            color="churn", scope="usa",
                            color_continuous_scale="Reds", title="Churn Rate by State")
        st.plotly_chart(fig)

    st.success("✅ Dashboard Loaded Successfully!")

# -------------------- PREDICTION --------------------
elif app_mode == "Churn Prediction":
    st.title("📈 Churn Prediction Tool")
    st.markdown("Fill the details below to check if a customer is likely to churn.")

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
