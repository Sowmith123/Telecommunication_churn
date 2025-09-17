# telecom_churn_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as p# telecom_churn_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# -----------------------------
# Load & Preprocess Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("telecommunication_churn.csv")   # 👈 your dataset path
    return df

def preprocess_data(df):
    df = df.copy()

    # Encode categorical features
    le = LabelEncoder()
    if "voice_mail_plan" in df.columns:
        df["voice_mail_plan"] = le.fit_transform(df["voice_mail_plan"])
    if "international_plan" in df.columns:
        df["international_plan"] = le.fit_transform(df["international_plan"])
    if "churn" in df.columns:
        df["churn"] = le.fit_transform(df["churn"])  # Target

    # Drop redundant column if exists
    if "total_charge" in df.columns:
        df.drop(columns=["total_charge"], inplace=True)

    return df

# -----------------------------
# Train & Save Model
# -----------------------------
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    # Save model
    pickle.dump((model, scaler), open("churn_model.pkl", "wb"))

    return model, scaler, acc, classification_report(y_test, y_pred, output_dict=True)

def load_model():
    return pickle.load(open("churn_model.pkl", "rb"))

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")

df = load_data()
df_clean = preprocess_data(df)

menu = st.sidebar.radio("Navigation", ["📊 Dashboard", "🔮 Predict Single", "📂 Bulk Prediction"])

# -----------------------------
# Dashboard
# -----------------------------
if menu == "📊 Dashboard":
    st.title("📊 Telecom Churn Dashboard")

    # Train Model (or load if already trained)
    model, scaler, acc, report = train_model(df_clean)

    # --- KPIs ---
    total_customers = len(df)
    churned = df[df["churn"] == "yes"].shape[0] if df["churn"].dtype == "object" else df[df["churn"] == 1].shape[0]
    churn_rate = round((churned / total_customers) * 100, 2)
    avg_calls = round(df["customer_service_calls"].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Customers", total_customers)
    col2.metric("❌ Churned Customers", churned)
    col3.metric("📉 Churn Rate (%)", f"{churn_rate}%")
    col4.metric("☎ Avg. Cust Service Calls", avg_calls)

    # --- Plots ---
    st.subheader("Churn Distribution")
    fig1, ax1 = plt.subplots()
    df["churn"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1, colors=["skyblue", "salmon"])
    st.pyplot(fig1)

    st.subheader("Churn by International Plan")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x="international_plan", hue="churn", palette="Set2", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(12,6))
    sns.heatmap(df_clean.corr(), cmap="coolwarm", annot=False, ax=ax3)
    st.pyplot(fig3)

    st.success(f"✅ Model Accuracy: {acc:.2f}")

# -----------------------------
# Single Prediction
# -----------------------------
elif menu == "🔮 Predict Single":
    st.title("🔮 Single Customer Prediction")

    model, scaler = load_model()

    # Input form
    account_length = st.number_input("Account Length", 1, 300, 100)
    voice_mail_plan = st.selectbox("Voice Mail Plan", ["no", "yes"])
    voice_mail_messages = st.number_input("Voice Mail Messages", 0, 100, 0)
    day_mins = st.number_input("Day Minutes", 0.0, 500.0, 200.0)
    evening_mins = st.number_input("Evening Minutes", 0.0, 500.0, 200.0)
    night_mins = st.number_input("Night Minutes", 0.0, 500.0, 200.0)
    international_mins = st.number_input("International Minutes", 0.0, 100.0, 10.0)
    day_calls = st.number_input("Day Calls", 0, 200, 100)
    evening_calls = st.number_input("Evening Calls", 0, 200, 100)
    night_calls = st.number_input("Night Calls", 0, 200, 100)
    international_calls = st.number_input("International Calls", 0, 20, 2)
    international_charge = st.number_input("International Charge", 0.0, 50.0, 2.0)
    day_charge = st.number_input("Day Charge", 0.0, 100.0, 30.0)
    evening_charge = st.number_input("Evening Charge", 0.0, 100.0, 20.0)
    night_charge = st.number_input("Night Charge", 0.0, 100.0, 10.0)
    customer_service_calls = st.number_input("Customer Service Calls", 0, 20, 1)

    international_plan = st.selectbox("International Plan", ["no", "yes"])

    input_data = pd.DataFrame([{
        "account_length": account_length,
        "voice_mail_plan": 1 if voice_mail_plan == "yes" else 0,
        "voice_mail_messages": voice_mail_messages,
        "day_mins": day_mins,
        "evening_mins": evening_mins,
        "night_mins": night_mins,
        "international_mins": international_mins,
        "day_calls": day_calls,
        "evening_calls": evening_calls,
        "night_calls": night_calls,
        "international_calls": international_calls,
        "international_charge": international_charge,
        "day_charge": day_charge,
        "evening_charge": evening_charge,
        "night_charge": night_charge,
        "customer_service_calls": customer_service_calls,
        "international_plan": 1 if international_plan == "yes" else 0
    }])

    if st.button("Predict"):
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        if pred == 1:
            st.error("⚠ Customer likely to *CHURN*")
        else:
            st.success("✅ Customer will *STAY*")

# -----------------------------
# Bulk Prediction
# -----------------------------
elif menu == "📂 Bulk Prediction":
    st.title("📂 Bulk Prediction")
    model, scaler = load_model()

    uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        new_data_enc = preprocess_data(new_data)

        X_new = new_data_enc.drop(columns=["churn"], errors="ignore")
        X_new_scaled = scaler.transform(X_new)

        preds = model.predict(X_new_scaled)
        new_data["Churn_Pred"] = ["Yes" if p==1 else "No" for p in preds]

        st.dataframe(new_data.head())

        csv = new_data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", data=csv, file_name="churn_predictions.csv", mime="text/csv")lt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# -----------------------------
# Load & Preprocess Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("telecommunication_churn.csv")   # 👈 your dataset path
    return df

def preprocess_data(df):
    df = df.copy()

    # Encode categorical features
    le = LabelEncoder()
    if "voice_mail_plan" in df.columns:
        df["voice_mail_plan"] = le.fit_transform(df["voice_mail_plan"])
    if "international_plan" in df.columns:
        df["international_plan"] = le.fit_transform(df["international_plan"])
    if "churn" in df.columns:
        df["churn"] = le.fit_transform(df["churn"])  # Target

    # Drop redundant column if exists
    if "total_charge" in df.columns:
        df.drop(columns=["total_charge"], inplace=True)

    return df

# -----------------------------
# Train & Save Model
# -----------------------------
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    # Save model
    pickle.dump((model, scaler), open("churn_model.pkl", "wb"))

    return model, scaler, acc, classification_report(y_test, y_pred, output_dict=True)

def load_model():
    return pickle.load(open("churn_model.pkl", "rb"))

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")

df = load_data()
df_clean = preprocess_data(df)

menu = st.sidebar.radio("Navigation", ["📊 Dashboard", "🔮 Predict Single", "📂 Bulk Prediction"])

# -----------------------------
# Dashboard
# -----------------------------
if menu == "📊 Dashboard":
    st.title("📊 Telecom Churn Dashboard")

    # Train Model (or load if already trained)
    model, scaler, acc, report = train_model(df_clean)

    # --- KPIs ---
    total_customers = len(df)
    churned = df[df["churn"] == "yes"].shape[0] if df["churn"].dtype == "object" else df[df["churn"] == 1].shape[0]
    churn_rate = round((churned / total_customers) * 100, 2)
    avg_calls = round(df["customer_service_calls"].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Customers", total_customers)
    col2.metric("❌ Churned Customers", churned)
    col3.metric("📉 Churn Rate (%)", f"{churn_rate}%")
    col4.metric("☎ Avg. Cust Service Calls", avg_calls)

    # --- Plots ---
    st.subheader("Churn Distribution")
    fig1, ax1 = plt.subplots()
    df["churn"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1, colors=["skyblue", "salmon"])
    st.pyplot(fig1)

    st.subheader("Churn by International Plan")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x="international_plan", hue="churn", palette="Set2", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(12,6))
    sns.heatmap(df_clean.corr(), cmap="coolwarm", annot=False, ax=ax3)
    st.pyplot(fig3)

    st.success(f"✅ Model Accuracy: {acc:.2f}")

# -----------------------------
# Single Prediction
# -----------------------------
elif menu == "🔮 Predict Single":
    st.title("🔮 Single Customer Prediction")

    model, scaler = load_model()

    # Input form
    account_length = st.number_input("Account Length", 1, 300, 100)
    voice_mail_plan = st.selectbox("Voice Mail Plan", ["no", "yes"])
    voice_mail_messages = st.number_input("Voice Mail Messages", 0, 100, 0)
    day_mins = st.number_input("Day Minutes", 0.0, 500.0, 200.0)
    evening_mins = st.number_input("Evening Minutes", 0.0, 500.0, 200.0)
    night_mins = st.number_input("Night Minutes", 0.0, 500.0, 200.0)
    international_mins = st.number_input("International Minutes", 0.0, 100.0, 10.0)
    day_calls = st.number_input("Day Calls", 0, 200, 100)
    evening_calls = st.number_input("Evening Calls", 0, 200, 100)
    night_calls = st.number_input("Night Calls", 0, 200, 100)
    international_calls = st.number_input("International Calls", 0, 20, 2)
    international_charge = st.number_input("International Charge", 0.0, 50.0, 2.0)
    day_charge = st.number_input("Day Charge", 0.0, 100.0, 30.0)
    evening_charge = st.number_input("Evening Charge", 0.0, 100.0, 20.0)
    night_charge = st.number_input("Night Charge", 0.0, 100.0, 10.0)
    customer_service_calls = st.number_input("Customer Service Calls", 0, 20, 1)

    international_plan = st.selectbox("International Plan", ["no", "yes"])

    input_data = pd.DataFrame([{
        "account_length": account_length,
        "voice_mail_plan": 1 if voice_mail_plan == "yes" else 0,
        "voice_mail_messages": voice_mail_messages,
        "day_mins": day_mins,
        "evening_mins": evening_mins,
        "night_mins": night_mins,
        "international_mins": international_mins,
        "day_calls": day_calls,
        "evening_calls": evening_calls,
        "night_calls": night_calls,
        "international_calls": international_calls,
        "international_charge": international_charge,
        "day_charge": day_charge,
        "evening_charge": evening_charge,
        "night_charge": night_charge,
        "customer_service_calls": customer_service_calls,
        "international_plan": 1 if international_plan == "yes" else 0
    }])

    if st.button("Predict"):
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        if pred == 1:
            st.error("⚠ Customer likely to *CHURN*")
        else:
            st.success("✅ Customer will *STAY*")

# -----------------------------
# Bulk Prediction
# -----------------------------
elif menu == "📂 Bulk Prediction":
    st.title("📂 Bulk Prediction")
    model, scaler = load_model()

    uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        new_data_enc = preprocess_data(new_data)

        X_new = new_data_enc.drop(columns=["churn"], errors="ignore")
        X_new_scaled = scaler.transform(X_new)

        preds = model.predict(X_new_scaled)
        new_data["Churn_Pred"] = ["Yes" if p==1 else "No" for p in preds]

        st.dataframe(new_data.head())

        csv = new_data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", data=csv, file_name="churn_predictions.csv", mime="text/csv")

