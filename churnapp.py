# churnapp.py

import streamlit as st
import pandas as pd
import numpy as np"""
Telecommunication Churn Dashboard (v1.0)
----------------------------------------
Author: Your Name
Purpose:
    Interactive dashboard to analyze and predict telecom customer churn.
Features:
    - Load & preprocess churn dataset
    - Outlier removal
    - Pre-trained RandomForest model (loaded from churn_model.pkl)
    - KPIs & interactive Plotly charts
    - Single & bulk prediction with churn probability
Dataset:
    telecommunications_churn.csv
"""

# -----------------------------
# Imports
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Page Config & Sidebar Info
# -----------------------------
st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")

with st.sidebar.expander("ℹ️ About this App", expanded=True):
    st.write("""
    This dashboard analyzes telecom churn data and predicts customer churn using a pre-trained Random Forest model.
    
    **Tech Stack:** Streamlit, Pandas, Plotly, Scikit-learn  
    **Dataset:** telecommunications_churn.csv  
    **Author:** Your Name  
    """)
    st.markdown("[🌐 GitHub Repo](https://github.com/your-repo)")

# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_data
def load_data():
    """Load dataset safely."""
    try:
        df = pd.read_csv("telecommunications_churn.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset not found! Please upload 'telecommunications_churn.csv' to the app directory.")
        st.stop()

def preprocess_data(df):
    """Encode categorical features and drop redundant columns."""
    df = df.copy()
    le = LabelEncoder()
    if "voice_mail_plan" in df.columns:
        df["voice_mail_plan"] = le.fit_transform(df["voice_mail_plan"])
    if "international_plan" in df.columns:
        df["international_plan"] = le.fit_transform(df["international_plan"])
    if "churn" in df.columns:
        df["churn"] = le.fit_transform(df["churn"])
    if "total_charge" in df.columns:
        df.drop(columns=["total_charge"], inplace=True)
    return df

def remove_outliers(df, cols):
    """Remove outliers using IQR method."""
    df_clean = df.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

@st.cache_resource
def load_model():
    """Load pre-trained model."""
    try:
        model, scaler, feature_names = pickle.load(open("churn_model.pkl", "rb"))
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("⚠️ Pre-trained model not found! Please ensure 'churn_model.pkl' exists.")
        st.stop()

# -----------------------------
# Load Data & Preprocess
# -----------------------------
raw_df = load_data()
processed_df = preprocess_data(raw_df)
num_cols = processed_df.select_dtypes(include=np.number).columns.drop("churn")
clean_df = remove_outliers(processed_df, num_cols)

# Load pre-trained model
model, scaler, feature_names = load_model()

# -----------------------------
# Sidebar Menu
# -----------------------------
menu = st.sidebar.radio("Navigation", ["📊 Dashboard", "🔮 Predict Single", "📂 Bulk Prediction"])

# -----------------------------
# Dashboard
# -----------------------------
if menu == "📊 Dashboard":
    st.title("📊 Telecom Churn Dashboard")

    # --- KPIs ---
    total_customers = len(raw_df)
    churned = raw_df[raw_df["churn"].isin([1, "yes"])].shape[0]
    churn_rate = round((churned / total_customers) * 100, 2)
    avg_calls = round(raw_df["customer_service_calls"].mean(), 2)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("👥 Total Customers", total_customers)
    kpi2.metric("❌ Churned Customers", churned)
    kpi3.metric("📉 Churn Rate (%)", f"{churn_rate}%")
    kpi4.metric("☎ Avg. Cust Service Calls", avg_calls)

    st.markdown("---")

    # --- Visual Insights ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Distribution")
        fig1 = px.pie(raw_df, names="churn", title="Churned vs Loyal Customers",
                      color_discrete_sequence=["skyblue", "salmon"])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Churn by International Plan")
        fig2 = px.histogram(raw_df, x="international_plan", color="churn", barmode="group",
                            title="International Plan vs Churn")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Customer Service Calls by Churn")
        fig3 = px.box(raw_df, x="churn", y="customer_service_calls", color="churn")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Correlation Heatmap")
        fig4, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(clean_df.corr(), cmap="coolwarm", annot=False, ax=ax)
        st.pyplot(fig4)

    # --- Feature Importance ---
    st.subheader("Feature Importance")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    fig5 = px.bar(importances.head(10), x=importances.head(10).values, y=importances.head(10).index,
                  orientation="h", title="Top 10 Important Features")
    st.plotly_chart(fig5, use_container_width=True)

    # --- Model Metrics ---
    with st.expander("📊 Model Metrics"):
        # Evaluate on current clean_df
        X = clean_df.drop(columns=["churn"])
        y = clean_df["churn"]
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        acc = accuracy_score(y, y_pred)
        st.success(f"✅ Model Accuracy on current dataset: {acc:.2f}")

        report = classification_report(y, y_pred, output_dict=True)
        st.write(pd.DataFrame(report).transpose())

        cm = confusion_matrix(y, y_pred)
        st.write("Confusion Matrix")
        st.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

# -----------------------------
# Single Prediction
# -----------------------------
elif menu == "🔮 Predict Single":
    st.title("🔮 Single Customer Prediction")

    input_dict = {}
    for feature in feature_names:
        if feature in ["voice_mail_plan", "international_plan"]:
            input_dict[feature] = 1 if st.selectbox(feature, ["no", "yes"]) == "yes" else 0
        else:
            input_dict[feature] = st.number_input(feature, value=0.0)

    input_data = pd.DataFrame([input_dict])

    if st.button("Predict"):
        input_data = input_data.reindex(columns=feature_names, fill_value=0)
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

    uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        new_data_enc = preprocess_data(new_data)

        X_new = new_data_enc.reindex(columns=feature_names, fill_value=0)
        X_new_scaled = scaler.transform(X_new)

        preds = model.predict(X_new_scaled)
        new_data["Churn_Pred"] = ["Yes" if p == 1 else "No" for p in preds]

        st.dataframe(new_data.head())

        csv = new_data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", data=csv,
                           file_name="churn_predictions.csv", mime="text/csv")

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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
    df = pd.read_csv("telecommunications_churn.csv")   # 👈 make sure dataset is uploaded
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
# Outlier Removal
# -----------------------------
def remove_outliers(df, cols):
    df_clean = df.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

# -----------------------------
# Train & Save Model
# -----------------------------
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    pickle.dump((model, scaler, X.columns), open("churn_model.pkl", "wb"))

    return model, scaler, acc, classification_report(y_test, y_pred, output_dict=True), X.columns

def load_model():
    return pickle.load(open("churn_model.pkl", "rb"))

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")

df = load_data()
df_clean = preprocess_data(df)

# Remove outliers
num_cols = df_clean.select_dtypes(include=np.number).columns.drop("churn")
df_clean = remove_outliers(df_clean, num_cols)

menu = st.sidebar.radio("Navigation", ["📊 Dashboard", "🔮 Predict Single", "📂 Bulk Prediction"])

# -----------------------------
# Dashboard
# -----------------------------
if menu == "📊 Dashboard":
    st.title("📊 Telecom Churn Dashboard")

    model, scaler, acc, report, feature_names = train_model(df_clean)

    # --- KPIs ---
    total_customers = len(df)
    churned = df[df["churn"].isin([1, "yes"])].shape[0]
    churn_rate = round((churned / total_customers) * 100, 2)
    avg_calls = round(df["customer_service_calls"].mean(), 2)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("👥 Total Customers", total_customers)
    kpi2.metric("❌ Churned Customers", churned)
    kpi3.metric("📉 Churn Rate (%)", f"{churn_rate}%")
    kpi4.metric("☎ Avg. Cust Service Calls", avg_calls)

    st.markdown("---")

    # --- Visual Insights ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig1 = px.pie(df, names="churn", title="Churned vs Loyal Customers", color_discrete_sequence=["skyblue", "salmon"])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Churn by International Plan")
        fig2 = px.histogram(df, x="international_plan", color="churn", barmode="group",
                            title="International Plan vs Churn")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Customer Service Calls by Churn")
        fig3 = px.box(df, x="churn", y="customer_service_calls", color="churn")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Correlation Heatmap")
        fig4, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df_clean.corr(), cmap="coolwarm", annot=False, ax=ax)
        st.pyplot(fig4)

    # --- Feature Importance ---
    st.subheader("Feature Importance")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    fig5 = px.bar(importances.head(10), x=importances.head(10).values, y=importances.head(10).index,
                  orientation="h", title="Top 10 Important Features")
    st.plotly_chart(fig5, use_container_width=True)

    st.success(f"✅ Model Accuracy: {acc:.2f}")

# -----------------------------
# Single Prediction
# -----------------------------
elif menu == "🔮 Predict Single":
    st.title("🔮 Single Customer Prediction")

    model, scaler, feature_names = load_model()

    # Input form
    input_dict = {}
    for feature in feature_names:
        if feature in ["voice_mail_plan", "international_plan"]:
            input_dict[feature] = 1 if st.selectbox(feature, ["no", "yes"]) == "yes" else 0
        else:
            input_dict[feature] = st.number_input(feature, value=0.0)

    input_data = pd.DataFrame([input_dict])

    if st.button("Predict"):
        input_data = input_data.reindex(columns=feature_names, fill_value=0)
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
    model, scaler, feature_names = load_model()

    uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        new_data_enc = preprocess_data(new_data)

        X_new = new_data_enc.reindex(columns=feature_names, fill_value=0)
        X_new_scaled = scaler.transform(X_new)

        preds = model.predict(X_new_scaled)
        new_data["Churn_Pred"] = ["Yes" if p == 1 else "No" for p in preds]

        st.dataframe(new_data.head())

        csv = new_data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", data=csv,
                           file_name="churn_predictions.csv", mime="text/csv")

