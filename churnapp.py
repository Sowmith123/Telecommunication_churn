import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("telecommunication_churn.csv")
    return df

df = load_data()

# -----------------------------
# Title
# -----------------------------
st.title("📊 Telecom Churn Dashboard")
st.markdown("A professional dashboard for *Churn Analysis, EDA, and Prediction*.")

# -----------------------------
# Data Quality Checks
# -----------------------------
st.header("🔍 Data Quality Checks")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Rows", df.shape[0])
with col2:
    st.metric("Null Values", int(df.isnull().sum().sum()))
with col3:
    st.metric("Duplicates", df.duplicated().sum())

# Churn distribution
churn_counts = df['churn'].value_counts()
st.write("### Churn Distribution")
fig, ax = plt.subplots()
ax.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90, colors=["#66b3ff", "#ff9999"])
st.pyplot(fig)

# -----------------------------
# Enhanced EDA
# -----------------------------
st.header("📈 Exploratory Data Analysis")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

st.subheader("Numerical Feature Distributions")
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for i, col in enumerate(num_cols[:6]):
    sns.histplot(df[col], kde=True, ax=axes[i//3, i%3], color="skyblue")
    axes[i//3, i%3].set_title(col)
st.pyplot(fig)

st.subheader("Categorical Feature Distributions")
if len(cat_cols) > 0:
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(12, 4))
    if len(cat_cols) == 1:
        axes = [axes]
    for i, col in enumerate(cat_cols):
        sns.countplot(x=df[col], ax=axes[i], palette="pastel")
        axes[i].set_title(col)
    st.pyplot(fig)

# -----------------------------
# Preprocessing
# -----------------------------
# Encode categorical
if 'voice_mail_plan' in df.columns:
    df['voice_mail_plan'] = df['voice_mail_plan'].map({'yes': 1, 'no': 0})
if 'international_plan' in df.columns:
    df['international_plan'] = df['international_plan'].map({'yes': 1, 'no': 0})

# Drop ID column if present
if 'phone_number' in df.columns:
    df = df.drop('phone_number', axis=1)

# Features/target
X = df.drop('churn', axis=1)
y = df['churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train Multiple Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds)
    }

# -----------------------------
# Model Comparison
# -----------------------------
st.header("🤖 Model Comparison")
results_df = pd.DataFrame(results).T
st.dataframe(results_df.style.background_gradient(cmap="Blues"))

fig, ax = plt.subplots(figsize=(8, 4))
results_df[["Accuracy", "F1"]].plot(kind="bar", ax=ax)
plt.title("Model Performance")
plt.xticks(rotation=0)
st.pyplot(fig)

# Best model
best_model_name = results_df["Accuracy"].idxmax()
best_model = models[best_model_name]
st.success(f"Best Model: {best_model_name} with Accuracy = {results_df['Accuracy'].max():.2f}")

# -----------------------------
# Prediction Section
# -----------------------------
st.header("🔮 Make Predictions")

option = st.radio("Choose Prediction Type", ("Single Prediction", "Bulk Prediction"))

if option == "Single Prediction":
    st.subheader("Single Customer Prediction")
    input_data = []
    for col in X.columns:
        value = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        input_data.append(value)
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_scaled = scaler.transform(input_df)
    pred = best_model.predict(input_scaled)[0]
    st.write("### Prediction Result:", "🚨 Churn" if pred == 1 else "✅ Not Churn")

else:
    st.subheader("Bulk Prediction")
    file = st.file_uploader("Upload CSV for Bulk Prediction", type=["csv"])
    if file is not None:
        bulk_data = pd.read_csv(file)
        bulk_scaled = scaler.transform(bulk_data)
        preds = best_model.predict(bulk_scaled)
        bulk_data['Churn_Prediction'] = preds
        st.dataframe(bulk_data.head())
        csv = bulk_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "bulk_predictions.csv", "text/csv")


Here’s your clean, professional, single-page dashboard with:
