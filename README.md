# Telecommunication_churn
# 📞 Telecom Customer Churn Prediction

## 📝 Business Objective
Customer churn is a major challenge for telecom companies, often exceeding *10% annually*. Retaining customers is more cost-effective than acquiring new ones, so predicting churn allows companies to proactively target at-risk customers with retention campaigns.  

This project builds a *machine learning model* to predict whether a customer will churn (leave) or remain loyal, based on account features and usage behavior.  

---

## 📂 Dataset Details
Each row corresponds to a telecom customer with attributes such as account length, usage, and subscription plans.

*Key variables:*
- state : Categorical (51 states + DC)  
- Area.code : Numeric code  
- account.length : Duration of customer’s account  
- voice.plan : Yes/No, voicemail plan  
- voice.messages : Number of voicemail messages  
- intl.plan : Yes/No, international plan  
- intl.mins, intl.calls, intl.charge : International call usage  
- day.mins, day.calls, day.charge : Daytime usage  
- eve.mins, eve.calls, eve.charge : Evening usage  
- night.mins, night.calls, night.charge : Night usage  
- customer.calls : Calls to customer service  
- churn : Target variable (Yes = churn, No = loyal)  

---

## 🔑 Project Workflow
1. *Exploratory Data Analysis (EDA)*  
   - Data cleaning, missing values, and distributions  
   - Churn rates by features (voice plan, intl plan, customer service calls, etc.)  
   - Correlation heatmaps and feature insights  

2. *Feature Engineering*  
   - Encoding categorical variables (state, plans)  
   - Aggregated features: total minutes, total charges  
   - Handling class imbalance (SMOTE / class weights)  

3. *Model Building*  
   - Logistic Regression (baseline)  
   - Decision Tree, Random Forest  
   - Gradient Boosting (XGBoost) with hyperparameter tuning  

4. *Model Evaluation*  
   - Metrics: Accuracy, Precision, Recall, F1, ROC AUC, PR AUC  
   - Confusion matrix visualization  
   - SHAP feature importance for interpretability  

5. *Deployment*  
   - *Streamlit App* → Interactive churn prediction dashboard  
   - *Flask API* → REST service for churn prediction  
   - Optional: Docker containerization for production  

6. *Final Presentation*  
   - Business problem & data overview  
   - Key EDA insights & churn drivers  
   - Model performance results  
   - Business recommendations & demo  

---

## 📊 Example Visuals
- Churn distribution  
- Churn rate by customer service calls  
- Feature correlation heatmap  
- SHAP summary plot for feature importance  

---

## 🚀 Deployment
### Streamlit App
```bash
streamlit run app/streamlit_app.py
