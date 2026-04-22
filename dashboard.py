import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("agriloanscore_model.pkl")

# Translations dictionary
translations = {
    "English": {
        "title": "🌱 AgriLoanScore – Farmer Loan Eligibility Checker",
        "desc": "Fill in your details to check if you are eligible for a loan:",
        "age": "Age",
        "education": "Education",
        "district": "District",
        "land": "Land Size (Acres)",
        "crop": "Crop Type",
        "yield": "Yield (kg per acre)",
        "price": "Market Price (INR per kg)",
        "income": "Annual Income (INR)",
        "cibil": "CIBIL Score",
        "history": "Credit History",
        "repayment": "Previous Loan Repayment",
        "subsidy": "Subsidy Received",
        "loan": "Loan Amount Requested (INR)",
        "purpose": "Loan Purpose",
        "capacity": "Repayment Capacity Est. (INR)",
        "check": "Check Eligibility",
        "eligible": "✅ Eligible for Loan!",
        "not_eligible": "❌ Not Eligible for Loan.",
        "analytics": "📊 Loan Eligibility Insights"
    },
    "Tamil": {
        "title": "🌱 AgriLoanScore – விவசாயி கடன் தகுதி சரிபார்ப்பு",
        "desc": "நீங்கள் கடனுக்கு தகுதியானவரா என்பதை சரிபார்க்க விவரங்களை நிரப்பவும்:",
        "age": "வயது",
        "education": "கல்வி",
        "district": "மாவட்டம்",
        "land": "நில அளவு (ஏக்கர்)",
        "crop": "பயிர் வகை",
        "yield": "உற்பத்தி (கிலோ / ஏக்கர்)",
        "price": "சந்தை விலை (₹ / கிலோ)",
        "income": "வருடாந்திர வருமானம் (₹)",
        "cibil": "சிபில் மதிப்பெண்",
        "history": "கடன் வரலாறு",
        "repayment": "முந்தைய கடன் திருப்பிச் செலுத்தல்",
        "subsidy": "உதவி தொகை பெற்றீர்களா?",
        "loan": "கோரப்பட்ட கடன் தொகை (₹)",
        "purpose": "கடன் நோக்கம்",
        "capacity": "திருப்பிச் செலுத்தும் திறன் (₹)",
        "check": "கடன் தகுதி சரிபார்க்கவும்",
        "eligible": "✅ நீங்கள் கடனுக்கு தகுதியானவர்!",
        "not_eligible": "❌ நீங்கள் கடனுக்கு தகுதியற்றவர்.",
        "analytics": "📊 கடன் தகுதி புள்ளிவிவரங்கள்"
    }
}

# Language selector
language = st.sidebar.selectbox("🌐 Choose Language / மொழியைத் தேர்ந்தெடுக்கவும்", ["English", "Tamil"])
t = translations[language]

# UI
st.title(t["title"])
st.write(t["desc"])

# Input fields
age = st.number_input(t["age"], min_value=18, max_value=100, value=30)
education = st.selectbox(t["education"], ["None", "Primary", "Secondary", "Graduate", "Postgraduate"])
district = st.text_input(t["district"], "Thanjavur")
land_size = st.number_input(t["land"], min_value=0.1, value=1.0)
crop_type = st.selectbox(t["crop"], ["Paddy", "Sugarcane", "Banana", "Turmeric", "Coconut", "Groundnut", "Other"])
yield_per_acre = st.number_input(t["yield"], min_value=100, value=2000)
market_price = st.number_input(t["price"], min_value=1.0, value=20.0)
annual_income = st.number_input(t["income"], min_value=10000, value=200000)
cibil = st.number_input(t["cibil"], min_value=300, max_value=900, value=650)
credit_history = st.selectbox(t["history"], ["Poor", "Average", "Good"])
previous_repayment = st.selectbox(t["repayment"], ["Yes", "No"])
subsidy = st.selectbox(t["subsidy"], ["Yes", "No"])
loan_amount = st.number_input(t["loan"], min_value=10000, value=100000)
loan_purpose = st.selectbox(t["purpose"], ["SeedsFertilizer", "Irrigation", "Livestock", "WorkingCapital", "Equipment"])
repayment_capacity = st.number_input(t["capacity"], min_value=10000, value=80000)

# Prepare input
input_data = pd.DataFrame([{
    "Age": age,
    "Education": education,
    "District": district,
    "Land_Size_Acres": land_size,
    "Crop_Type": crop_type,
    "Yield_kg_per_acre": yield_per_acre,
    "Market_Price_INR_per_kg": market_price,
    "Annual_Income_INR": annual_income,
    "CIBIL_Score": cibil,
    "Credit_History": credit_history,
    "Previous_Loan_Repayment": previous_repayment,
    "Subsidy_Received": subsidy,
    "Loan_Amount_Requested_INR": loan_amount,
    "Loan_Purpose": loan_purpose,
    "Repayment_Capacity_Est_INR": repayment_capacity
}])

# Predict
if st.button(t["check"]):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"{t['eligible']} (Confidence: {probability:.2f})")
    else:
        st.error(f"{t['not_eligible']} (Confidence: {probability:.2f})")

# Analytics section
st.subheader(t["analytics"])
df = pd.read_csv("agriloanscore_dataset_tn.csv")

fig, ax = plt.subplots()
sns.boxplot(x="Eligible", y="CIBIL_Score", data=df, ax=ax)
st.pyplot(fig)
