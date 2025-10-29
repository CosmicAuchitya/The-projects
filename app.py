import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math

# --- Page Config (must be first Streamlit command) ---
st.set_page_config(page_title="Fraud Detection App 🚨", page_icon="💳", layout="wide")

# Load trained pipeline
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_rf_model.pkl")

model = load_model()

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
st.sidebar.info("Enter transaction details in the form and get instant fraud prediction.")

# --- Main Title ---
st.title("💳 Fraud Detection Web App")
st.markdown("### Predict whether a transaction is **Fraudulent** or **Legit** in real-time.")

# --- Input Form ---
with st.form("fraud_form"):
    st.subheader("📝 Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox("Category", [
            "misc_pos","gas_transport","grocery_pos","shopping_pos","food_dining",
            "personal_care","health_fitness","home","entertainment","travel","misc_net"
        ])
        gender = st.selectbox("Gender", ["M", "F"])
        state = st.text_input("State (2-letter code)", value="CA")
        amt = st.number_input("Amount ($)", min_value=0.0, value=50.0, format="%.2f")
        age = st.number_input("Age", min_value=0, value=35)

    with col2:
        lat = st.number_input("Customer latitude", value=34.05)
        long = st.number_input("Customer longitude", value=-118.24)
        city_pop = st.number_input("City population", min_value=0, value=100000)
        merch_lat = st.number_input("Merchant latitude", value=34.06)
        merch_long = st.number_input("Merchant longitude", value=-118.25)

    st.subheader("⏰ Transaction Time")
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        trans_hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)
    with col4:
        trans_day = st.number_input("Day (1-31)", min_value=1, max_value=31, value=21)
    with col5:
        trans_weekday = st.number_input("Weekday (0=Mon ... 6=Sun)", min_value=0, max_value=6, value=3)
    with col6:
        trans_month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=6)

    submitted = st.form_submit_button("🔍 Predict Fraud")

# --- Helper: haversine distance ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# --- Prediction ---
if submitted:
    # Engineered features
    log_amt = np.log1p(amt)
    log_city_pop = np.log1p(city_pop)
    is_weekend = 1 if trans_weekday in [5, 6] else 0
    distance = haversine_km(lat, long, merch_lat, merch_long)
    odd_hour = 1 if (trans_hour < 6 or trans_hour > 22) else 0
    high_amount = 1 if amt > 200 else 0
    long_distance = 1 if distance > 50 else 0
    fraud_risk_score = int(odd_hour + high_amount + long_distance)

    row = {
        "category": category,
        "gender": gender,
        "state": state,
        "amt": amt,
        "lat": lat,
        "long": long,
        "city_pop": int(city_pop),
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "trans_hour": int(trans_hour),
        "trans_day": int(trans_day),
        "trans_weekday": int(trans_weekday),
        "trans_month": int(trans_month),
        "age": int(age),
        "distance": distance,
        "odd_hour": odd_hour,
        "high_amount": high_amount,
        "long_distance": long_distance,
        "fraud_risk_score": fraud_risk_score,
        "log_amt": log_amt,
        "log_city_pop": log_city_pop,
        "is_weekend": is_weekend
    }
    X_input = pd.DataFrame([row])

    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    # --- Stylish Output ---
    st.subheader("📢 Prediction Result")
    if pred == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Probability: {proba:.2%})")
    else:
        st.success(f"✅ Legit Transaction (Fraud Probability: {proba:.2%})")

    st.progress(proba)

    with st.expander("🔍 Show Computed Features"):
        st.write(pd.DataFrame([row]).T.rename(columns={0: "value"}))