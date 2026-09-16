import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# --- Page Config ---
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")

st.title("💳 Credit Card Fraud Detection AI")
st.write("Enter the transaction details below to check if it's fraudulent.")

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model('fraud_detection_model.keras')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, str(e)

model, scaler_or_error = load_assets()

if model is None:
    st.error(f"⚠️ Failed to load model/scaler. The exact error is:\n\n`{scaler_or_error}`")
    st.stop()
    
scaler = scaler_or_error

# --- UI Input Fields ---
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0)
    customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
    trans_hour = st.slider("Transaction Hour (0-23)", min_value=0, max_value=23, value=14)
    distance_km = st.number_input("Distance from Home (km)", min_value=0.0, value=5.0)

with col2:
    city_pop = st.number_input("City Population", min_value=0, value=10000)
    satisfaction = st.slider("Customer Satisfaction Score", 1, 10, 5)
    loyalty_points = st.number_input("Loyalty Points Earned", min_value=0, value=100)

st.info("💡 **Developer Note:** We are currently predicting using 7 numerical features. The categorical features (like Gender/Merchant) are zero-padded to match the 28 columns your model was trained on.")

if st.button("🔍 Predict Fraud", type="primary"):
    
    # 1. Create a dictionary of the numerical inputs
    input_data = {
        'amt': amt,
        'city_pop': city_pop,
        'Customer_Satisfaction_Score': satisfaction,
        'Customer_Age': customer_age,
        'Loyalty_Points_Earned': loyalty_points,
        'distance_km': distance_km,
        'trans_hour': trans_hour
    }
    
    numerical_df = pd.DataFrame([input_data])
    
    # 2. Reconstruct the full 28-dimension input array
    # Since you scaled all 28 columns (including dummies) in your notebook,
    # we must create the 28 columns BEFORE scaling.
    num_dummy_cols = model.input_shape[1] - len(numerical_df.columns)
    dummy_padding = np.zeros((1, num_dummy_cols))
    
    # Combine numericals + categorical padding (Total 28 columns)
    full_input_array = np.hstack((numerical_df.values, dummy_padding))
    
    # 3. Scale the ENTIRE array (because StandardScaler was fitted on all 28 columns)
    scaled_final_input = scaler.transform(full_input_array)
    
    # 4. Make Prediction
    with st.spinner("Analyzing transaction..."):
        prediction_prob = model.predict(scaled_final_input)[0][0]
        
    # 5. Display Results
    if prediction_prob > 0.80:
        st.error(f"🚨 **FRAUD DETECTED!** (Confidence: {prediction_prob*100:.2f}%)")
        st.write("This transaction matches the pattern of known fraudulent activity.")
    else:
        st.success(f"✅ **Transaction Approved.** (Fraud Probability: {prediction_prob*100:.2f}%)")
        st.write("This transaction appears normal.")
