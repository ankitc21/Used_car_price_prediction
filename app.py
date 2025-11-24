import streamlit as st
import pandas as pd
import pickle
import json
import os

#  Load mapping and files
with open("columns.json", "r") as f:
    column_data = json.load(f)
    model_names = column_data["model_names"]

with open("model_label_map.json", "r") as f:
    model_map = json.load(f)  # keys are strings like {"Swift": 0, ...}

# Convert keys back to strings if needed
model_map = {k: v for k, v in model_map.items() if isinstance(v, int)}

# Load model and preprocessor
model = pickle.load(open("random_forest_regressor.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

# Streamlit UI
st.set_page_config(page_title="Car Price App", layout="centered")
st.title("Car Price Prediction (with Label Encoding)")
st.markdown("Enter car details to estimate its **selling price**.")

#  Input Fields
model_name = st.selectbox("Car Model", model_names)
vehicle_age = st.slider("Vehicle Age (years)", 0, 25, 5)
km_driven = st.number_input("Kilometers Driven", 0, 300000, 50000, step=1000)
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission_type = st.selectbox("Transmission", ["Manual", "Automatic"])
mileage = st.number_input("Mileage (km/l)", 0.0, 50.0, 18.0)
engine = st.number_input("Engine (CC)", 500, 5000, 1200)
max_power = st.number_input("Max Power (bhp)", 20.0, 400.0, 80.0)
seats = st.slider("Seats", 2, 10, 5)

# Encode model name using mapping
model_encoded = model_map.get(model_name)
if model_encoded is None:
    st.error(f"'{model_name}' not in trained model labels.")
    st.stop()

#  Create input DataFrame
input_df = pd.DataFrame({
    "model": [model_encoded],
    "vehicle_age": [vehicle_age],
    "km_driven": [km_driven],
    "seller_type": [seller_type],
    "fuel_type": [fuel_type],
    "transmission_type": [transmission_type],
    "mileage": [mileage],
    "engine": [engine],
    "max_power": [max_power],
    "seats": [seats]
})

# 🔮 Predict
if st.button("Predict Selling Price"):
    try:
        transformed = preprocessor.transform(input_df)
        price = model.predict(transformed)[0]
        st.success(f"Estimated Price: ₹ {price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
