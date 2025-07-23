import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler_X = pickle.load(open('scaler_X.pkl', 'rb'))  # Save this from your notebook
scaler_y = pickle.load(open('scaler_y.pkl', 'rb'))

st.title("🏠 Boston Housing Price Predictor")

st.markdown("Enter the property details below:")

# Input fields for features used in the model
crim = st.number_input("Crime Rate (crim)", min_value=0.0, max_value=100.0, value=0.1)
indus = st.number_input("Industrial Proportion (indus)", min_value=0.0, max_value=30.0, value=10.0)
rm = st.number_input("Average Rooms per Dwelling (rm)", min_value=1.0, max_value=10.0, value=6.0)
age = st.number_input("Proportion of Older Buildings (age)", min_value=0.0, max_value=100.0, value=50.0)
tax = st.number_input("Tax Rate (tax)", min_value=100.0, max_value=1000.0, value=300.0)
ptratio = st.number_input("Pupil-Teacher Ratio (ptratio)", min_value=10.0, max_value=30.0, value=18.0)
lstat = st.number_input("Lower Status Population (%) (lstat)", min_value=0.0, max_value=40.0, value=12.0)

if st.button("Predict Price"):
    # Prepare the input
    input_data = np.array([[crim, indus, rm, age, tax, ptratio, lstat]])
    input_scaled = scaler_X.transform(input_data)

    # Predict and inverse scale
    pred_scaled = model.predict(input_scaled)
    prediction = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

    st.success(f"🏠 Estimated Price: ${prediction * 1000:.2f}")
