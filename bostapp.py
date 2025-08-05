import streamlit as st
import numpy as np
import pickle
from xgboost import XGBRegressor
import pandas as pd

# Load the model and scalers
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler_X = pickle.load(open('scaler_X.pkl', 'rb'))
scaler_y = pickle.load(open('scaler_y.pkl', 'rb'))

# App title and description
st.set_page_config(page_title="Boston House Price Predictor", page_icon="🏠")
st.title("🏠 Boston Housing Price Predictor")
st.markdown("Predict the value of a house based on location features in Boston. "
            "This app uses a **trained XGBoost Regressor** for estimation.")

st.header("📋 Enter Property Details")

with st.expander("ℹ️ Feature Descriptions"):
    st.markdown("""
    - **Crime Rate (`crim`)**: Per capita crime rate by town  
    - **Industrial Proportion (`indus`)**: Proportion of non-retail business acres per town  
    - **Average Rooms (`rm`)**: Average number of rooms per dwelling  
    - **Older Buildings (%) (`age`)**: Proportion of owner-occupied units built before 1940  
    - **Tax Rate (`tax`)**: Full-value property-tax rate per $10,000  
    - **Pupil-Teacher Ratio (`ptratio`)**: By town  
    - **Lower Status Population (`lstat`)**: % lower status of the population  
    """)

# Inputs
col1, col2, col3 = st.columns(3)
with col1:
    crim = st.number_input("Crime Rate (crim)", 0.0, 100.0, 0.1)
    rm = st.number_input("Avg Rooms (rm)", 1.0, 10.0, 6.0)
    tax = st.number_input("Tax Rate (tax)", 100.0, 1000.0, 300.0)
with col2:
    indus = st.number_input("Industrial Proportion (indus)", 0.0, 30.0, 10.0)
    age = st.number_input("Older Buildings (%) (age)", 0.0, 100.0, 50.0)
    ptratio = st.number_input("Pupil-Teacher Ratio (ptratio)", 10.0, 30.0, 18.0)
with col3:
    lstat = st.number_input("Lower Status Pop. (%) (lstat)", 0.0, 40.0, 12.0)

if st.button("💰 Predict Price"):
    input_data = np.array([[crim, indus, rm, age, tax, ptratio, lstat]])
    input_scaled = scaler_X.transform(input_data)

    # Prediction
    pred_scaled = model.predict(input_scaled)
    predicted_price = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

    st.success(f"🏡 **Estimated House Price:** ${predicted_price * 1000:.2f}")

    # Show input summary
    st.subheader("📊 Input Summary")
    summary = pd.DataFrame(input_data, columns=['crim', 'indus', 'rm', 'age', 'tax', 'ptratio', 'lstat']).T
    summary.columns = ['Value']
    st.table(summary)

    # Optional: Download result
    result_df = pd.DataFrame({
        'Predicted Price (USD)': [f"${predicted_price * 1000:.2f}"]
    })
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Prediction", csv, "prediction.csv", "text/csv")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & XGBoost | Boston Housing Dataset")
