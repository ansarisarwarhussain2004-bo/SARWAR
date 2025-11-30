import streamlit as st
import pandas as pd 
import numpy as np 
from joblib import load
from sklearn.preprocessing import LabelEncoder
 
 # Load the trained Random Forest Model 
model = load('random_forest_model.joblib')

#Create a streamlit app
st.title("Customer Churn Prediction App")

#input fields for features values on the main screan
st.header("Enter Customer Information")
tenure=st.number_input("Tenure (in months)",min_value=0, max_value=100, value=1)
internet_service = st.selectbox("internet_service",('DSL','No','Fiber','Optic',))
contract= st.selectbox("Contract",('month-to-month','One year','Two year'))
monthly_charges = st.number_input("Monthly_Charges", min_value=0, max_value=200, value=50)
total_charges=st.number_input("Monthly_Charges",min_value=0, max_value=10000, value=0)

# MAP input values to numerical using using the label mapping 
internet_mapping = {
    'DSL': 0,
    'Fiber optic': 1,
    'No': 2
}
contract_mapping = {
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2
}
internet_service = st.selectbox("Internet Service", 
                                ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", 
                        ["Month-to-month", "One year", "Two year"])
internet_service = internet_mapping[internet_service]
contract = contract_mapping[contract]

# Make a prediction using the model
prediction = model.predict([[tenure,internet_service,contract,monthly_charges,total_charges]])

#Display  the prediction result on the main screen
st.header("Prediction Result")
if prediction [0] == 0:
    st.success("This customer is likely to stay")
else:
    st.error("This customer is likely to churn")
# ✅ PREDICT BUTTON
if st.button("Predict Churn"):
   
    prediction = model.predict([[tenure, internet_service, contract, monthly_charges, total_charges]])

    # Display the prediction result
    st.header("Prediction Result")

if prediction[0] == 0:
    st.success("✅ This customer is likely to stay.")
else:
    st.error("⚠️ This customer is likely to churn.")