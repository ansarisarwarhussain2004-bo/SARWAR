import streamlit as st
import pandas as pd 
import numpy as np 
from joblib import dump 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier 


# Load the dataset 
telecom_cust = pd.read_csv('Telco_Customer_Churn.csv')

# Data preprocessing 
# Fill missing values in 'TotalCharges' and convert to numerical 
telecom_cust['TotalCharges'] = pd.to_numeric(telecom_cust['TotalCharges'], errors='coerce')
telecom_cust['TotalCharges'].fillna(0, inplace=True)

#Convert 'Churn' to binary labels 
label_encoder = LabelEncoder()
telecom_cust['Churn'] = label_encoder.fit_transform(telecom_cust['Churn'])

#Use Labels Encoding for 'Internet Service' and 'Contract'
telecom_cust['InternetService'] = label_encoder.fit_transform(telecom_cust['InternetService'])
telecom_cust['Contract'] = label_encoder.fit_transform(telecom_cust['Contract'])

# select features 
selected_features=['tenure','InternetService','Contract','MonthlyCharges','TotalCharges']
X=telecom_cust[selected_features]
y=telecom_cust['Churn']

# Train the random forest model 
model= RandomForestClassifier(n_estimators=100, random_state=101)
model.fit(X,y)

#save the train
dump(model,'random_forest_model.joblib')

