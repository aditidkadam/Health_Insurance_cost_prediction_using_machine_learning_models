import streamlit as st # for web app
import pandas as pd # for data manipulation
import numpy as np # for numerical operations
import joblib # for loading pickle object that we exported in jupyter notebook


scaler = joblib.load('scaler.pkl') # for scaling the numerical features
le_gender = joblib.load("gender_label_encoder.pkl")
le_diabetic = joblib.load("diabetic_label_encoder.pkl")
le_smoker = joblib.load("smoker_label_encoder.pkl")
model = joblib.load("best_model.pkl") # model for prediction


# Set up UI for the web app
st.set_page_config(page_title="Insuarance Claim Predictor", layout="centered")
st.title("Health Insurance Payment Prediction App")
st.write("Enter the details below to estimate the insurance payment amount.")


# Creating a submit form for taking inputs

with st.form(":input_form"):
    col1,col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value= 10.0, max_value= 60.0, value=25.0)
        children = st.number_input("Number of Children", min_value=0, max_value= 8, value=0)

    with col2: 
        bloodpressure = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120)
        gender = st.selectbox("Gender", options= le_gender.classes_)
        diabetic = st.selectbox("Diabetic", options= le_diabetic.classes_)
        smoker = st.selectbox("Smoker", options= le_smoker.classes_)

# Submit button
    submitted = st.form_submit_button("Predict Payment")

if submitted: 

    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "bmi": [bmi],
        "bloodpressure": [bloodpressure],
        "children": [children],
        "diabetic": [diabetic],
        "smoker": [smoker]

    })

# Encode the categorical features using the saved label encoders

    input_data["gender"] = le_gender.transform(input_data["gender"]) #le is label encoder object
    input_data["diabetic"] = le_diabetic.transform(input_data["diabetic"])
    input_data["smoker"] = le_smoker.transform(input_data["smoker"])


    num_cols = ["age", "bmi", "bloodpressure", "children"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])
    prediction = model.predict(input_data)[0]
    st.success(f"**Estimated Insurance Payment Amount** ${prediction:,.2f}")