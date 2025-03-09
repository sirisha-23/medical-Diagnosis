import streamlit as st
import numpy as np
import joblib
import os

# Function to load the model
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Model file not found: {path}")
        return None

# Path to the model file
model_path = 'medical_diagnosis_model.pkl'

# Load the pre-trained model
model = load_model(model_path)

if model is not None:
    # Define the symptom list
    symptom_list = ['fever', 'cough', 'fatigue', 'headache', 'shortness of breath']

    # Define the page layout
    st.title('AI-powered Medical Diagnosis System')
    st.write('Enter your symptoms below to get a medical diagnosis prediction.')

    # Create a form for user input
    with st.form('diagnosis_form'):
        symptoms = {}
        for symptom in symptom_list:
            symptoms[symptom] = st.checkbox(symptom)
        
        submitted = st.form_submit_button('Get Diagnosis')

    # Predict the diagnosis based on user input
    if submitted:
        input_features = np.array([int(symptoms[symptom]) for symptom in symptom_list]).reshape(1, -1)
        prediction = model.predict(input_features)
        
        st.write(f'The predicted medical condition is: {prediction[0]}')

    # Display additional information
    st.write('Disclaimer: This is an AI-powered system and should not be considered as a substitute for professional medical advice.')
else:
    st.error("Failed to load the model. Please check the model file path and try again.")