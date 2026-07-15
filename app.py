import streamlit as st
import joblib
import pandas as pd

# Load your model and helper files directly from the root directory
# Remove 'model_data/' from all these paths
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')
tag_to_name = joblib.load('tag_to_name.pkl')
target_names = joblib.load('target_names.pkl')

st.title("AI-Powered Social Media Growth Assistant")

# Example of how you would use these in your app:
st.write("Model and assets loaded successfully!")

# Add your input fields and prediction logic here below
# Example:
# user_input = st.number_input("Enter some feature")
# prediction = model.predict([[user_input]])
