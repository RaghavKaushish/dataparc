import streamlit as st
import joblib
import pandas as pd

# Load assets from the model_data folder (matches your train.py)
model = joblib.load('model_data/model.pkl')
scaler = joblib.load('model_data/scaler.pkl')
features = joblib.load('model_data/features.pkl')
tag_to_name = joblib.load('model_data/tag_to_name.pkl')
target_names = joblib.load('model_data/target_names.pkl')

st.title("Dataparc Sensor Data Predictor")

uploaded_file = st.file_uploader("Upload your Sensor Data (Excel)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    # Process the input data to match the training format
    # Ensure columns in uploaded file match 'features'
    try:
        X_input = df[features] 
        X_scaled = scaler.transform(X_input)
        
        predictions = model.predict(X_scaled)
        
        # Create a result dataframe using target_names
        results = pd.DataFrame(predictions, columns=target_names)
        
        st.write("Prediction Results:")
        st.write(results)
    except Exception as e:
        st.error(f"Error processing data: {e}")
