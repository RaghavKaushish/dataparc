import streamlit as st
import joblib
import pandas as pd

# 1. Load the model and assets from the root directory
# These names match the files seen in your GitHub repository
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')
tag_to_name = joblib.load('tag_to_name.pkl')
target_names = joblib.load('target_names.pkl')

# 2. App Interface
st.title("AI-Powered Social Media Growth Assistant")

st.subheader("Input Your Content Metrics")

# 3. Add your sliders and inputs
# Replace these examples with your specific input features
followers = st.slider("Current Followers", 0, 100000, 1000)
engagement = st.number_input("Engagement Rate (%)", 0.0, 100.0, 5.0)

# 4. Prediction Logic
if st.button("Predict Growth"):
    try:
        # Prepare the input data
        # Ensure the column order matches what your model expects
        input_data = pd.DataFrame([[followers, engagement]], columns=features)
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Get prediction
        prediction = model.predict(input_scaled)
        
        # Display result
        st.success(f"Predicted Growth Outcome: {prediction[0]}")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# 5. Debugging / Info
st.sidebar.info("Model and assets loaded successfully from root directory.")
