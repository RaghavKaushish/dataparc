import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model_data/model.pkl')
scaler = joblib.load('model_data/scaler.pkl')
features = joblib.load('model_data/features.pkl')
target_names = joblib.load('model_data/target_names.pkl')
tag_to_name = joblib.load('model_data/tag_to_name.pkl')

st.title("Boiler Impact Analysis")

user_data = {}
# Show readable names in the UI
with st.expander("Adjust Input Process Parameters"):
    for tag in features:
        name = tag_to_name.get(tag, tag)
        user_data[tag] = st.slider(f"{name}", 0.0, 500.0, 100.0)

if st.button("Calculate System-Wide Impact"):
    input_df = pd.DataFrame([user_data])[features]
    predictions = model.predict(scaler.transform(input_df))[0]
    
    st.subheader("Predicted Output Impacts")
    cols = st.columns(len(target_names))
    for i, name in enumerate(target_names):
        cols[i].metric(label=name, value=f"{predictions[i]:.2f}")