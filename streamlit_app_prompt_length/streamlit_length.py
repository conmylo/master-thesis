import streamlit as st
import joblib
import numpy as np

# Function to load the SVM model and scaler for a specific user
def load_user_model_and_scaler(user_id):
    try:
        model = joblib.load(f"models/{user_id}_svm_model.pkl")
        scaler = joblib.load(f"models/{user_id}_scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None

# Streamlit application layout
st.title("User Authentication via Prompt Length")

# Input fields
username = st.text_input("Enter your User ID:")
prompt = st.text_area("Enter your prompt:")

# Button to authenticate the user
if st.button("Authenticate"):
    if username and prompt:
        # Load the user's SVM model and scaler
        model, scaler = load_user_model_and_scaler(username)
        
        if model and scaler:
            # Extract the prompt length as the feature
            prompt_length = len(prompt)
            prompt_length_array = np.array(prompt_length).reshape(1, -1)
            
            # Normalize the prompt length using the scaler
            normalized_prompt_length = scaler.transform(prompt_length_array)
            
            # Authenticate the user using the SVM model
            prediction = model.predict(normalized_prompt_length)
            
            if prediction == 1:
                st.write("User authenticated successfully!")
            else:
                st.write("Authentication failed. The prompt length does not match your typical pattern.")
        else:
            st.write(f"No model or scaler found for {username}. Please ensure the model is trained.")
    else:
        st.write("Please provide both a User ID and a prompt.")
