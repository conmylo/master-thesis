import streamlit as st
import pandas as pd
import os
from utils import load_model, extract_features, train_svm_model

# Load dataset (ensure dataset path is correct)
data = pd.read_csv('data/subsmall.csv', on_bad_lines='skip')

# Streamlit app interface
st.title('Stylometric User Authentication & Model Training')

# Train SVM Model
st.subheader("Train a Model for a User")
train_username = st.text_input('Enter username to train the model:')
if st.button('Train Model'):
    if train_username:
        # Filter data for the specific user
        user_data = data[data['username'] == train_username]

        if not user_data.empty:
            # Train the model
            model_filename = train_svm_model(train_username, user_data)
            
            if model_filename and os.path.exists(model_filename):
                st.success(f"Model created successfully at {model_filename}")
            else:
                st.error(f"Failed to save model at {model_filename}")
        else:
            st.error(f"No data found for user {train_username}.")
    else:
        st.error("Please enter a valid username to train the model.")

# User Authentication
st.subheader("Authenticate a User Based on Writing Style")
username = st.text_input('Enter your User ID:')
raw_text = st.text_area('Enter your prompt for authentication:')

# Authenticate user based on prompt
if st.button('Authenticate'):
    model = load_model(username)
    
    if model:
        svm, scaler = model  # Unpack the saved model and scaler
        
        # Extract features from the prompt
        features_df = extract_features(raw_text)
        
        # Scale the input features using the same scaler
        scaled_input = scaler.transform(features_df)
        
        # Use the SVM model to predict
        prediction = svm.predict(scaled_input)
        
        if prediction == 1:
            st.success('User authenticated successfully!')
        else:
            st.error('Authentication failed.')
            
            # Display a table showing why the user was not authenticated
            st.write("Analyzing feature differences:")
            
            # Get the decision function score for the user's input
            decision_scores = svm.decision_function(scaled_input)
            
            # Create a comparison table with the feature values and decision score
            comparison_table = pd.DataFrame({
                'Feature': features_df.columns,
                'User Input (raw)': features_df.iloc[0],
                'SVM Decision Score': decision_scores[0]
            })
            
            # Display the table
            st.write(comparison_table)
    else:
        st.error(f'Model not found for {username}. Please ensure the model is trained and saved.')
