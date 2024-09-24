import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Function to load SVM model and scaler for a user
def load_model_and_scaler(user_id):
    model_filename = f'models/{user_id}_svm_model.pkl'
    scaler_filename = f'models/{user_id}_scaler.pkl'
    
    try:
        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Model not found for user {user_id}.")
        return None, None

# Extract stylometric features from the user's prompt
def extract_stylometric_features(text):
    words = text.split()
    unique_words = set(words)
    
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    lexical_diversity = len(unique_words) / len(words) if words else 0
    
    function_words = {'the', 'and', 'is', 'in', 'on', 'at', 'by', 'with', 'a', 'an'}
    function_word_ratio = sum(1 for word in words if word in function_words) / len(words) if words else 0
    
    punctuation_count = sum(1 for char in text if char in '.,!?')
    punctuation_usage = punctuation_count / len(text) if text else 0
    
    pronoun_usage = sum(1 for word in words if word.lower() in ['i', 'he', 'she', 'we', 'they']) / len(words) if words else 0
    
    return [avg_word_length, function_word_ratio, punctuation_usage, pronoun_usage, lexical_diversity]

# Streamlit app
st.title('Stylometric User Authentication')

# Input username and prompt
username = st.text_input('Enter your username:')
prompt = st.text_area('Enter your prompt:')

# Authenticate the user
if st.button('Authenticate'):
    if username and prompt:
        model, scaler = load_model_and_scaler(username)
        
        if model and scaler:
            # Extract and normalize features
            features = extract_stylometric_features(prompt)
            features_scaled = scaler.transform([features])
            
            # Predict using the SVM
            prediction = model.predict(features_scaled)
            decision_function_value = model.decision_function(features_scaled)[0]  # Decision score
            
            if prediction == 1:
                st.success('User authenticated successfully!')
            else:
                st.error('Authentication failed.')

            # Display the actual feature values and the SVM decision function score
            st.write("### Analyzing Feature Differences:")
            
            # Create a DataFrame to display the comparison
            comparison_table = pd.DataFrame({
                'Feature': ['Avg Word Length', 'Function Word Ratio', 'Punctuation Usage', 
                            'Pronoun Usage', 'Lexical Diversity'],
                'User Input (Raw)': features,
                'SVM Decision Score': [decision_function_value] * len(features)
            })
            
            st.write(comparison_table)
            
            # Provide additional explanation for why authentication failed or succeeded
            if decision_function_value < 0:
                st.warning("The negative decision score indicates the input features are outside the expected range.")
            else:
                st.success("The decision score indicates the input features are within the expected range.")
