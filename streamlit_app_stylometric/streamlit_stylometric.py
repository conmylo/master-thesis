import streamlit as st
import pandas as pd
import joblib

# Function to load SVM model for a user
def load_model(user_id):
    model_filename = f'models/{user_id}_svm_model.pkl'
    try:
        model = joblib.load(model_filename)
        return model
    except FileNotFoundError:
        st.error(f"Model not found for {user_id}. Please ensure the model is trained and saved.")
        return None

# Streamlit app interface
st.title('Stylometric User Authentication')

# User input for authentication
username = st.text_input('Enter your User ID:')
raw_text = st.text_area('Enter your prompt:')

# Extract features from the text (matching the features used in training)
def extract_features(raw_text):
    words = raw_text.split()
    unique_words = set(words)
    lexical_diversity = len(unique_words) / len(words)
    
    function_words = {'the', 'and', 'is', 'in', 'on', 'at', 'by', 'with', 'a', 'an'}
    function_word_ratio = sum(1 for word in words if word in function_words) / len(words)
    
    punctuation_count = sum(1 for char in raw_text if char in '.,!?')
    punctuation_usage = punctuation_count / len(raw_text)
    
    pronoun_usage = sum(1 for word in words if word.lower() in ['i', 'he', 'she', 'we', 'they']) / len(words)
    
    vocabulary_richness = len(unique_words) / len(words)
    
    # Return a DataFrame with feature names
    return pd.DataFrame([[
        lexical_diversity, function_word_ratio, punctuation_usage,
        pronoun_usage, vocabulary_richness
    ]], columns=['lexical_diversity', 'function_word_ratio', 'punctuation_usage',
                 'pronoun_usage', 'vocabulary_richness'])

# Authenticate user based on prompt
if st.button('Authenticate'):
    model = load_model(username)
    
    if model:
        # Extract features from the prompt
        features_df = extract_features(raw_text)
        
        # Use the SVM model to predict
        prediction = model.predict(features_df)
        
        if prediction == 1:
            st.success('User authenticated successfully!')
        else:
            st.error('Authentication failed.')
            
            # Display a table showing why the user was not authenticated
            st.write("Analyzing feature differences:")
            
            # Get the decision function score for the user's input
            decision_scores = model.decision_function(features_df)
            
            # Create a comparison table with the feature values and decision score
            comparison_table = pd.DataFrame({
                'Feature': features_df.columns,
                'User Input (raw)': features_df.iloc[0],
                'SVM Decision Score': decision_scores[0]
            })
            
            # Display the table
            st.write(comparison_table)
    else:
        st.error('Model not found for this user.')
