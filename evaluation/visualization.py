import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Feature names (used during training)
FEATURE_NAMES = ['avg_word_length', 'function_word_ratio', 'punctuation_usage', 'pronoun_usage', 'lexical_diversity', 'prompt_length']

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

# Extract stylometric features from the user's prompt, including prompt length
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
    
    prompt_length = len(text)  # Added prompt length (in characters)
    
    # Return as a DataFrame with feature names
    return pd.DataFrame([[avg_word_length, function_word_ratio, punctuation_usage, pronoun_usage, lexical_diversity, prompt_length]],
                        columns=FEATURE_NAMES)

# Streamlit app
st.title('Stylometric User Authentication with Visualization')

# Input username and prompt
username = st.text_input('Enter your username:')
prompt = st.text_area('Enter your prompt:')

# Authenticate the user
if st.button('Authenticate'):
    if username and prompt:
        model, scaler = load_model_and_scaler(username)
        
        if model and scaler:
            # Extract and normalize features for the given prompt
            given_features_df = extract_stylometric_features(prompt)
            given_features_scaled = scaler.transform(given_features_df)

            # Predict using the SVM for the given prompt
            prediction = model.predict(given_features_scaled)
            decision_function_value_given = model.decision_function(given_features_scaled)[0]  # Decision score for the given prompt
            
            if prediction == 1:
                st.success('User authenticated successfully!')
            else:
                st.error('Authentication failed.')

            # Display the actual feature values and the SVM decision function score for the given prompt
            st.write("### Analyzing Feature Differences (Given Prompt):")
            comparison_table = pd.DataFrame({
                'Feature': ['Avg Word Length', 'Function Word Ratio', 'Punctuation Usage', 
                            'Pronoun Usage', 'Lexical Diversity', 'Prompt Length'],
                'Given Prompt (Raw)': given_features_df.values[0],
                'SVM Decision Score': [decision_function_value_given] * len(given_features_df.values[0])
            })
            st.write(comparison_table)
            
            # Load the user's historical data for visualization
            df = pd.read_csv('data/subClean.csv')  # Adjust path if needed
            user_data = df[df['username'] == username]['title']  # Assuming 'title' is the prompt text

            if not user_data.empty:
                # Extract features for all the user's historical prompts
                historical_features = [extract_stylometric_features(text) for text in user_data]
                historical_features_df = pd.concat(historical_features)

                # Normalize historical features using the scaler
                historical_features_scaled = scaler.transform(historical_features_df)

                # Apply PCA to reduce dimensions to 2D for visualization (on combined data)
                pca = PCA(n_components=2)

                # Combine given prompt and historical data for PCA
                combined_features_scaled = np.vstack([historical_features_scaled, given_features_scaled])
                combined_features_pca = pca.fit_transform(combined_features_scaled)

                # Separate the PCA-transformed data
                historical_pca = combined_features_pca[:-1]  # All but last point (historical data)
                given_pca = combined_features_pca[-1]  # Last point (given prompt)

                # Calculate decision function values for the historical prompts
                decision_values_historical = model.decision_function(historical_features_scaled)

                # Plot the historical data points with color based on decision function values
                plt.figure(figsize=(10, 6))
                plt.scatter(historical_pca[:, 0], historical_pca[:, 1], c=decision_values_historical, cmap='coolwarm', edgecolors='k', s=100, label="Historical Prompts")

                # Plot the given prompt in a distinct color (e.g., green)
                plt.scatter(given_pca[0], given_pca[1], color='green', edgecolors='k', s=200, label="Given Prompt", marker='x')

                # Add labels, title, and legend
                plt.colorbar(label='SVM Decision Function Value (Historical Prompts)')
                plt.title(f"SVM Decision Function Values for {username}")
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.grid(True)
                plt.legend()

                # Display the plot in Streamlit
                st.pyplot(plt.gcf())
            else:
                st.warning(f"No historical data found for user {username}.")
        else:
            st.warning(f"Failed to load model for user {username}.")
