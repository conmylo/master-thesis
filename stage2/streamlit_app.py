import streamlit as st
import joblib
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter

# Feature extraction function
import string
from nltk.corpus import stopwords
from textstat.textstat import textstatistics

# Initialize stopwords set
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def extract_features(prompt):
    tokens = word_tokenize(prompt)
    words = [word for word in tokens if word.isalpha()]
    
    # Stylometric features
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    lexical_diversity = len(set(words)) / len(words) if words else 0
    hapax_legomena_ratio = sum(1 for word in set(words) if words.count(word) == 1) / len(words) if words else 0
    avg_sentence_length = len(words) / max(1, len(prompt.split('.')))
    pos_tags = Counter([token.lower() for token in words])
    noun_ratio = pos_tags['n'] / len(words) if 'n' in pos_tags else 0
    verb_ratio = pos_tags['v'] / len(words) if 'v' in pos_tags else 0
    adj_ratio = pos_tags['adj'] / len(words) if 'adj' in pos_tags else 0
    function_word_ratio = len([word for word in words if word in ['and', 'the', 'is', 'a', 'in', 'of']]) / len(words) if words else 0

    # Additional features
    punctuation_usage = sum([1 for char in prompt if char in string.punctuation]) / len(prompt) if len(prompt) > 0 else 0
    stopword_ratio = len([word for word in words if word in stop_words]) / len(words) if words else 0
    readability_score = textstatistics().flesch_kincaid_grade(prompt) if len(prompt) > 0 else 0
    pronoun_usage = len([word for word in words if word.lower() in ['i', 'we', 'he', 'she', 'they']]) / len(words) if words else 0
    capitalization_usage = len([char for char in prompt if char.isupper()]) / len(prompt) if len(prompt) > 0 else 0
    
    return [
        avg_word_length, lexical_diversity, hapax_legomena_ratio,
        avg_sentence_length, noun_ratio, verb_ratio, adj_ratio, function_word_ratio,
        punctuation_usage, stopword_ratio, readability_score, pronoun_usage, capitalization_usage
    ]


# Function to load a user's SVM model, scaler, and feature statistics
def load_user_model_and_scaler(user_id):
    try:
        # Load the SVM model, scaler, and feature statistics
        model = joblib.load(f"models/{user_id}_svm_model.pkl")
        scaler = joblib.load(f"models/{user_id}_scaler.pkl")
        feature_stats = joblib.load(f"models/{user_id}_feature_stats.pkl")
        return model, scaler, feature_stats
    except FileNotFoundError:
        return None, None, None

# Streamlit application layout
st.title("User Authentication via One-Class SVM")

# Input fields
username = st.text_input("Enter your User ID:")
prompt = st.text_area("Enter your prompt:")

# Button to authenticate the user
if st.button("Authenticate"):
    if username and prompt:
        # Load the user's SVM model, scaler, and feature statistics
        model, scaler, feature_stats = load_user_model_and_scaler(username)
        
        if model:
            # Extract features from the prompt
            features = extract_features(prompt)
            
            # Normalize the features using the same scaler used during training
            normalized_features = scaler.transform([features])
            
            # Authenticate the user using SVM prediction
            prediction = model.predict(normalized_features)
            
            if prediction == 1:
                st.write("User authenticated successfully!")
            else:
                st.write("Authentication failed.")
                
                # Analyze feature deviations
                feature_names = [
                    "Average Word Length", "Lexical Diversity", "Hapax Legomena Ratio", "Avg Sentence Length",
                    "Noun Ratio", "Verb Ratio", "Adj Ratio", "Function Word Ratio"
                ]
                st.write("Analyzing feature differences:")

                mean_values = feature_stats['mean']
                std_values = feature_stats['std']

                # Display detailed feature deviations
                for i, feature_value in enumerate(features):
                    expected_mean = mean_values[i]
                    expected_std = std_values[i]
                    deviation = abs(feature_value - expected_mean)
                    if deviation > 2 * expected_std:  # Threshold for abnormality
                        st.write(f"{feature_names[i]}: {feature_value:.2f} (expected: {expected_mean:.2f} ± {expected_std:.2f})")
                    else:
                        st.write(f"{feature_names[i]}: {feature_value:.2f} is within normal range.")
        else:
            st.write(f"No model found for {username}. Please ensure the model is trained.")
    else:
        st.write("Please provide both a User ID and a prompt.")
