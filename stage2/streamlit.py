import streamlit as st
from sklearn.svm import OneClassSVM
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')

# Sample function to extract stylometric features from a prompt
def extract_features(prompt):
    tokens = word_tokenize(prompt)
    words = [word for word in tokens if word.isalpha()]
    
    # Feature calculations
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    lexical_diversity = len(set(words)) / len(words) if words else 0
    hapax_legomena_ratio = sum(1 for word in set(words) if words.count(word) == 1) / len(words) if words else 0
    avg_sentence_length = len(words) / max(1, len(prompt.split('.')))
    pos_tags = Counter([token.lower() for token in words])
    noun_ratio = pos_tags['n'] / len(words) if 'n' in pos_tags else 0
    verb_ratio = pos_tags['v'] / len(words) if 'v' in pos_tags else 0
    adj_ratio = pos_tags['adj'] / len(words) if 'adj' in pos_tags else 0
    function_word_ratio = len([word for word in words if word in ['and', 'the', 'is', 'a', 'in', 'of']]) / len(words) if words else 0
    
    return [
        avg_word_length, lexical_diversity, hapax_legomena_ratio,
        avg_sentence_length, noun_ratio, verb_ratio, adj_ratio, function_word_ratio
    ]

# Streamlit application layout
st.title("Continuous Authentication with One-Class SVM")

# User inputs
username = st.text_input("Enter your username:")
prompt = st.text_area("Enter your prompt:")

# Initialize SVM models (Simulated for the demo)
user_models = {}

if "user_1" not in user_models:
    example_features = np.array([
        [5.0, 0.8, 0.2, 12, 0.15, 0.1, 0.05, 0.3],  # Simulated feature data
        [4.5, 0.75, 0.25, 10, 0.14, 0.09, 0.04, 0.28],
        [5.1, 0.85, 0.15, 13, 0.16, 0.11, 0.06, 0.32]
    ])
    svm_model = OneClassSVM(kernel="rbf", gamma='auto').fit(example_features)
    user_models["user_1"] = svm_model

# Authenticate the user when both username and prompt are provided
if st.button("Authenticate"):
    if username and prompt:
        if username in user_models:
            features = extract_features(prompt)
            model = user_models[username]
            prediction = model.predict([features])
            if prediction == 1:
                st.write("User authenticated successfully.")
            else:
                st.write("Authentication failed.")
        else:
            st.write("User model not found. Please train the model first.")
    else:
        st.write("Please enter both a username and a prompt.")

