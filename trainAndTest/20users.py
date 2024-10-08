import streamlit as st
import numpy as np
import re
import os
import pickle
import nltk
from textstat import flesch_reading_ease
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define the directory where the models are saved
model_dir = "models"

# Preprocessing function to clean the tweet
def preprocess_tweet(tweet):
    tweet = re.sub(r"http\S+|@\S+|[^A-Za-z0-9\s]", "", tweet)  # Remove URLs, mentions, and special characters
    tweet = tweet.lower()  # Lowercasing
    return tweet

# Stylometric feature extraction
def stylometric_features(tweet, tokens):
    prompt_length_chars = len(tweet)
    
    # Hapax Legomena Ratio (words that occur only once in the tweet)
    word_freq = nltk.FreqDist(tokens)
    hapax_legomena_count = len([word for word in word_freq if word_freq[word] == 1])
    hapax_legomena_ratio = hapax_legomena_count / len(tokens) if len(tokens) > 0 else 0
    
    # Readability Score (Flesch reading ease)
    readability_score = flesch_reading_ease(tweet)
    
    # Punctuation Usage Ratio
    num_punctuation = len(re.findall(r'[.,!?;:]', tweet))
    punctuation_ratio = num_punctuation / len(tweet) if len(tweet) > 0 else 0
    
    # Capitalization Usage Ratio
    num_capitalized = len([char for char in tweet if char.isupper()])
    capitalization_ratio = num_capitalized / len(tweet) if len(tweet) > 0 else 0
    
    # Function Word Ratio
    function_words = set(stopwords.words('english'))
    function_word_count = len([word for word in tokens if word in function_words])
    function_word_ratio = function_word_count / len(tokens) if len(tokens) > 0 else 0
    
    return [
        prompt_length_chars, hapax_legomena_ratio, readability_score, 
        punctuation_ratio, capitalization_ratio, function_word_ratio
    ]

# Load the trained OCSVM model for a given user
def load_user_model(username):
    model_path = os.path.join(model_dir, f"ocsvm_model_{username}.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    else:
        return None

# Streamlit App Interface
st.title("User Authentication via OCSVM")

# Input fields for username and prompt
username = st.text_input("Enter your username")
prompt = st.text_area("Enter your tweet")

# Authenticate user when the button is clicked
if st.button("Authenticate"):
    if username and prompt:
        # Load the OCSVM model for the entered username
        model = load_user_model(username)
        
        if model is None:
            st.error(f"No trained model found for user '{username}'.")
        else:
            # Preprocess the entered prompt (tweet)
            processed_prompt = preprocess_tweet(prompt)
            tokens = word_tokenize(processed_prompt)
            
            # Extract stylometric features from the tweet
            stylometric_data = np.array(stylometric_features(processed_prompt, tokens)).reshape(1, -1)
            
            # Make a prediction using the loaded model
            prediction = model.predict(stylometric_data)
            
            # Display the result
            if prediction == 1:
                st.success(f"User '{username}' authenticated successfully!")
            else:
                st.error(f"Authentication failed for user '{username}'.")
    else:
        st.warning("Please provide both username and tweet.")
