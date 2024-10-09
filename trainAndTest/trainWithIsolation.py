import os
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import nltk
from textstat import flesch_reading_ease
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Create models directory if it doesn't exist
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Load Dataset
data_path = "data/tweets.csv"
df = pd.read_csv(data_path)

# Preprocessing function to clean the tweets
def preprocess_tweet(tweet):
    tweet = re.sub(r"http\S+|@\S+|[^A-Za-z0-9\s]", "", tweet)  # Remove URLs, mentions, and special characters
    tweet = tweet.lower()  # Lowercasing
    return tweet

df['processed_text'] = df['content'].apply(preprocess_tweet)
df['tokenized_text'] = df['processed_text'].apply(word_tokenize)

# Stylometric + Advanced Feature extraction
def extract_features(tweet, tokens):
    # Stylometric Features
    prompt_length_chars = len(tweet)
    
    word_freq = nltk.FreqDist(tokens)
    hapax_legomena_count = len([word for word in word_freq if word_freq[word] == 1])
    hapax_legomena_ratio = hapax_legomena_count / len(tokens) if len(tokens) > 0 else 0
    
    readability_score = flesch_reading_ease(tweet)
    
    num_punctuation = len(re.findall(r'[.,!?;:]', tweet))
    punctuation_ratio = num_punctuation / len(tweet) if len(tweet) > 0 else 0
    
    num_capitalized = len([char for char in tweet if char.isupper()])
    capitalization_ratio = num_capitalized / len(tweet) if len(tweet) > 0 else 0
    
    function_words = set(stopwords.words('english'))
    function_word_count = len([word for word in tokens if word in function_words])
    function_word_ratio = function_word_count / len(tokens) if len(tokens) > 0 else 0
    
    # New Features: POS Tag Distribution
    pos_tags = pos_tag(tokens)
    num_nouns = len([word for word, tag in pos_tags if tag.startswith('NN')])
    num_verbs = len([word for word, tag in pos_tags if tag.startswith('VB')])
    num_adjectives = len([word for word, tag in pos_tags if tag.startswith('JJ')])
    num_adverbs = len([word for word, tag in pos_tags if tag.startswith('RB')])
    
    noun_ratio = num_nouns / len(tokens) if len(tokens) > 0 else 0
    verb_ratio = num_verbs / len(tokens) if len(tokens) > 0 else 0
    adjective_ratio = num_adjectives / len(tokens) if len(tokens) > 0 else 0
    adverb_ratio = num_adverbs / len(tokens) if len(tokens) > 0 else 0
    
    # New Feature: Average Word Length
    avg_word_length = np.mean([len(word) for word in tokens]) if len(tokens) > 0 else 0
    
    # New Feature: Sentence Length (words per sentence)
    num_sentences = len(re.findall(r'[.!?]', tweet)) or 1  # Avoid division by zero
    avg_sentence_length = len(tokens) / num_sentences

    return [
        prompt_length_chars, hapax_legomena_ratio, readability_score, punctuation_ratio,
        capitalization_ratio, function_word_ratio, noun_ratio, verb_ratio,
        adjective_ratio, adverb_ratio, avg_word_length, avg_sentence_length
    ]

# Apply feature extraction to each tweet
features = df.apply(lambda row: extract_features(row['processed_text'], row['tokenized_text']), axis=1)
features_np = np.array(features.tolist())  # Convert to NumPy array

# Feature Scaling (Min-Max Scaling) on NumPy array
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features_np)

# Save the scaler to the 'models/' directory
scaler_path = os.path.join(model_dir, 'scaler.pkl')
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Train and save OCSVM + Isolation Forest models for each user
users = df['author'].unique()

for user in users:
    # Filter the data for the current user (inliers)
    user_data = scaled_features[df['author'] == user]
    
    # Train the OCSVM model for the current user
    ocsvm_model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)  # Make the model stricter
    ocsvm_model.fit(user_data)
    
    # Train the Isolation Forest model for the current user
    isolation_forest_model = IsolationForest(contamination=0.1, random_state=42)
    isolation_forest_model.fit(user_data)
    
    # Save the OCSVM model
    ocsvm_model_path = os.path.join(model_dir, f"ocsvm_model_{user}.pkl")
    with open(ocsvm_model_path, 'wb') as model_file:
        pickle.dump(ocsvm_model, model_file)
    
    # Save the Isolation Forest model
    isolation_forest_model_path = os.path.join(model_dir, f"isolation_forest_model_{user}.pkl")
    with open(isolation_forest_model_path, 'wb') as model_file:
        pickle.dump(isolation_forest_model, model_file)
    
    print(f"Models saved for user: {user}")

print("All user models and the scaler have been trained and saved.")
