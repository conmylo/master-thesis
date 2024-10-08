import os
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.svm import OneClassSVM
import nltk
from textstat import flesch_reading_ease
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

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

# Stylometric feature extraction
def stylometric_features(tweet, tokens):
    prompt_length_chars = len(tweet)  # Length of the tweet in characters
    
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
    
    # Function Word Ratio (common stopwords)
    function_words = set(stopwords.words('english'))
    function_word_count = len([word for word in tokens if word in function_words])
    function_word_ratio = function_word_count / len(tokens) if len(tokens) > 0 else 0
    
    return [
        prompt_length_chars, hapax_legomena_ratio, readability_score, 
        punctuation_ratio, capitalization_ratio, function_word_ratio
    ]

# Apply feature extraction to each tweet
stylometric_data = df.apply(lambda row: stylometric_features(row['processed_text'], row['tokenized_text']), axis=1)
stylometric_df = pd.DataFrame(stylometric_data.tolist(), columns=[
    'prompt_length_chars', 'hapax_legomena_ratio', 'readability_score', 
    'punctuation_ratio', 'capitalization_ratio', 'function_word_ratio'
])

# Train and save OCSVM models for each user
users = df['author'].unique()

for user in users:
    # Filter the data for the current user (inliers)
    user_data = stylometric_df[df['author'] == user].values
    
    # Train the OCSVM model for the current user
    model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    model.fit(user_data)
    
    # Save the model to the 'models/' directory
    model_path = os.path.join(model_dir, f"ocsvm_model_{user}.pkl")
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    
    print(f"Model saved for user: {user}")

print("All user models have been trained and saved.")
