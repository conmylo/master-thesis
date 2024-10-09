import os
import pickle
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textstat import flesch_reading_ease
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Define the directory where the models are saved
model_dir = "models"

# Preprocessing function to clean the tweet
def preprocess_tweet(tweet):
    tweet = re.sub(r"http\S+|@\S+|[^A-Za-z0-9\s]", "", tweet)  # Remove URLs, mentions, and special characters
    tweet = tweet.lower()  # Lowercasing
    return tweet

# Feature extraction function (same as in train.py)
def extract_features(tweet, tokens):
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
    
    pos_tags = pos_tag(tokens)
    num_nouns = len([word for word, tag in pos_tags if tag.startswith('NN')])
    num_verbs = len([word for word, tag in pos_tags if tag.startswith('VB')])
    num_adjectives = len([word for word, tag in pos_tags if tag.startswith('JJ')])
    num_adverbs = len([word for word, tag in pos_tags if tag.startswith('RB')])
    
    noun_ratio = num_nouns / len(tokens) if len(tokens) > 0 else 0
    verb_ratio = num_verbs / len(tokens) if len(tokens) > 0 else 0
    adjective_ratio = num_adjectives / len(tokens) if len(tokens) > 0 else 0
    adverb_ratio = num_adverbs / len(tokens) if len(tokens) > 0 else 0
    
    avg_word_length = np.mean([len(word) for word in tokens]) if len(tokens) > 0 else 0
    num_sentences = len(re.findall(r'[.!?]', tweet)) or 1
    avg_sentence_length = len(tokens) / num_sentences

    return [
        prompt_length_chars, hapax_legomena_ratio, readability_score, punctuation_ratio,
        capitalization_ratio, function_word_ratio, noun_ratio, verb_ratio,
        adjective_ratio, adverb_ratio, avg_word_length, avg_sentence_length
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

# Function to calculate FAR and FRR
def calculate_far_frr(y_true, y_pred, usernames):
    false_acceptances = 0
    false_rejections = 0
    total_inliers = 0
    total_outliers = 0
    
    for true_label, pred_label, username in zip(y_true, y_pred, usernames):
        if true_label == 1:  # Inliers (actual user's tweet)
            total_inliers += 1
            if pred_label == -1:  # False Rejection
                false_rejections += 1
        else:  # Outliers (tweets from other users)
            total_outliers += 1
            if pred_label == 1:  # False Acceptance
                false_acceptances += 1

    far = false_acceptances / total_outliers if total_outliers > 0 else 0
    frr = false_rejections / total_inliers if total_inliers > 0 else 0

    return far, frr

# Test the system using train-test split
def test_system(df):
    # Split the dataset into training and testing sets (80% train, 20% test)
    train_data, test_data = train_test_split(df, test_size=0.2, stratify=df['author'], random_state=42)
    
    y_true = []
    y_pred = []
    usernames = []

    # Loop through the test data
    for idx, row in test_data.iterrows():
        username = row['author']  # Actual username (the true author of the tweet)
        tweet = row['content']  # The tweet
        
        # Load the corresponding model for the username
        model = load_user_model(username)
        
        if model is None:
            continue  # If the model doesn't exist, skip
        
        # Preprocess the tweet and extract features
        processed_prompt = preprocess_tweet(tweet)
        tokens = word_tokenize(processed_prompt)
        features = np.array(extract_features(processed_prompt, tokens)).reshape(1, -1)
        
        # Make a prediction using the OCSVM model
        prediction = model.predict(features)
        
        # Check if it's an inlier (1 = inlier, -1 = outlier)
        if prediction == 1:
            y_pred.append(1)  # Inlier
        else:
            y_pred.append(-1)  # Outlier
        
        # The true label is 1 (inlier) since this tweet belongs to the actual user
        y_true.append(1)  # True label (inlier)
        usernames.append(username)
    
    # Now we test for outliers by testing the tweets from different users
    for idx, row in test_data.iterrows():
        username = row['author']  # Actual username
        tweet = row['content']  # The tweet
        
        # Load a different model (test if tweet is wrongly accepted by a different user)
        for other_username in df['author'].unique():
            if other_username == username:
                continue  # Skip the correct user
            
            model = load_user_model(other_username)
            
            if model is None:
                continue
            
            # Preprocess the tweet and extract features
            processed_prompt = preprocess_tweet(tweet)
            tokens = word_tokenize(processed_prompt)
            features = np.array(extract_features(processed_prompt, tokens)).reshape(1, -1)
            
            # Make a prediction using the other user's OCSVM model
            prediction = model.predict(features)
            
            # Check if the model wrongly accepted the tweet as belonging to the other user
            if prediction == 1:
                y_pred.append(1)  # False acceptance
            else:
                y_pred.append(-1)  # Correct rejection (outlier)
            
            # The true label here is -1 (outlier)
            y_true.append(-1)  # True label (outlier)
            usernames.append(other_username)
    
    # Calculate FAR and FRR
    far, frr = calculate_far_frr(y_true, y_pred, usernames)
    
    print(f"False Acceptance Rate (FAR): {far:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")

# Load your dataset (original dataset with tweets)
data_path = "data/tweets.csv"
df = pd.read_csv(data_path)

# Run the test
test_system(df)
