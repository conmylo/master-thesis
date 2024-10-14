import pandas as pd
import re
from sklearn.svm import OneClassSVM
import pickle
import os
import numpy as np
import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from collections import Counter
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Preprocess data: Remove URLs
def remove_urls(text):
    return re.sub(r'http\\S+', '', text)

# Load and preprocess dataset
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['content'] = df['content'].apply(remove_urls)
    return df[['author', 'content', 'date_time', 'id']]

# Extract additional features
def type_token_ratio(words):
    return len(set(words)) / len(words) if words else 0

def character_ngrams(text, n=3):
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def stop_word_frequency(words):
    stop_words = set(stopwords.words('english'))
    stop_word_count = sum(1 for word in words if word.lower() in stop_words)
    return stop_word_count / len(words) if words else 0

def word_length_distribution(words):
    word_lengths = [len(word) for word in words]
    avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
    return avg_word_length

def punctuation_patterns(text):
    punctuations = Counter(c for c in text if c in '.,!?')
    return punctuations['.'], punctuations[','], punctuations['!'], punctuations['?']

# Extract features
def extract_features(text):
    features = {}
    words = nltk.word_tokenize(text)
    
    # Existing Features
    features['avg_word_length'] = np.mean([len(word) for word in words])
    char_count = len(text)
    whitespace_count = text.count(' ')
    features['whitespace_ratio'] = whitespace_count / char_count if char_count > 0 else 0
    features['upper_to_lower_ratio'] = sum(1 for c in text if c.isupper()) / (sum(1 for c in text if c.islower()) + 1)
    unique_words = set(words)
    features['vocabulary_richness'] = len(unique_words) / len(words) if words else 0
    bigrams = nltk.bigrams(words)
    features['bigrams'] = len(list(bigrams))
    pos_tags = nltk.pos_tag(words)
    pos_counts = nltk.FreqDist(tag for (word, tag) in pos_tags)
    features['noun_ratio'] = pos_counts['NN'] / len(words) if words else 0

    # New Features
    features['type_token_ratio'] = type_token_ratio(words)
    features['char_ngrams_3'] = len(character_ngrams(text, n=3))  # Number of 3-grams
    features['stop_word_freq'] = stop_word_frequency(words)
    features['word_length_avg'] = word_length_distribution(words)
    features['period_count'], features['comma_count'], features['exclamation_count'], features['question_count'] = punctuation_patterns(text)

    return list(features.values())

# Train One-Class SVM model with Grid Search
def train_model(X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Parameters for grid search
    param_grid = {
        'nu': [0.01, 0.02, 0.05, 0.07, 0.1, 0.2],
        'gamma': [0.05, 0.1, 0.2, 0.3, 'scale']
    }

    # Define the One-Class SVM model
    model = OneClassSVM(kernel='rbf')
    
    # Grid search for best parameters
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train_scaled, np.ones(len(X_train_scaled)))  # One-Class SVM is trained on inliers (1s)

    # Return the best model and scaler
    return grid_search.best_estimator_, scaler, grid_search.best_params_

# Save model, scaler, and best parameters
def save_model_and_scaler(model, scaler, user_id, best_params, model_dir="models/"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(f"{model_dir}/user_{user_id}_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open(f"{model_dir}/user_{user_id}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(f"{model_dir}/user_{user_id}_best_params.pkl", 'wb') as f:
        pickle.dump(best_params, f)

# Main train function
def main_train(file_path):
    df = preprocess_data(file_path)
    for user, group in df.groupby('author'):
        features = [extract_features(text) for text in group['content']]
        X_train, _ = train_test_split(features, test_size=0.2, random_state=42)
        model, scaler, best_params = train_model(X_train)
        save_model_and_scaler(model, scaler, user, best_params)

# Run training
if __name__ == '__main__':
    data_path = 'data/cleaned.csv'  # Path to cleaned data
    main_train(data_path)
