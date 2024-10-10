import pandas as pd
import re
from sklearn.svm import OneClassSVM
import pickle
import os
import numpy as np
import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Preprocess data: Remove URLs
def remove_urls(text):
    return re.sub(r'http\\S+', '', text)

# Load and preprocess dataset
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['content'] = df['content'].apply(remove_urls)
    return df[['author', 'content', 'date_time', 'id']]

# Extract features
def extract_features(text):
    features = {}
    words = nltk.word_tokenize(text)
    features['avg_word_length'] = np.mean([len(word) for word in words])
    char_count = len(text)
    whitespace_count = text.count(' ')
    features['whitespace_ratio'] = whitespace_count / char_count if char_count > 0 else 0
    features['upper_to_lower_ratio'] = sum(1 for c in text if c.isupper()) / (sum(1 for c in text if c.islower()) + 1)
    unique_words = set(words)
    features['vocabulary_richness'] = len(unique_words) / len(words) if words else 0
    features['function_word_count'] = sum(1 for word in words if word.lower() in ['the', 'and', 'in', 'of']) / len(words)
    bigrams = nltk.bigrams(words)
    features['bigrams'] = len(list(bigrams))
    pos_tags = nltk.pos_tag(words)
    pos_counts = nltk.FreqDist(tag for (word, tag) in pos_tags)
    features['noun_ratio'] = pos_counts['NN'] / len(words) if words else 0
    punctuation_count = sum(1 for c in text if c in '.,!?')
    features['punctuation_ratio'] = punctuation_count / len(words) if words else 0
    sentences = nltk.sent_tokenize(text)
    features['sentence_length'] = np.mean([len(nltk.word_tokenize(sentence)) for sentence in sentences]) if sentences else 0
    contraction_count = sum(1 for word in words if "'" in word)
    features['contraction_usage'] = contraction_count / len(words) if words else 0
    return list(features.values())

# Train One-Class SVM model
def train_model(X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    model.fit(X_train_scaled)
    return model, scaler

# Save model and scaler
def save_model_and_scaler(model, scaler, user_id, model_dir="models/"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(f"{model_dir}/user_{user_id}_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open(f"{model_dir}/user_{user_id}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

# Main train function
def main_train(file_path):
    df = preprocess_data(file_path)
    for user, group in df.groupby('author'):
        features = [extract_features(text) for text in group['content']]
        X_train, _ = train_test_split(features, test_size=0.2, random_state=42)
        model, scaler = train_model(X_train)
        save_model_and_scaler(model, scaler, user)

# Run training
if __name__ == '__main__':
    data_path = 'data/cleaned.csv'  # Path to cleaned data
    main_train(data_path)
