import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
import joblib
import os
from nltk.tokenize import word_tokenize
from collections import Counter

# Ensure the models directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# Feature extraction function
def extract_features(prompt):
    tokens = word_tokenize(prompt)
    words = [word for word in tokens if word.isalpha()]
    
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

# Train and save SVM model and feature statistics
def train_svm_for_user(df, user_id):
    user_data = df[df['user_id'] == user_id]
    features = np.array([extract_features(text) for text in user_data['raw_text']])
    
    # Train SVM
    svm_model = OneClassSVM(kernel="rbf", gamma='auto').fit(features)
    
    # Save the SVM model
    joblib.dump(svm_model, f"models/{user_id}_svm_model.pkl")
    
    # Calculate feature statistics (mean and standard deviation)
    mean_values = np.mean(features, axis=0)
    std_values = np.std(features, axis=0)
    feature_stats = {"mean": mean_values, "std": std_values}
    
    # Save the feature statistics
    joblib.dump(feature_stats, f"models/{user_id}_feature_stats.pkl")
    print(f"Model and feature statistics for {user_id} saved!")

# Example usage
if __name__ == "__main__":
    # Load the historical data (replace with actual file path)
    df = pd.read_csv('data/historical_data.csv')

    # Train and save models for each user
    for user_id in df['user_id'].unique():
        train_svm_for_user(df, user_id)
