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

# Extract features as proportions
def extract_features(text):
    features = {}
    words = nltk.word_tokenize(text)
    char_count = len(text) if len(text) > 0 else 1  # Ensure no division by zero
    
    # Existing Features (converted to proportions)
    features['avg_word_length'] = np.mean([len(word) for word in words]) / char_count
    whitespace_count = text.count(' ')
    features['whitespace_ratio'] = whitespace_count / char_count if char_count > 0 else 0
    features['upper_to_lower_ratio'] = sum(1 for c in text if c.isupper()) / (sum(1 for c in text if c.islower()) + 1)
    unique_words = set(words)
    features['vocabulary_richness'] = len(unique_words) / len(words) if words else 0
    bigrams = nltk.bigrams(words)
    features['bigrams_ratio'] = len(list(bigrams)) / len(words) if words else 0
    pos_tags = nltk.pos_tag(words)
    pos_counts = nltk.FreqDist(tag for (word, tag) in pos_tags)
    features['noun_ratio'] = pos_counts['NN'] / len(words) if words else 0

    return list(features.values())

# Train One-Class SVM model with Grid Search and Custom Logging
def train_model(X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Extended parameter grid for better exploration
    param_grid = {
        'nu': [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
        'gamma': [0.05, 0.1, 0.2, 0.5, 1.0]
    }

    # Define the One-Class SVM model
    model = OneClassSVM(kernel='rbf')
    
    # Grid search for best parameters
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    
    # Fit the model with training data (One-Class SVM is trained only on inliers)
    grid_search.fit(X_train_scaled, np.ones(len(X_train_scaled)))  # Use np.ones since it's all inliers
    
    # Return the best model, scaler, and best parameters
    return grid_search.best_estimator_, scaler, grid_search.best_params_, grid_search.best_score_

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

# Main train function with user, best parameters, and F1 score logging
def main_train(file_path):
    df = preprocess_data(file_path)
    for user, group in df.groupby('author'):
        features = [extract_features(text) for text in group['content']]
        X_train, _ = train_test_split(features, test_size=0.2, random_state=42)
        
        # Train the model and get the best parameters and F1 score
        model, scaler, best_params, best_f1_score = train_model(X_train)
        
        # Print the results for the user
        print(f"User: {user}")
        print(f"Best Parameters: {best_params}")
        print(f"Best F1 Score: {best_f1_score:.5f}\n")
        
        # Save the best model, scaler, and parameters for the user
        save_model_and_scaler(model, scaler, user, best_params)

# Run training
if __name__ == '__main__':
    data_path = 'data/cleaned.csv'  # Path to cleaned data
    main_train(data_path)
