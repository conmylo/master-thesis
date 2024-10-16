import pickle
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import nltk
from collections import Counter
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Load and preprocess dataset
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
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

# Load model, scaler, and best params from the grid search
def load_model_and_scaler(user_id, model_dir="models/"):
    with open(f"{model_dir}/user_{user_id}_model.pkl", 'rb') as f:
        model = pickle.load(f)
    with open(f"{model_dir}/user_{user_id}_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(f"{model_dir}/user_{user_id}_best_params.pkl", 'rb') as f:
        best_params = pickle.load(f)
    return model, scaler, best_params

# Evaluate model on test data with custom decision threshold
def evaluate_model_with_threshold(model, scaler, X_test, threshold=-0.2):
    X_test_scaled = scaler.transform(X_test)
    decision_scores = model.decision_function(X_test_scaled)
    
    # Apply the custom threshold to decide inliers/outliers
    y_pred = np.where(decision_scores >= threshold, 1, -1)  # Custom threshold
    return y_pred

# Calculate metrics including FAR and FRR
def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='binary', zero_division=1, pos_label=1)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=1, pos_label=1)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=1, pos_label=1)
    
    # Calculate FAR and FRR
    false_rejections = sum((y_pred == -1) & (y_true == 1))  # False negatives (inliers predicted as outliers)
    false_acceptances = sum((y_pred == 1) & (y_true == -1))  # False positives (outliers predicted as inliers)
    total_inliers = sum(y_true == 1)
    total_outliers = sum(y_true == -1)
    
    frr = false_rejections / total_inliers if total_inliers > 0 else 0  # False Rejection Rate (FRR)
    far = false_acceptances / total_outliers if total_outliers > 0 else 0  # False Acceptance Rate (FAR)
    
    return precision, recall, f1, far, frr

# Main test function
def main_test(file_path):
    df = preprocess_data(file_path)
    overall_far, overall_frr = 0, 0
    total_users = len(df['author'].unique())
    
    results = {}
    for user, group in df.groupby('author'):
        model, scaler, best_params = load_model_and_scaler(user)

        # Extract features for inliers (user's own data)
        X_test_inliers = [extract_features(text) for text in group['content']]

        # Collect outliers (data from other users): 50-50 split between inliers and outliers
        outlier_data = df[df['author'] != user]  # Data from all other users
        X_test_outliers = [extract_features(text) for text in outlier_data['content'].sample(n=len(X_test_inliers))]  # Match the number of inliers

        # True labels: 1 for inliers, -1 for outliers
        y_true_inliers = np.ones(len(X_test_inliers))  # 1 for legitimate data (inliers)
        y_true_outliers = -1 * np.ones(len(X_test_outliers))  # -1 for outliers (other users)

        # Combine inliers and outliers for evaluation
        X_test_combined = X_test_inliers + X_test_outliers
        y_true = np.concatenate([y_true_inliers, y_true_outliers])

        # Predictions with the decision threshold applied
        y_pred = evaluate_model_with_threshold(model, scaler, X_test_combined, threshold=0.05)

        # Calculate precision, recall, F1, FAR, FRR
        precision, recall, f1, far, frr = calculate_metrics(y_true, y_pred)
        results[user] = (precision, recall, f1, far, frr)

        # Update overall FAR and FRR
        overall_far += far
        overall_frr += frr

        # Print individual results (rounded to 5 decimals)
        print(f"User {user}: Precision: {precision:.5f}, Recall: {recall:.5f}, F1: {f1:.5f}, FAR: {far:.5f}, FRR: {frr:.5f}")
    
    # Print overall FAR and FRR (rounded to 5 decimals)
    print(f"Overall FAR: {overall_far / total_users:.5f}, Overall FRR: {overall_frr / total_users:.5f}")

# Run testing
if __name__ == '__main__':
    data_path = 'data/cleaned.csv'  # Path to cleaned data
    main_test(data_path)
