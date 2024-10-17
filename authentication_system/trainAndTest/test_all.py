import pickle
import os
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import nltk
from collections import Counter
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Preprocess dataset
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    return df[['author', 'content', 'date_time', 'id']]

# Extract features
def extract_features(text):
    features = {}
    words = nltk.word_tokenize(text)
    char_count = len(text) if len(text) > 0 else 1  # Avoid division by zero

    # Existing features converted to proportions
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

# Train One-Class SVM model
def train_model(X_train, nu, gamma):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Define One-Class SVM model
    model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)

    # Fit the model to training data
    model.fit(X_train_scaled)

    return model, scaler

# Evaluate model on test data with custom decision threshold
def evaluate_model_with_threshold(model, scaler, X_test, threshold):
    X_test_scaled = scaler.transform(X_test)
    decision_scores = model.decision_function(X_test_scaled)

    # Apply custom threshold
    y_pred = np.where(decision_scores >= threshold, 1, -1)
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

    frr = false_rejections / total_inliers if total_inliers > 0 else 0  # False Rejection Rate
    far = false_acceptances / total_outliers if total_outliers > 0 else 0  # False Acceptance Rate

    return precision, recall, f1, far, frr

# Test all users with a given combination of nu, gamma, and threshold
def test_with_params(df, nu, gamma, threshold):
    overall_f1, overall_far, overall_frr = 0, 0, 0
    total_users = len(df['author'].unique())

    for user, group in df.groupby('author'):
        # Train model with nu and gamma
        features = [extract_features(text) for text in group['content']]
        model, scaler = train_model(features, nu, gamma)

        # Extract features for inliers
        X_test_inliers = features

        # Collect outliers (data from other users): 50-50 split
        outlier_data = df[df['author'] != user]
        X_test_outliers = [extract_features(text) for text in outlier_data['content'].sample(n=len(X_test_inliers))]

        # True labels: 1 for inliers, -1 for outliers
        y_true_inliers = np.ones(len(X_test_inliers))
        y_true_outliers = -1 * np.ones(len(X_test_outliers))

        # Combine inliers and outliers for evaluation
        X_test_combined = X_test_inliers + X_test_outliers
        y_true = np.concatenate([y_true_inliers, y_true_outliers])

        # Evaluate model with given threshold
        y_pred = evaluate_model_with_threshold(model, scaler, X_test_combined, threshold)

        # Calculate precision, recall, F1, FAR, FRR for this user
        _, _, f1, far, frr = calculate_metrics(y_true, y_pred)

        # Update overall metrics
        overall_f1 += f1
        overall_far += far
        overall_frr += frr

    # Average metrics across all users
    avg_f1 = overall_f1 / total_users
    avg_far = overall_far / total_users
    avg_frr = overall_frr / total_users

    return avg_f1, avg_far, avg_frr

# Main grid search function
def main_test(file_path):
    df = preprocess_data(file_path)
    
    # Define grid for nu, gamma, and threshold
    nu_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
    gamma_values = [0.05, 0.1, 0.2, 0.5, 1.0]
    thresholds_to_test = [-0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]  # Thresholds to test

    # Iterate over all combinations of nu, gamma, and threshold
    for nu in nu_values:
        for gamma in gamma_values:
            for threshold in thresholds_to_test:
                avg_f1, avg_far, avg_frr = test_with_params(df, nu, gamma, threshold)
                
                # Print the results for this combination
                print(f"nu: {nu}, gamma: {gamma}, threshold: {threshold} -> F1: {avg_f1:.5f}, FAR: {avg_far:.5f}, FRR: {avg_frr:.5f}")

# Run the grid search
if __name__ == '__main__':
    data_path = 'data/cleaned.csv'
    main_test(data_path)
