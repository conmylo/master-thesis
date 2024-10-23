import os
import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import nltk
from collections import Counter
from nltk.corpus import stopwords
import textstat

nltk.download('punkt')
nltk.download('stopwords')

# Preprocess dataset
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    return df[['author', 'content', 'date_time', 'id']]

# Extract features (without time of day)
def extract_features(text):
    features = {}
    words = nltk.word_tokenize(text)
    char_count = len(text) if len(text) > 0 else 1  # Avoid division by zero
    
    # Existing features...
    features['char_count'] = char_count
    features['char_ngrams_3'] = len([text[i:i+3] for i in range(len(text) - 2)]) / char_count
    stop_words = set(stopwords.words('english'))
    stop_word_count = sum(1 for word in words if word.lower() in stop_words)
    features['stop_word_freq'] = stop_word_count / len(words) if words else 0
    exclamation_count = text.count('!')
    features['exclamation_count'] = exclamation_count / char_count
    question_count = text.count('?')
    features['question_count'] = question_count / char_count
    features['word_length_avg'] = np.mean([len(word) for word in words]) if words else 0
    features['type_token_ratio'] = len(set(words)) / len(words) if words else 0
    features['uppercase_proportion'] = sum(1 for c in text if c.isupper()) / char_count
    features['digit_proportion'] = sum(1 for c in text if c.isdigit()) / char_count
    punctuation_count = sum(1 for c in text if c in '.,!?')
    features['punctuation_proportion'] = punctuation_count / char_count

    pos_tags = nltk.pos_tag(words)
    pos_counts = Counter(tag for word, tag in pos_tags)
    features['adjective_ratio'] = pos_counts.get('JJ', 0) / len(words) if words else 0
    features['noun_ratio'] = pos_counts.get('NN', 0) / len(words) if words else 0
    sentences = nltk.sent_tokenize(text)
    features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0

    pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'us', 'him', 'her', 'them'}
    pronoun_count = sum(1 for word in words if word.lower() in pronouns)
    features['pronoun_usage_proportion'] = pronoun_count / len(words) if words else 0

    word_freq = Counter(words)
    hapax_legomena_count = sum(1 for word in word_freq if word_freq[word] == 1)
    features['hapax_legomena_ratio'] = hapax_legomena_count / len(words) if words else 0

    features['readability_score'] = textstat.flesch_reading_ease(text)

    return list(features.values())

# Load model and scaler
def load_model_and_scaler(user, model_dir="models"):
    with open(f"{model_dir}/user_{user}_model.pkl", 'rb') as f:
        model = pickle.load(f)
    with open(f"{model_dir}/user_{user}_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Evaluate model on test data with fixed decision threshold
def evaluate_model_with_threshold(model, scaler, X_test, threshold=0.0):
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

# Test only specified users
def test_specified_users(df, threshold=0.0, model_dir="models"):
    specified_users = ['BarackObama', 'cnnbrk', 'instagram', 'shakira', 'Twitter', 'justinbieber', 'jtimberlake']
    df = df[df['author'].isin(specified_users)]
    
    overall_f1, overall_far, overall_frr = 0, 0, 0
    total_users = len(df['author'].unique())

    for user, group in df.groupby('author'):
        model, scaler = load_model_and_scaler(user, model_dir)
        X_test_inliers = [extract_features(text) for text in group['content']]
        outlier_data = df[df['author'] != user]
        X_test_outliers = [extract_features(text) for text in outlier_data['content'].sample(n=len(X_test_inliers))]

        y_true_inliers = np.ones(len(X_test_inliers))
        y_true_outliers = -1 * np.ones(len(X_test_outliers))
        X_test_combined = X_test_inliers + X_test_outliers
        y_true = np.concatenate([y_true_inliers, y_true_outliers])
        y_pred = evaluate_model_with_threshold(model, scaler, X_test_combined, threshold)
        _, _, f1, far, frr = calculate_metrics(y_true, y_pred)

        overall_f1 += f1
        overall_far += far
        overall_frr += frr

        print(f"User {user}: F1: {f1:.5f}, FAR: {far:.5f}, FRR: {frr:.5f}")

    avg_f1 = overall_f1 / total_users
    avg_far = overall_far / total_users
    avg_frr = overall_frr / total_users

    print(f"\nOverall F1 Score: {avg_f1:.5f}")
    print(f"Overall FAR (False Acceptance Rate): {avg_far:.5f}")
    print(f"Overall FRR (False Rejection Rate): {avg_frr:.5f}")

# Main testing function
def main_test(file_path, model_dir="models", threshold=0.1115):
    df = preprocess_data(file_path)
    test_specified_users(df, threshold, model_dir)

# Run the testing phase
if __name__ == '__main__':
    data_path = 'data/cleaned.csv'
    threshold = 0.1115
    main_test(data_path, threshold=threshold)
