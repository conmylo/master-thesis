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

# Extract features
def extract_features(text):
    features = {}
    words = nltk.word_tokenize(text)
    char_count = len(text) if len(text) > 0 else 1  # Avoid division by zero
    
    # 1. Character Count
    features['char_count'] = char_count
    
    # 2. Character N-grams (3-grams)
    features['char_ngrams_3'] = len([text[i:i+3] for i in range(len(text) - 2)]) / char_count
    
    # 3. Stop Word Frequency
    stop_words = set(stopwords.words('english'))
    stop_word_count = sum(1 for word in words if word.lower() in stop_words)
    features['stop_word_freq'] = stop_word_count / len(words) if words else 0
    
    # 4. Exclamation Count
    exclamation_count = text.count('!')
    features['exclamation_count'] = exclamation_count / char_count
    
    # 5. Question Count
    question_count = text.count('?')
    features['question_count'] = question_count / char_count
    
    # 6. Word Length Average
    features['word_length_avg'] = np.mean([len(word) for word in words]) if words else 0
    
    # 7. Type-Token Ratio
    features['type_token_ratio'] = len(set(words)) / len(words) if words else 0
    
    # 8. Uppercase Letter Proportion
    features['uppercase_proportion'] = sum(1 for c in text if c.isupper()) / char_count
    
    # 9. Digit Proportion
    features['digit_proportion'] = sum(1 for c in text if c.isdigit()) / char_count
    
    # 10. Punctuation Proportion
    punctuation_count = sum(1 for c in text if c in '.,!?')
    features['punctuation_proportion'] = punctuation_count / char_count
    
    # 11. Adjective Ratio (Using POS tagging)
    pos_tags = nltk.pos_tag(words)
    pos_counts = Counter(tag for word, tag in pos_tags)
    features['adjective_ratio'] = pos_counts.get('JJ', 0) / len(words) if words else 0
    
    # 12. Noun Ratio (Using POS tagging)
    features['noun_ratio'] = pos_counts.get('NN', 0) / len(words) if words else 0
    
    # 13. Average Sentence Length (words per sentence)
    sentences = nltk.sent_tokenize(text)
    features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
    
    # 14. Pronoun Usage Proportion
    pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'us', 'him', 'her', 'them'}
    pronoun_count = sum(1 for word in words if word.lower() in pronouns)
    features['pronoun_usage_proportion'] = pronoun_count / len(words) if words else 0
    
    # 15. Hapax Legomena Ratio
    word_freq = Counter(words)
    hapax_legomena_count = sum(1 for word in word_freq if word_freq[word] == 1)
    features['hapax_legomena_ratio'] = hapax_legomena_count / len(words) if words else 0
    
    # 16. Readability Score (Flesch Reading Ease)
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

# Test all users with a fixed threshold
def test_with_saved_models(df, threshold=0.0, model_dir="models"):
    overall_f1, overall_far, overall_frr = 0, 0, 0
    total_users = len(df['author'].unique())

    for user, group in df.groupby('author'):
        # Load saved model and scaler for the user
        model, scaler = load_model_and_scaler(user, model_dir)

        # Extract features for inliers
        X_test_inliers = [extract_features(text) for text in group['content']]

        # Collect outliers (data from other users): 50-50 split
        outlier_data = df[df['author'] != user]
        X_test_outliers = [extract_features(text) for text in outlier_data['content'].sample(n=len(X_test_inliers))]

        # True labels: 1 for inliers, -1 for outliers
        y_true_inliers = np.ones(len(X_test_inliers))
        y_true_outliers = -1 * np.ones(len(X_test_outliers))

        # Combine inliers and outliers for evaluation
        X_test_combined = X_test_inliers + X_test_outliers
        y_true = np.concatenate([y_true_inliers, y_true_outliers])

        # Evaluate model with the given threshold
        y_pred = evaluate_model_with_threshold(model, scaler, X_test_combined, threshold)

        # Calculate precision, recall, F1, FAR, FRR for this user
        _, _, f1, far, frr = calculate_metrics(y_true, y_pred)

        # Update overall metrics
        overall_f1 += f1
        overall_far += far
        overall_frr += frr

        # Print individual results for each user
        print(f"User {user}: F1: {f1:.5f}, FAR: {far:.5f}, FRR: {frr:.5f}")

    # Average metrics across all users
    avg_f1 = overall_f1 / total_users
    avg_far = overall_far / total_users
    avg_frr = overall_frr / total_users

    # Print overall results
    print(f"\nOverall F1 Score: {avg_f1:.5f}")
    print(f"Overall FAR (False Acceptance Rate): {avg_far:.5f}")
    print(f"Overall FRR (False Rejection Rate): {avg_frr:.5f}")

# Main testing function
def main_test(file_path, model_dir="models", threshold=0.0):
    df = preprocess_data(file_path)
    test_with_saved_models(df, threshold, model_dir)

# Run the testing phase
if __name__ == '__main__':
    data_path = 'data/cleaned.csv'  # Path to cleaned data
    threshold = 0.1115  # Example threshold value; can be adjusted
    main_test(data_path, threshold=threshold)
