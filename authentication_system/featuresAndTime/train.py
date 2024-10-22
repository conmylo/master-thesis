import os
import pickle
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
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

# Extract time of day category from timestamp
def time_of_day(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

# Convert time of day to a numerical feature
def time_of_day_to_numeric(time_of_day):
    mapping = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
    return mapping.get(time_of_day, -1)

# Extract features
def extract_features(text, timestamp):
    features = {}
    words = nltk.word_tokenize(text)
    char_count = len(text) if len(text) > 0 else 1  # Avoid division by zero

    # Time of Day Feature
    hour = pd.to_datetime(timestamp).hour
    features['time_of_day'] = time_of_day_to_numeric(time_of_day(hour))

    # Other existing features...
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

# Train and save models for each user
def train_and_save_models(file_path, nu, gamma, model_dir="models"):
    df = preprocess_data(file_path)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for user, group in df.groupby('author'):
        features = [extract_features(text, timestamp) for text, timestamp in zip(group['content'], group['date_time'])]
        
        # Train One-Class SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(features)
        model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
        model.fit(X_train_scaled)

        # Save model and scaler
        with open(f"{model_dir}/user_{user}_model.pkl", 'wb') as f:
            pickle.dump(model, f)
        with open(f"{model_dir}/user_{user}_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        print(f"Model and scaler saved for user: {user}")

# Main training function
if __name__ == '__main__':
    data_path = 'data/cleaned.csv'
    nu, gamma = 0.005, 0.05  # Updated parameters for nu and gamma
    train_and_save_models(data_path, nu, gamma)
