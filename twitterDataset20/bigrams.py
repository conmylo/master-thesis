import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter
import nltk
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load dataset (Replace with your actual CSV file path)
df = pd.read_csv('twitterDataset20/data/tweets.csv')

# Define columns for analysis and feature extraction
FEATURE_NAMES = [
    'avg_word_length', 'function_word_ratio', 'punctuation_usage', 
    'pronoun_usage', 'lexical_diversity', 'prompt_length',
    'capitalization_ratio', 'stopword_ratio', 'special_char_ratio',
    'bigram_freq_ratio', 'trigram_freq_ratio', 'pos_tag_diversity'
]

# Function to extract stylometric features from tweet content
def extract_stylometric_features(text):
    words = word_tokenize(text)
    unique_words = set(words)
    characters = list(text)
    pos_tags = pos_tag(words)

    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    lexical_diversity = len(unique_words) / len(words) if words else 0
    function_words = {'the', 'and', 'is', 'in', 'on', 'at', 'by', 'with', 'a', 'an'}
    function_word_ratio = sum(1 for word in words if word in function_words) / len(words) if words else 0
    punctuation_count = sum(1 for char in text if char in '.,!?')
    punctuation_usage = punctuation_count / len(text) if text else 0
    pronoun_usage = sum(1 for word in words if word.lower() in ['i', 'he', 'she', 'we', 'they']) / len(words) if words else 0
    prompt_length = len(text)

    # Capitalization Ratio
    capitalization_count = sum(1 for char in text if char.isupper())
    capitalization_ratio = capitalization_count / len(characters) if characters else 0

    # Stopword Ratio
    stopwords_set = set(stopwords.words('english'))
    stopword_ratio = sum(1 for word in words if word in stopwords_set) / len(words) if words else 0

    # Special Character Ratio
    special_char_ratio = sum(1 for char in characters if char in string.punctuation) / len(characters) if characters else 0

    # Bigram Frequency Ratio
    bigrams = list(nltk.bigrams(words))
    bigram_freq_ratio = len(bigrams) / len(words) if words else 0

    # Trigram Frequency Ratio
    trigrams = list(nltk.trigrams(words))
    trigram_freq_ratio = len(trigrams) / len(words) if words else 0

    # POS Tag Diversity
    pos_counts = Counter(tag for word, tag in pos_tags)
    pos_tag_diversity = len(pos_counts) / len(words) if words else 0

    return [
        avg_word_length, function_word_ratio, punctuation_usage, pronoun_usage, lexical_diversity, prompt_length,
        capitalization_ratio, stopword_ratio, special_char_ratio,
        bigram_freq_ratio, trigram_freq_ratio, pos_tag_diversity
    ]

# Apply feature extraction to each tweet
features = df['content'].apply(extract_stylometric_features)
features_df = pd.DataFrame(features.tolist(), columns=FEATURE_NAMES)

# Add author column
features_df['author'] = df['author']

# Prepare the dataset for training and testing
unique_users = features_df['author'].unique()

FAR_values = []
FRR_values = []

# Use K-Fold Cross Validation (e.g., 5-fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over each user to train and evaluate their individual model
for user_id in unique_users:
    user_data = features_df[features_df['author'] == user_id].drop(columns=['author'])
    impostor_data = features_df[features_df['author'] != user_id].drop(columns=['author'])

    # Split the user's data into K-Folds
    for train_index, test_index in kf.split(user_data):
        X_train, X_test = user_data.iloc[train_index], user_data.iloc[test_index]

        # Normalize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        impostor_data_scaled = scaler.transform(impostor_data)

        # Train One-Class SVM
        svm = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.03)
        svm.fit(X_train_scaled)

        # Train Isolation Forest
        iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        iso_forest.fit(X_train_scaled)

        # Predict on user test data (inliers) and impostor data (outliers) for both models
        user_svm_predictions = svm.predict(X_test_scaled)
        user_iso_predictions = iso_forest.predict(X_test_scaled)
        
        impostor_svm_predictions = svm.predict(impostor_data_scaled)
        impostor_iso_predictions = iso_forest.predict(impostor_data_scaled)

        # Combine predictions using majority voting
        def ensemble_prediction(svm_pred, iso_pred):
            if svm_pred == iso_pred:
                return svm_pred  # If both models agree, use their prediction
            else:
                return -1  # If they disagree, consider it an impostor for security

        # Get final predictions for user and impostor data
        user_predictions = [ensemble_prediction(svm_pred, iso_pred) for svm_pred, iso_pred in zip(user_svm_predictions, user_iso_predictions)]
        impostor_predictions = [ensemble_prediction(svm_pred, iso_pred) for svm_pred, iso_pred in zip(impostor_svm_predictions, impostor_iso_predictions)]

        # Calculate False Acceptance and False Rejection
        false_acceptances = np.sum(np.array(impostor_predictions) == 1)
        false_rejections = np.sum(np.array(user_predictions) == -1)

        # Calculate FAR and FRR
        FAR = false_acceptances / len(impostor_predictions) if len(impostor_predictions) > 0 else 0
        FRR = false_rejections / len(user_predictions) if len(user_predictions) > 0 else 0

        # Store FAR and FRR for each fold
        FAR_values.append(FAR)
        FRR_values.append(FRR)

# Calculate overall FAR and FRR
overall_FAR = np.mean(FAR_values)
overall_FRR = np.mean(FRR_values)

# Print overall FAR and FRR
print(f"Overall False Acceptance Rate (FAR): {overall_FAR:.2f}")
print(f"Overall False Rejection Rate (FRR): {overall_FRR:.2f}")
