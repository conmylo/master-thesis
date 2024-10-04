import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Load the dataset
data_path = 'twitterDataset20/data/tweets.csv'
df = pd.read_csv(data_path)

# Define columns for feature analysis
BASE_FEATURE_NAMES = [
    'avg_word_length', 'function_word_ratio', 'punctuation_usage',
    'pronoun_usage', 'lexical_diversity', 'prompt_length', 
    'avg_sentence_length', 'stopword_ratio', 'hapax_legomena_ratio', 
    'special_char_ratio', 'noun_ratio', 'verb_ratio', 'adj_ratio'
]

# Function to extract stylometric, linguistic, and POS ratio features
def extract_features(text):
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
    avg_sentence_length = len(words) / (text.count('.') + text.count('!') + text.count('?') + 1e-5)  # To avoid division by zero

    stopwords_set = set(stopwords.words('english'))
    stopword_ratio = sum(1 for word in words if word in stopwords_set) / len(words) if words else 0
    hapax_legomena_ratio = len([word for word in unique_words if words.count(word) == 1]) / len(words) if words else 0
    special_char_ratio = sum(1 for char in characters if char in string.punctuation) / len(characters) if characters else 0

    # Part of Speech ratios
    num_nouns = sum(1 for word, pos in pos_tags if pos.startswith('NN'))
    num_verbs = sum(1 for word, pos in pos_tags if pos.startswith('VB'))
    num_adjs = sum(1 for word, pos in pos_tags if pos.startswith('JJ'))

    noun_ratio = num_nouns / len(words) if words else 0
    verb_ratio = num_verbs / len(words) if words else 0
    adj_ratio = num_adjs / len(words) if words else 0

    return [
        avg_word_length, function_word_ratio, punctuation_usage, pronoun_usage, lexical_diversity, prompt_length,
        avg_sentence_length, stopword_ratio, hapax_legomena_ratio, special_char_ratio, noun_ratio, verb_ratio, adj_ratio
    ]

# Apply feature extraction to each tweet
features = df['content'].apply(extract_features)
features_df = pd.DataFrame(features.tolist(), columns=BASE_FEATURE_NAMES)

# Prepare the dataset for training and testing
features_df['author'] = df['author']
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
        svm = OneClassSVM(kernel='rbf', gamma=0.005, nu=0.03)
        svm.fit(X_train_scaled)

        # Predict on user test data (inliers) and impostor data (outliers)
        user_decision_scores = svm.decision_function(X_test_scaled)
        impostor_decision_scores = svm.decision_function(impostor_data_scaled)

        # Adjust decision threshold dynamically
        decision_threshold = np.percentile(user_decision_scores, 5)

        # Classify based on adjusted threshold
        user_predictions = [1 if score > decision_threshold else -1 for score in user_decision_scores]
        impostor_predictions = [1 if score > decision_threshold else -1 for score in impostor_decision_scores]

        # Calculate False Acceptance and False Rejection
        false_acceptances = np.sum(np.array(impostor_predictions) == 1)  # Impostors classified as genuine
        false_rejections = np.sum(np.array(user_predictions) == -1)  # Genuine user prompts classified as impostors

        # Calculate FAR and FRR
        FAR = false_acceptances / len(impostor_predictions) if len(impostor_predictions) > 0 else 0
        FRR = false_rejections / len(user_predictions) if len(user_predictions) > 0 else 0

        # Store FAR and FRR for each fold
        FAR_values.append(FAR)
        FRR_values.append(FRR)

# Calculate overall FAR and FRR
overall_FAR = np.mean(FAR_values)
overall_FRR = np.mean(FRR_values)

print(f"Overall False Acceptance Rate (FAR): {overall_FAR:.2f}")
print(f"Overall False Rejection Rate (FRR): {overall_FRR:.2f}")
