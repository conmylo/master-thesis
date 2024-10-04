import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# Load the dataset
data_path = 'twitterDataset20/data/tweets.csv'
df = pd.read_csv(data_path)

# Define columns for analysis and feature extraction
FEATURE_NAMES = ['avg_word_length', 'function_word_ratio', 'punctuation_usage', 
                 'pronoun_usage', 'lexical_diversity', 'prompt_length', 
                 'number_of_likes', 'number_of_shares']

# Function to extract stylometric features from tweet content
def extract_stylometric_features(text):
    words = text.split()
    unique_words = set(words)

    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    lexical_diversity = len(unique_words) / len(words) if words else 0
    function_words = {'the', 'and', 'is', 'in', 'on', 'at', 'by', 'with', 'a', 'an'}
    function_word_ratio = sum(1 for word in words if word in function_words) / len(words) if words else 0
    punctuation_count = sum(1 for char in text if char in '.,!?')
    punctuation_usage = punctuation_count / len(text) if text else 0
    pronoun_usage = sum(1 for word in words if word.lower() in ['i', 'he', 'she', 'we', 'they']) / len(words) if words else 0
    prompt_length = len(text)

    return [avg_word_length, function_word_ratio, punctuation_usage, pronoun_usage, lexical_diversity, prompt_length]

# Apply stylometric feature extraction to each tweet
stylometric_features = df['content'].apply(extract_stylometric_features)
stylometric_features_df = pd.DataFrame(stylometric_features.tolist(), columns=FEATURE_NAMES[:6])

# Add extra features like number_of_likes and number_of_shares
stylometric_features_df['number_of_likes'] = df['number_of_likes']
stylometric_features_df['number_of_shares'] = df['number_of_shares']

# Extract TF-IDF Features
tfidf_vectorizer = TfidfVectorizer(max_features=100)  # Limit to 100 features to reduce complexity
tfidf_features = tfidf_vectorizer.fit_transform(df['content'])

# Convert TF-IDF features to DataFrame
tfidf_features_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Combine Stylometric and TF-IDF Features
features_df = pd.concat([stylometric_features_df, tfidf_features_df], axis=1)
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
        svm = OneClassSVM(kernel='rbf', gamma=0.005, nu=0.03)
        svm.fit(X_train_scaled)

        # Predict on user test data (inliers) and impostor data (outliers)
        user_decision_scores = svm.decision_function(X_test_scaled)
        impostor_decision_scores = svm.decision_function(impostor_data_scaled)

        # Adjust decision threshold dynamically
        # Use a custom threshold, e.g., 5th percentile of genuine user's decision scores
        decision_threshold = np.percentile(user_decision_scores, 5)

        # Classify based on adjusted threshold
        user_predictions = [1 if score > decision_threshold else -1 for score in user_decision_scores]
        impostor_predictions = [1 if score > decision_threshold else -1 for score in impostor_decision_scores]

        # Calculate False Acceptance and False Rejection
        false_acceptances = np.sum(np.array(impostor_predictions) == 1)  # Impostors classified as genuine
        false_rejections = np.sum(np.array(user_predictions) == -1)  # Genuine user prompts classified as impostors

        # Calculate FAR and FRR
        FAR = false_acceptances / len(impostor_predictions)
        FRR = false_rejections / len(user_predictions)

        # Store FAR and FRR for each fold
        FAR_values.append(FAR)
        FRR_values.append(FRR)

# Calculate overall FAR and FRR
overall_FAR = np.mean(FAR_values)
overall_FRR = np.mean(FRR_values)

print(f"Overall False Acceptance Rate (FAR): {overall_FAR:.2f}")
print(f"Overall False Rejection Rate (FRR): {overall_FRR:.2f}")