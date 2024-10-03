import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset (replace with your specific dataset)
df = pd.read_csv('data/subClean.csv')

# Extract user-specific data
FEATURE_NAMES = ['avg_word_length', 'function_word_ratio', 'punctuation_usage', 'pronoun_usage', 'lexical_diversity', 'prompt_length']

# Function to extract features from text data
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

# Preprocessing: Extract features from the dataset for all users
features = []
labels = []

for index, row in df.iterrows():
    user = row['username']
    text = row['title']  # Adjust to match your column name for text
    features.append(extract_stylometric_features(text))
    labels.append(user)

features_df = pd.DataFrame(features, columns=FEATURE_NAMES)
features_df['user'] = labels

# Model Training and Evaluation
unique_users = features_df['user'].unique()

FAR_values = []
FRR_values = []

for user_id in unique_users:
    user_data = features_df[features_df['user'] == user_id].drop(columns=['user'])
    
    # Consider other users' data as impostors
    impostor_data = features_df[features_df['user'] != user_id].drop(columns=['user'])

    # Train-test split for user's data (80% train, 20% test)
    X_train, X_test = train_test_split(user_data, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    impostor_data_scaled = scaler.transform(impostor_data)

    # Train One-Class SVM
    svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
    svm.fit(X_train_scaled)

    # Predict on user test data (inliers) and impostor data (outliers)
    user_predictions = svm.predict(X_test_scaled)
    impostor_predictions = svm.predict(impostor_data_scaled)

    # Evaluation Metrics
    false_acceptances = np.sum(impostor_predictions == 1)  # Impostors classified as genuine
    false_rejections = np.sum(user_predictions == -1)  # Genuine user prompts classified as impostors

    FAR = false_acceptances / len(impostor_predictions)
    FRR = false_rejections / len(user_predictions)

    FAR_values.append(FAR)
    FRR_values.append(FRR)

    print(f"User: {user_id}")
    # print(f"False Acceptance Rate (FAR): {FAR:.2f}")
    # print(f"False Rejection Rate (FRR): {FRR:.2f}")
    # print("--------------------------------------")

# Overall Average FAR and FRR
print("Overall FAR:", np.mean(FAR_values))
print("Overall FRR:", np.mean(FRR_values))
