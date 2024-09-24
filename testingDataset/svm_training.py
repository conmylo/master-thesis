import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import joblib

# Load the dataset
df = pd.read_csv('data/subSmall.csv', on_bad_lines='skip')

# Directory to store the models
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Features for the SVM model
features = ['avg_word_length', 'function_word_ratio', 'punctuation_usage', 'pronoun_usage', 'lexical_diversity']

# Extract stylometric features
def extract_stylometric_features(text):
    words = text.split()
    unique_words = set(words)
    
    # Feature calculations
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    lexical_diversity = len(unique_words) / len(words) if words else 0
    
    function_words = {'the', 'and', 'is', 'in', 'on', 'at', 'by', 'with', 'a', 'an'}
    function_word_ratio = sum(1 for word in words if word in function_words) / len(words) if words else 0
    
    punctuation_count = sum(1 for char in text if char in '.,!?')
    punctuation_usage = punctuation_count / len(text) if text else 0
    
    pronoun_usage = sum(1 for word in words if word.lower() in ['i', 'he', 'she', 'we', 'they']) / len(words) if words else 0
    
    return [avg_word_length, function_word_ratio, punctuation_usage, pronoun_usage, lexical_diversity]

# Preprocess the dataset and train models for each user
for user_id in df['username'].unique():
    user_data = df[df['username'] == user_id]['text']
    
    # Extract features for each prompt
    feature_data = [extract_stylometric_features(text) for text in user_data]
    feature_df = pd.DataFrame(feature_data, columns=features)
    
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)
    
    # Train the One-Class SVM
    svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    svm.fit(X_scaled)
    
    # Save the model and scaler
    model_filename = f'{MODEL_DIR}/{user_id}_svm_model.pkl'
    scaler_filename = f'{MODEL_DIR}/{user_id}_scaler.pkl'
    joblib.dump(svm, model_filename)
    joblib.dump(scaler, scaler_filename)
    
    print(f"Model and scaler saved for user {user_id}.")
