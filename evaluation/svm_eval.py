import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load the dataset
df = pd.read_csv('subSmall_modified.csv')  # Adjust this path as needed

# Features for the SVM model
features = ['avg_word_length', 'function_word_ratio', 'punctuation_usage', 'pronoun_usage', 'lexical_diversity']

# Extract stylometric features
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
    
    return [avg_word_length, function_word_ratio, punctuation_usage, pronoun_usage, lexical_diversity]

# Preprocess the dataset and train models for each user
for user_id in df['username'].unique():
    user_data = df[df['username'] == user_id]['title']  # Assuming 'title' is the prompt
    
    # Extract features for each prompt
    feature_data = [extract_stylometric_features(text) for text in user_data]
    feature_df = pd.DataFrame(feature_data, columns=features)
    
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)
    
    # Train the One-Class SVM
    svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    
    # Perform cross-validation (5-fold)
    scores = cross_val_score(svm, X_scaled, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy for user {user_id}: {scores.mean()}")
    
    # Train final model on all user data
    svm.fit(X_scaled)
    
    # Save the model and scaler
    model_filename = f'models/{user_id}_svm_model.pkl'
    scaler_filename = f'models/{user_id}_scaler.pkl'
    joblib.dump(svm, model_filename)
    joblib.dump(scaler, scaler_filename)

    print(f"Model and scaler saved for user {user_id}.")
