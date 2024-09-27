import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# Load the dataset
df = pd.read_csv('evaluation/data/subClean.csv')  # Adjust this path as needed

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
    
    # Mix in some data from other users as outliers (non-user data)
    non_user_data = df[df['username'] != user_id]['title']  # Data from other users
    non_user_feature_data = [extract_stylometric_features(text) for text in non_user_data]
    
    # Create a new DataFrame combining the user's data with non-user data (outliers)
    feature_df['label'] = 1  # User's data is labeled as 1 (positive class)
    non_user_df = pd.DataFrame(non_user_feature_data, columns=features)
    non_user_df['label'] = -1  # Non-user data is labeled as -1 (negative class)
    
    # Combine user and non-user data
    combined_df = pd.concat([feature_df, non_user_df], ignore_index=True)
    
    # Split the data into training and test sets (80/20 split)
    X = combined_df[features]
    y = combined_df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the One-Class SVM on the training data (only use positive class)
    svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)  # Adjust the nu parameter
    svm.fit(X_train_scaled[y_train == 1])  # Only train on the user's data
    
    # Save the model and scaler
    model_filename = f'{MODEL_DIR}/{user_id}_svm_model.pkl'
    scaler_filename = f'{MODEL_DIR}/{user_id}_scaler.pkl'
    joblib.dump(svm, model_filename)
    joblib.dump(scaler, scaler_filename)

    print(f"Model and scaler saved for user {user_id}.")
    
    # Evaluate the model on the test set
    decision_function_values = svm.decision_function(X_test_scaled)

    # Try adjusting the threshold here
    threshold = -0.01  # Adjust this value for more experimentation

    # Predict based on the threshold
    y_pred = [1 if score > threshold else -1 for score in decision_function_values]

    # Print the true labels and predicted labels to inspect the model's performance
    print("True labels (y_test):", y_test)
    print("Predicted labels (y_pred):", y_pred)

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
    recall = recall_score(y_test, y_pred, pos_label=1, average='binary')
    f1 = f1_score(y_test, y_pred, pos_label=1, average='binary')

    # Calculate FAR and FRR
    false_acceptances = sum((y_test == -1) & (y_pred == 1))  # Non-user data classified as user
    false_rejections = sum((y_test == 1) & (y_pred == -1))  # User data classified as non-user

    total_outliers = sum(y_test == -1)  # Total non-user data  
    total_inliers = sum(y_test == 1)    # Total user data

    far = false_acceptances / total_outliers if total_outliers > 0 else 0.0
    frr = false_rejections / total_inliers if total_inliers > 0 else 0.0

    # Print results for inspection
    print(f"Evaluation for user {user_id}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"False Acceptance Rate (FAR): {far}")
    print(f"False Rejection Rate (FRR): {frr}")
    print("--------------------------------------------------------")
