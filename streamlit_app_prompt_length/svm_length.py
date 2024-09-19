import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ensure the models directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# Function to train and save an SVM model for a user based on normalized prompt length
def train_svm_for_user(df, user_id):
    # Filter the dataset by user_id
    user_data = df[df['user_id'] == user_id]
    
    # Extract the prompt lengths as the feature
    prompt_lengths = np.array(user_data['prompt_length']).reshape(-1, 1)
    
    # Normalize the prompt lengths
    scaler = StandardScaler()
    normalized_prompt_lengths = scaler.fit_transform(prompt_lengths)
    
    # Train One-Class SVM model on normalized data
    svm_model = OneClassSVM(kernel="rbf", gamma=0.01, nu=0.1).fit(normalized_prompt_lengths)
    
    # Save the model and the scaler
    joblib.dump(svm_model, f"models/{user_id}_svm_model.pkl")
    joblib.dump(scaler, f"models/{user_id}_scaler.pkl")
    
    print(f"Model and scaler for {user_id} saved!")

# Example usage
if __name__ == "__main__":
    # Load the dataset from csv_svm_prompt_length.csv
    dataset_path = 'streamlit_app_prompt_length/data/csv_svm_prompt_length.csv'
    df = pd.read_csv(dataset_path)

    # Train models for each user
    for user_id in df['user_id'].unique():
        train_svm_for_user(df, user_id)
