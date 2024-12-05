import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import save_model_and_scaler
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from extract_features import extract_features
from config import MODEL_DIR

def preprocess_data(file_path):
    """
    Reads the dataset and selects relevant columns.
    """
    df = pd.read_csv(file_path)
    return df[['author', 'content']]

def train_and_save_models_with_split(file_path, nus, gammas, model_dir=MODEL_DIR):
    """
    For each user, performs a train-test split, extracts features, and trains multiple One-Class SVM models
    with different hyperparameters. Each model and its scaler are saved for future use.
    """
    df = preprocess_data(file_path)
    
    # Ensure model directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    for user, group in df.groupby('author'):
        # Split each user's data into training (85%) and testing (15%)
        train_texts, test_texts = train_test_split(group['content'], test_size=0.15, random_state=42)
        
        # Save test texts for bulk testing
        test_texts.to_csv(f"{model_dir}/user_{user}_test.csv", index=False, header=["content"])
        
        # Extract features from training data
        train_features = [extract_features(text) for text in train_texts]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_features)
        
        # Train multiple OCSVM models for each user with different nu and gamma values
        for nu in nus:
            for gamma in gammas:
                model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
                model.fit(X_train_scaled)
                
                # Save model and scaler
                save_model_and_scaler(user, model, scaler, nu, gamma, model_dir)
                print(f"Model saved for user: {user}, nu: {nu}, gamma: {gamma}")