import os
import pickle
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

def save_model_and_scaler(user, model, scaler, nu, gamma, model_dir):
    """
    Saves the trained One-Class SVM model, scaler, and max_distance to a file.
    max_distance is calculated using distances from training data points to ensure consistency.
    """
    # Calculate max_distance directly from training data distances
    distances = model.decision_function(scaler.transform(model.support_vectors_))
    max_distance = max(abs(distances))
    
    # Paths to save model, scaler, and max_distance
    model_path = f"{model_dir}/user_{user}_nu_{nu}_gamma_{gamma}_model.pkl"
    scaler_path = f"{model_dir}/user_{user}_nu_{nu}_gamma_{gamma}_scaler.pkl"
    distance_path = f"{model_dir}/user_{user}_nu_{nu}_gamma_{gamma}_distance.pkl"
    
    # Save model, scaler, and max_distance
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(distance_path, 'wb') as f:
        pickle.dump(max_distance, f)
    print(f"Model, scaler, and max_distance saved for user {user}, nu {nu}, gamma {gamma}")

def load_model_and_scaler_with_distance(user, nu, gamma, model_dir):
    """
    Loads the One-Class SVM model, scaler, and max_distance for a specific user and hyperparameters.
    """
    # Define file paths
    model_path = f"{model_dir}/user_{user}_nu_{nu}_gamma_{gamma}_model.pkl"
    scaler_path = f"{model_dir}/user_{user}_nu_{nu}_gamma_{gamma}_scaler.pkl"
    distance_path = f"{model_dir}/user_{user}_nu_{nu}_gamma_{gamma}_distance.pkl"
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load max_distance
    with open(distance_path, 'rb') as f:
        max_distance = pickle.load(f)
    
    return model, scaler, max_distance
