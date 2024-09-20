import os
import pandas as pd
from sklearn.svm import OneClassSVM
import joblib

# Load the dataset without avg_word_length
df = pd.read_csv('streamlit_app_stylometric/data/stylometric.csv')

# Select features for training
features = [
    'lexical_diversity', 'function_word_ratio', 'punctuation_usage',
    'pronoun_usage', 'vocabulary_richness'
]

# Create the 'models' folder if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Function to train and save SVM model for each user
def train_svm_for_user(df, user_id):
    # Filter the dataset for the current user
    user_data = df[df['user_id'] == user_id]
    
    # Extract features for the current user (without normalization)
    X = user_data[features]
    
    # Check if there is sufficient data for the user
    if len(X) > 0:
        # Train the One-Class SVM model
        svm = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.1)
        svm.fit(X)
        
        # Save the model with a consistent naming scheme
        model_filename = f'models/{user_id}_svm_model.pkl'
        joblib.dump(svm, model_filename)
        print(f"Model for {user_id} saved as {model_filename}")
    else:
        print(f"No data available for user {user_id}. Skipping model training.")

# Train models for each user in the dataset
for user_id in df['user_id'].unique():
    train_svm_for_user(df, user_id)

print("All models have been trained and saved.")
