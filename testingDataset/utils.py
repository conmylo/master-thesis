import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import re
import string

# Function to load SVM model for a user
def load_model(user_id):
    model_filename = f'models/{user_id}_svm_model.pkl'
    try:
        model = joblib.load(model_filename)
        return model
    except FileNotFoundError:
        return None

# Function to extract features from the input text
def extract_features(raw_text):
    text_no_punct = re.sub(f"[{string.punctuation}]", "", raw_text)
    words = text_no_punct.split()
    unique_words = set(words)

    # Lexical diversity (unique words / total words)
    lexical_diversity = len(unique_words) / len(words) if words else 0

    # Function word ratio (commonly used words like "the", "and", etc.)
    function_words = {'the', 'and', 'is', 'in', 'on', 'at', 'by', 'with', 'a', 'an'}
    function_word_ratio = sum(1 for word in words if word.lower() in function_words) / len(words) if words else 0

    # Punctuation usage (frequency of punctuation in the text)
    punctuation_count = sum(1 for char in raw_text if char in string.punctuation)
    punctuation_usage = punctuation_count / len(raw_text) if len(raw_text) > 0 else 0

    # Pronoun usage (frequency of personal pronouns like "I", "he", "we", etc.)
    pronouns = {'i', 'you', 'he', 'she', 'we', 'they', 'me', 'us', 'him', 'her', 'them'}
    pronoun_usage = sum(1 for word in words if word.lower() in pronouns) / len(words) if words else 0

    # Vocabulary richness (ratio of unique words to total words)
    vocabulary_richness = len(unique_words) / len(words) if words else 0

    return pd.DataFrame([[lexical_diversity, function_word_ratio, punctuation_usage,
                          pronoun_usage, vocabulary_richness]],
                        columns=['lexical_diversity', 'function_word_ratio', 'punctuation_usage',
                                 'pronoun_usage', 'vocabulary_richness'])

# Function to train and save the SVM model for a specific user
def train_svm_model(username, user_data):
    try:
        # Extract features from each user's data (assuming 'title' column contains the posts)
        feature_list = []
        for text in user_data['title']:  # assuming 'title' column has the text data
            features = extract_features(text)
            feature_list.append(features)

        # Concatenate the features into a single DataFrame
        user_features = pd.concat(feature_list, ignore_index=True)

        # Scale the features for better SVM performance
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(user_features)

        # Train the One-Class SVM model
        svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
        svm.fit(scaled_features)

        # Ensure models directory exists
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)

        # Save the model and scaler
        model_filename = os.path.join(models_dir, f'{username}_svm_model.pkl')
        joblib.dump((svm, scaler), model_filename)

        # Debug: Check if the model is saved correctly
        if os.path.exists(model_filename):
            print(f"Model saved successfully at {model_filename}")
            return model_filename
        else:
            print(f"Failed to save the model at {model_filename}")
            return None
    except Exception as e:
        print(f"An error occurred while training the model: {e}")
        return None
