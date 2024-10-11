from sklearn.svm import OneClassSVM
import pickle
import os

def train_model(X_train):
    """Train a One-Class SVM model."""
    model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    model.fit(X_train)
    return model

def save_model(model, user_id, model_dir="models/"):
    """Save the trained model to disk."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(f"{model_dir}/user_{user_id}_model.pkl", 'wb') as f:
        pickle.dump(model, f)

def load_model(user_id, model_dir="models/"):
    """Load a saved model from disk."""
    with open(f"{model_dir}/user_{user_id}_model.pkl", 'rb') as f:
        return pickle.load(f)
