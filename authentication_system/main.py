from src.data_preprocessing import preprocess_data
from src.feature_extraction import extract_features
from src.model import train_model, save_model, load_model
from src.evaluation import bulk_evaluate
import pandas as pd

# Load and preprocess the dataset
data_path = 'data/cleaned.csv'
df = preprocess_data(data_path)

# Extract features for each user
users_features = {}
for user, group in df.groupby('author'):
    features = [extract_features(text) for text in group['content']]
    users_features[user] = features

# Train and save models for each user
for user, features in users_features.items():
    model = train_model(features)
    save_model(model, user)

# Evaluation phase (this assumes you've split data earlier)
# Load models, run tests, and print metrics
models = {user: load_model(user) for user in users_features.keys()}
# (Placeholder for test data and labels)
X_tests, y_trues = {}, {}  # Load or split your test data here
results = bulk_evaluate(models, X_tests, y_trues)

for user, precision, recall, f1 in results:
    print(f"User {user}: Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
