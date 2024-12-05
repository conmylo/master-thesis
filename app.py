import streamlit as st
import os
import nltk
import numpy as np
from src.extract_features import extract_features
from src.utils import load_model_and_scaler_with_distance

# config.py
NUS = [0.001, 0.005, 0.01]
GAMMAS = [0.05, 0.07, 0.1, 0.15, 0.2, 0.5]
MODEL_DIR = "models"
DATA_PATH = "data/cleaned.csv"
TEST_DATA_PATH = "data/filtered_cleaned.csv"
CONFIDENCE_THRESHOLD = 0.3

# Function to calculate certainty level
def calculate_certainty(model, sample_features, max_distance):
    distance = model.decision_function([sample_features])[0]
    abs_distance = abs(distance)
    certainty = (abs_distance / max_distance) * (1 if distance > 0 else -1) if abs_distance < max_distance else (1 if distance > 0 else -1)
    return certainty

# Weighted majority voting function
def weighted_majority_voting(models, scalers, distances, features):
    votes = 0
    scores = []
    for model, scaler, max_distance in zip(models, scalers, distances):
        scaled_features = scaler.transform([features])  # Ensure scaling aligns with updated features
        certainty = calculate_certainty(model, scaled_features[0], max_distance)
        prediction = model.predict(scaled_features)[0]
        weighted_vote = prediction * abs(certainty)
        votes += weighted_vote
        scores.append(abs(certainty))

    avg_certainty = np.mean(scores) if scores else 0
    final_decision = 1 if votes > 0 else -1

    return final_decision, avg_certainty

# Function to authenticate
def authenticate(username, text, nus, gammas, model_dir=MODEL_DIR, confidence_threshold=0.3):
    models, scalers, distances = [], [], []
    for nu in nus:
        for gamma in gammas:
            try:
                model, scaler, max_distance = load_model_and_scaler_with_distance(username, nu, gamma, model_dir)
                models.append(model)
                scalers.append(scaler)
                distances.append(max_distance)
            except FileNotFoundError:
                st.error(f"Model for user {username} with nu={nu} and gamma={gamma} not found.")
                return None

    # Extract features from input text
    features = extract_features(text)
    
    # Perform weighted majority voting
    decision, certainty_score = weighted_majority_voting(models, scalers, distances, features)

    # Return decision and certainty level
    if decision == 1:
        return "Access Granted", certainty_score
    else:
        return "Access Denied", certainty_score

# Streamlit app layout
st.title("Continuous Implicit Authentication System")
st.write("Select a username and enter a text prompt for authentication.")

trained_users = [
    "justinbieber", "taylorswift13", "BarackObama", "YouTube", 
    "ladygaga", "TheEllenShow", "Twitter", "jtimberlake", 
    "britneyspears", "Cristiano", "cnnbrk", "jimmyfallon", 
    "shakira", "Instagram"
]

username = st.selectbox("Select a username", trained_users)
text = st.text_area("Enter a text prompt")

if st.button("Authenticate"):
    if username and text:
        result, certainty_score = authenticate(username, text, NUS, GAMMAS)
        if result:
            if result == "Access Granted":
                st.success(f"{result}")
                st.write(f"Certainty Score: {certainty_score:.2f}")
            else:
                st.error(f"{result}")
                st.write(f"Certainty Score: {certainty_score:.2f}")

    else:
        st.warning("Please select a username and enter a prompt.")
