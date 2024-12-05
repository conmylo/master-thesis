import os
import pandas as pd
import numpy as np
from src.utils import load_model_and_scaler_with_distance
from src.extract_features import extract_features
from src.config import MODEL_DIR


def calculate_certainty(model, sample_features, max_distance):
    distance = model.decision_function([sample_features])[0]
    abs_distance = abs(distance)
    certainty = (abs_distance / max_distance) * (1 if distance > 0 else -1) if abs_distance < max_distance else (1 if distance > 0 else -1)
    return certainty


def weighted_majority_voting(models, scalers, distances, features, return_score=False):
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

    if return_score:
        return final_decision, avg_certainty
    return final_decision


def bulk_test_with_confidence(file_path, nus, gammas, model_dir=MODEL_DIR, confidence_threshold=0.3):
    df = pd.read_csv(file_path)

    # Configuration variables
    base_increase = 0.06
    base_decrease = 0.12
    high_certainty_threshold = 0.7
    high_certainty_boost_factor = 0.4

    # Calculate boosts based on high-certainty factor
    high_certainty_boost_increase = base_increase * high_certainty_boost_factor
    high_certainty_boost_decrease = base_decrease * high_certainty_boost_factor
    consecutive_genuine_boost = 0.04
    consecutive_impostor_penalty = 0.05

    frr_list = []
    far_list = []
    mean_rejected_genuine_prompts_list = []
    mean_accepted_impostor_prompts_list = []

    for user in df['author'].unique():
        print(f"\nTesting user: {user}")
        test_file = os.path.join(model_dir, f"user_{user}_test.csv")
        if not os.path.exists(test_file):
            print(f"Test data for user {user} not found.")
            continue

        test_texts = pd.read_csv(test_file)['content'].values
        impostor_texts = df[df['author'] != user]['content'].sample(len(test_texts), random_state=42).values

        models, scalers, distances = [], [], []
        for nu in nus:
            for gamma in gammas:
                try:
                    model, scaler, max_distance = load_model_and_scaler_with_distance(user, nu, gamma, model_dir)
                    models.append(model)
                    scalers.append(scaler)
                    distances.append(max_distance)
                except FileNotFoundError:
                    print(f"Model for user {user} with nu={nu} and gamma={gamma} not found.")
                    continue

        # Testing genuine user prompts
        false_rejections = 0
        rejected_count_genuine = 0
        rejected_counts_for_genuine_locks = []

        confidence_level = 0.6
        consecutive_genuine = 0
        consecutive_impostor = 0

        for text in test_texts:
            features = extract_features(text)  # Updated feature extraction
            decision, certainty_score = weighted_majority_voting(models, scalers, distances, features, return_score=True)

            if decision == 1:
                consecutive_genuine += 1
                consecutive_impostor = 0
                confidence_level += base_increase
                confidence_level += high_certainty_boost_increase if certainty_score > high_certainty_threshold else 0
                if consecutive_genuine % 3 == 0:
                    confidence_level += consecutive_genuine_boost
            else:
                consecutive_impostor += 1
                consecutive_genuine = 0
                confidence_level -= base_decrease
                confidence_level -= high_certainty_boost_decrease if certainty_score > high_certainty_threshold else 0
                if consecutive_impostor % 2 == 0:
                    confidence_level -= consecutive_impostor_penalty
                false_rejections += 1
                rejected_count_genuine += 1

            if confidence_level < confidence_threshold:
                rejected_counts_for_genuine_locks.append(rejected_count_genuine)
                confidence_level = 0.6
                rejected_count_genuine = 0

        if rejected_count_genuine > 0:
            rejected_counts_for_genuine_locks.append(rejected_count_genuine)

        frr = (false_rejections / len(test_texts)) * 100
        mean_rejected_genuine_prompts = np.mean(rejected_counts_for_genuine_locks) if rejected_counts_for_genuine_locks else 0
        frr_list.append(frr)
        mean_rejected_genuine_prompts_list.append(mean_rejected_genuine_prompts)

        # Testing impostor prompts
        false_acceptances = 0
        accepted_count = 0
        accepted_counts_for_locks = []

        confidence_level = 0.6
        consecutive_genuine = 0
        consecutive_impostor = 0

        for prompt in impostor_texts:
            features = extract_features(prompt)  # Updated feature extraction
            decision, certainty_score = weighted_majority_voting(models, scalers, distances, features, return_score=True)

            if decision == -1:
                consecutive_impostor += 1
                consecutive_genuine = 0
                confidence_level -= base_decrease
                confidence_level -= high_certainty_boost_decrease if certainty_score > high_certainty_threshold else 0
                if consecutive_impostor % 2 == 0:
                    confidence_level -= consecutive_impostor_penalty
            else:
                consecutive_genuine += 1
                consecutive_impostor = 0
                confidence_level += base_increase
                confidence_level += high_certainty_boost_increase if certainty_score > high_certainty_threshold else 0
                if consecutive_genuine % 3 == 0:
                    confidence_level += consecutive_genuine_boost
                false_acceptances += 1
                accepted_count += 1

            if confidence_level < confidence_threshold:
                accepted_counts_for_locks.append(accepted_count)
                confidence_level = 0.6
                accepted_count = 0

        if accepted_count > 0:
            accepted_counts_for_locks.append(accepted_count)

        far = (false_acceptances / len(impostor_texts)) * 100 if len(impostor_texts) > 0 else 0
        mean_accepted_impostor_prompts = np.mean(accepted_counts_for_locks) if accepted_counts_for_locks else 0
        far_list.append(far)
        mean_accepted_impostor_prompts_list.append(mean_accepted_impostor_prompts)

    mean_frr = np.mean(frr_list) if frr_list else 0
    mean_far = np.mean(far_list) if far_list else 0
    overall_mean_rejected_genuine_prompts = np.mean(mean_rejected_genuine_prompts_list) if mean_rejected_genuine_prompts_list else 0
    overall_mean_accepted_impostor_prompts = np.mean(mean_accepted_impostor_prompts_list) if mean_accepted_impostor_prompts_list else 0

    print("\n--- Overall Metrics ---")
    print(f"Mean FRR: {mean_frr:.2f}%")
    print(f"Mean FAR: {mean_far:.2f}%")
    print(f"Mean Genuine Rejected Prompts Before Lock: {overall_mean_rejected_genuine_prompts:.2f}")
    print(f"Mean Impostor Accepted Prompts Before Lock: {overall_mean_accepted_impostor_prompts:.2f}")
