import pickle
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import nltk
nltk.download('punkt')

# Load and preprocess dataset
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    return df[['author', 'content', 'date_time', 'id']]

# Extract features (same as in training)
def extract_features(text):
    features = {}
    words = nltk.word_tokenize(text)
    features['avg_word_length'] = np.mean([len(word) for word in words])
    char_count = len(text)
    whitespace_count = text.count(' ')
    features['whitespace_ratio'] = whitespace_count / char_count if char_count > 0 else 0
    features['upper_to_lower_ratio'] = sum(1 for c in text if c.isupper()) / (sum(1 for c in text if c.islower()) + 1)
    unique_words = set(words)
    features['vocabulary_richness'] = len(unique_words) / len(words) if words else 0
    features['function_word_count'] = sum(1 for word in words if word.lower() in ['the', 'and', 'in', 'of']) / len(words)
    bigrams = nltk.bigrams(words)
    features['bigrams'] = len(list(bigrams))
    pos_tags = nltk.pos_tag(words)
    pos_counts = nltk.FreqDist(tag for (word, tag) in pos_tags)
    features['noun_ratio'] = pos_counts['NN'] / len(words) if words else 0
    punctuation_count = sum(1 for c in text if c in '.,!?')
    features['punctuation_ratio'] = punctuation_count / len(words) if words else 0
    sentences = nltk.sent_tokenize(text)
    features['sentence_length'] = np.mean([len(nltk.word_tokenize(sentence)) for sentence in sentences]) if sentences else 0
    contraction_count = sum(1 for word in words if "'" in word)
    features['contraction_usage'] = contraction_count / len(words) if words else 0
    return list(features.values())

# Load model
def load_model(user_id, model_dir="models/"):
    with open(f"{model_dir}/user_{user_id}_model.pkl", 'rb') as f:
        return pickle.load(f)

# Evaluate model on test data
def evaluate_model(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# Calculate metrics including FAR and FRR
def calculate_metrics(y_true, y_pred):
    # Convert -1 labels (outliers) to 0, as sklearn expects binary classes to be 0 and 1
    y_true = np.where(y_true == -1, 0, y_true)
    y_pred = np.where(y_pred == -1, 0, y_pred)

    precision = precision_score(y_true, y_pred, average='binary', zero_division=1)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=1)
    
    # Calculate FAR and FRR
    false_rejections = sum((y_pred == 0) & (y_true == 1))  # False negatives
    false_acceptances = sum((y_pred == 1) & (y_true == 0))  # False positives
    total_inliers = sum(y_true == 1)
    total_outliers = sum(y_true == 0)
    
    frr = false_rejections / total_inliers if total_inliers > 0 else 0  # False Rejection Rate (FRR)
    far = false_acceptances / total_outliers if total_outliers > 0 else 0  # False Acceptance Rate (FAR)
    
    return precision, recall, f1, far, frr

# Main test function
def main_test(file_path):
    df = preprocess_data(file_path)
    overall_far, overall_frr = 0, 0
    total_users = len(df['author'].unique())
    
    results = {}
    for user, group in df.groupby('author'):
        model = load_model(user)

        # **Extract features for inliers (user's own data)**
        X_test_inliers = [extract_features(text) for text in group['content']]

        # **Collect outliers (data from other users)**: 50-50 split between inliers and outliers
        outlier_data = df[df['author'] != user]  # Data from all other users
        X_test_outliers = [extract_features(text) for text in outlier_data['content'].sample(n=len(X_test_inliers))]  # Match the number of inliers

        # True labels: 1 for inliers, 0 for outliers
        y_true_inliers = np.ones(len(X_test_inliers))  # **1 for legitimate data (inliers)**
        y_true_outliers = -1 * np.ones(len(X_test_outliers))  # **-1 for outliers (other users)**

        # **Predictions for inliers and outliers**
        y_pred_inliers = evaluate_model(model, X_test_inliers)  # Predict for inliers
        y_pred_outliers = evaluate_model(model, X_test_outliers)  # Predict for outliers

        # **Combine inliers and outliers for evaluation**
        y_true = np.concatenate([y_true_inliers, y_true_outliers])
        y_pred = np.concatenate([y_pred_inliers, y_pred_outliers])

        # Calculate precision, recall, F1, FAR, FRR
        precision, recall, f1, far, frr = calculate_metrics(y_true, y_pred)
        results[user] = (precision, recall, f1, far, frr)

        # **Update overall FAR and FRR**
        overall_far += far
        overall_frr += frr

        print(f"User {user}: Precision: {precision}, Recall: {recall}, F1: {f1}, FAR: {far}, FRR: {frr}")
    
    # **Print overall FAR and FRR**
    print(f"Overall FAR: {overall_far / total_users}, Overall FRR: {overall_frr / total_users}")

# Run testing
if __name__ == '__main__':
    data_path = 'data/cleaned.csv'  # Replace with your dataset path
    main_test(data_path)

# ----------------------------
# # Results
# User ArianaGrande: Precision: 0.49104396598516376, Recall: 0.8800259403372244, F1: 0.6303565207292997, FAR: 0.9121271076523995, FRR: 0.11997405966277562
# User BarackObama: Precision: 0.6040787623066104, Recall: 0.9001047851903597, F1: 0.7229625473418432, FAR: 0.5899406217254628, FRR: 0.09989521480964024
# User Cristiano: Precision: 0.5146788990825688, Recall: 0.8972411035585766, F1: 0.6541320507214692, FAR: 0.846061575369852, FRR: 0.10275889644142343
# User KimKardashian: Precision: 0.4992467043314501, Recall: 0.905087060430181, F1: 0.6435246995994659, FAR: 0.907818368043701, FRR: 0.09491293956981905
# User TheEllenShow: Precision: 0.5265593188969091, Recall: 0.9040355894502701, F1: 0.6654970760233918, FAR: 0.8128376231331427, FRR: 0.0959644105497299
# User Twitter: Precision: 0.534043659043659, Recall: 0.8977719528178244, F1: 0.669708326543914, FAR: 0.7833114897335081, FRR: 0.10222804718217562
# User YouTube: Precision: 0.5751144402829796, Recall: 0.8982775430614235, F1: 0.7012558670556894, FAR: 0.6636334091647709, FRR: 0.10172245693857654
# User britneyspears: Precision: 0.5277424819991529, Recall: 0.898989898989899, F1: 0.6650653856418468, FAR: 0.8044733044733045, FRR: 0.10101010101010101
# User cnnbrk: Precision: 0.7125697125697126, Recall: 0.9017372421281216, F1: 0.7960699736400672, FAR: 0.36373507057546145, FRR: 0.0982627578718784
# User ddlovato: Precision: 0.5054721303130567, Recall: 0.8986425339366516, F1: 0.647010913829614, FAR: 0.8791855203619909, FRR: 0.10135746606334842
# User instagram: Precision: 0.5707729468599034, Recall: 0.9169577027551417, F1: 0.7035879112699122, FAR: 0.6895615056266977, FRR: 0.08304229724485836
# User jimmyfallon: Precision: 0.5402010050251256, Recall: 0.8955462992630567, F1: 0.6738999397227246, FAR: 0.762255687279718, FRR: 0.10445370073694328
# User jtimberlake: Precision: 0.5065053640721296, Recall: 0.8965656565656566, F1: 0.647316219369895, FAR: 0.8735353535353535, FRR: 0.10343434343434343
# User justinbieber: Precision: 0.5466584539574992, Recall: 0.9033078880407125, F1: 0.6811204911742134, FAR: 0.7491094147582698, FRR: 0.09669211195928754
# User katyperry: Precision: 0.4889384643985871, Recall: 0.9025394646533974, F1: 0.6342698661521765, FAR: 0.9433768016472203, FRR: 0.09746053534660261
# User ladygaga: Precision: 0.508582638548308, Recall: 0.9017391304347826, F1: 0.6503606146127313, FAR: 0.871304347826087, FRR: 0.09826086956521739
# User rihanna: Precision: 0.5054552668121405, Recall: 0.8918445922296114, F1: 0.645226639655609, FAR: 0.8725936296814841, FRR: 0.10815540777038853
# User selenagomez: Precision: 0.502219648716464, Recall: 0.9037860368183397, F1: 0.6456575682382134, FAR: 0.8957971517888156, FRR: 0.0962139631816603
# User shakira: Precision: 0.5420738974970203, Recall: 0.9005940594059406, F1: 0.6767857142857143, FAR: 0.7607920792079208, FRR: 0.09940594059405941
# User taylorswift13: Precision: 0.5023513139695712, Recall: 0.899009900990099, F1: 0.6445430346051464, FAR: 0.8905940594059406, FRR: 0.100990099009901
# Overall FAR: 0.7936022060495549, Overall FRR: 0.10030978094713652