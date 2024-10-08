import pandas as pd
import numpy as np
import re
from sklearn.model_selection import KFold
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textstat import flesch_reading_ease
from gensim.models import Word2Vec

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load Dataset
data_path = "twitterDataset20/data/tweets.csv"
df = pd.read_csv(data_path)

# Step 1: Data Preparation and Preprocessing
def preprocess_tweet(tweet):
    # Removing URLs, mentions, and special characters
    tweet = re.sub(r"http\S+|@\S+|[^A-Za-z0-9\s]", "", tweet)
    # Lowercasing
    tweet = tweet.lower()
    return tweet

df['processed_text'] = df['content'].apply(preprocess_tweet)

# Tokenization for feature extraction
df['tokenized_text'] = df['processed_text'].apply(word_tokenize)

# Step 2: Feature Extraction (Stylometric and Semantic Features)
def stylometric_features(tweet, tokens):
    prompt_length_chars = len(tweet)
    
    # Hapax Legomena Ratio
    word_freq = nltk.FreqDist(tokens)
    hapax_legomena_count = len([word for word in word_freq if word_freq[word] == 1])
    hapax_legomena_ratio = hapax_legomena_count / len(tokens) if len(tokens) > 0 else 0

    # Readability Score
    readability_score = flesch_reading_ease(tweet)

    # Punctuation Usage Metric
    num_punctuation = len(re.findall(r'[.,!?;:]', tweet))
    punctuation_ratio = num_punctuation / len(tweet) if len(tweet) > 0 else 0

    # Capitalization Usage Metric
    num_capitalized = len([char for char in tweet if char.isupper()])
    capitalization_ratio = num_capitalized / len(tweet) if len(tweet) > 0 else 0

    # Function Word Ratio
    function_words = set(stopwords.words('english'))
    function_word_count = len([word for word in tokens if word in function_words])
    function_word_ratio = function_word_count / len(tokens) if len(tokens) > 0 else 0

    return [
        prompt_length_chars, hapax_legomena_ratio, readability_score, 
        punctuation_ratio, capitalization_ratio, function_word_ratio
    ]

stylometric_data = df.apply(lambda row: stylometric_features(row['processed_text'], row['tokenized_text']), axis=1)
stylometric_df = pd.DataFrame(stylometric_data.tolist(), columns=[
    'prompt_length_chars', 'hapax_legomena_ratio', 'readability_score', 
    'punctuation_ratio', 'capitalization_ratio', 'function_word_ratio'
])

# TF-IDF Feature Extraction
tfidf_vectorizer = TfidfVectorizer(max_features=300)
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text']).toarray()

# Latent Dirichlet Allocation (LDA) Topic Features
lda_vectorizer = TfidfVectorizer(max_features=100)
X_lda_tfidf = lda_vectorizer.fit_transform(df['processed_text'])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
X_lda = lda.fit_transform(X_lda_tfidf)

# Word2Vec Embedding Features (average word embeddings)
word2vec_model = Word2Vec(sentences=df['tokenized_text'], vector_size=100, window=5, min_count=1, workers=4)
def average_word2vec(tokens, model, vec_size=100):
    valid_words = [model.wv[word] for word in tokens if word in model.wv]
    if len(valid_words) > 0:
        return np.mean(valid_words, axis=0)
    else:
        return np.zeros(vec_size)

word2vec_features = df['tokenized_text'].apply(lambda tokens: average_word2vec(tokens, word2vec_model))
word2vec_features_df = pd.DataFrame(word2vec_features.tolist())

# Combine All Features
X = np.hstack((X_tfidf, X_lda, stylometric_df.values, word2vec_features_df.values))
y = df['author']

# Step 3: Training One-Class SVM using K-Fold Cross Validation for each author
authors = df['author'].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Precision tracking
precision_scores = []

# Initialize counters
overall_false_rejection_count = 0
overall_false_acceptance_count = 0
overall_samples_tested = 0

# Use fixed parameters after grid search
nu = 0.01
gamma = 'scale'

for author in authors:
    author_data = X[df['author'] == author]  # Inliers: the current user's tweets
    other_authors_data = X[df['author'] != author]  # Outliers: tweets from all other users
    
    false_rejection_count = 0
    false_acceptance_count = 0
    samples_tested = 0

    for train_index, test_index in kf.split(author_data):
        # Training on only the current user's tweets (inliers)
        X_train = author_data[train_index]
        # Testing on both the current user's tweets and tweets from other users (outliers)
        X_test_inliers = author_data[test_index]  # Current user's tweets
        X_test_outliers = other_authors_data  # Other users' tweets
        
        X_test = np.vstack((X_test_inliers, X_test_outliers))  # Combine inliers and outliers in the test set
        
        # One-Class SVM Training with optimized parameters
        model = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
        model.fit(X_train)

        # Predicting on test data
        y_pred_test = model.predict(X_test)

        # For Precision Calculation:
        # Create true labels: 1 for inliers (current user's tweets), -1 for outliers (other users' tweets)
        y_true_test = np.hstack((np.ones(len(X_test_inliers)), -1 * np.ones(len(X_test_outliers))))
        precision = precision_score(y_true_test, y_pred_test, pos_label=1, zero_division=0)
        precision_scores.append(precision)

        # Count the number of false rejections and false acceptances
        false_rejection_count += np.sum(y_pred_test[:len(X_test_inliers)] == -1)  # False rejection of inliers
        false_acceptance_count += np.sum(y_pred_test[len(X_test_inliers):] == 1)  # False acceptance of outliers

        samples_tested += len(y_pred_test)

    # Accumulate results across all authors
    overall_false_rejection_count += false_rejection_count
    overall_false_acceptance_count += false_acceptance_count
    overall_samples_tested += samples_tested

# Calculate overall False Rejection Rate (FRR) and False Acceptance Rate (FAR)
overall_FRR = overall_false_rejection_count / overall_samples_tested
overall_FAR = overall_false_acceptance_count / overall_samples_tested

# Calculate average Precision
avg_precision = np.mean(precision_scores)

print(f"Overall False Rejection Rate (FRR): {overall_FRR:.2f}")
print(f"Overall False Acceptance Rate (FAR): {overall_FAR:.2f}")
print(f"Average Precision: {avg_precision:.4f}")
