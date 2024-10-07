import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import OneClassSVM
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
y = df['author'].values

# Step 3: Hyperparameter Combinations for Manual Grid Search
param_grid = {
    'nu': [0.01, 0.05, 0.1, 0.2],
    'gamma': ['scale', 0.001, 0.01, 0.1, 1]
}

# Perform Stratified K-Fold Cross Validation for Each Combination
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results for each combination of hyperparameters
results = []

for nu in param_grid['nu']:
    for gamma in param_grid['gamma']:
        overall_false_rejection_count = 0
        overall_false_acceptance_count = 0
        overall_samples_tested = 0
        overall_samples_train = 0
        
        print(f"Testing combination: nu={nu}, gamma={gamma}")
        
        # Loop over each author to evaluate the hyperparameter combination
        for author in df['author'].unique():
            author_indices = df.index[df['author'] == author].tolist()
            author_data = X[author_indices]
            
            # Ensure there is enough data for cross-validation
            if len(author_data) < 2:
                continue
            
            # Perform Stratified K-Fold Cross Validation
            for train_index, test_index in skf.split(author_data, [author] * len(author_data)):
                X_train, X_test = author_data[train_index], author_data[test_index]

                # Initialize and train One-Class SVM with current hyperparameters
                model = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
                model.fit(X_train)

                # Predicting on train and test data
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                # Evaluating model: +1 means inlier, -1 means outlier
                false_rejection_count = np.sum(y_pred_test == -1)  # False rejection (legit samples marked as outliers)
                false_acceptance_count = np.sum(y_pred_train == -1)  # False acceptance (training samples marked as outliers)

                samples_tested = len(y_pred_test)
                samples_train = len(y_pred_train)

                # Accumulate results across all authors in the fold
                overall_false_rejection_count += false_rejection_count
                overall_false_acceptance_count += false_acceptance_count
                overall_samples_tested += samples_tested
                overall_samples_train += samples_train
        
        # Calculate False Rejection Rate (FRR) and False Acceptance Rate (FAR)
        overall_FRR = overall_false_rejection_count / overall_samples_tested if overall_samples_tested > 0 else 0
        overall_FAR = overall_false_acceptance_count / overall_samples_train if overall_samples_train > 0 else 0

        # Store results for this combination of hyperparameters
        results.append({
            'nu': nu,
            'gamma': gamma,
            'FRR': overall_FRR,
            'FAR': overall_FAR
        })
        print(f"FRR: {overall_FRR:.4f}, FAR: {overall_FAR:.4f}\n")

# Print summary of all results
print("\nSummary of all hyperparameter combinations:")
for result in results:
    print(f"nu={result['nu']}, gamma={result['gamma']}, FRR={result['FRR']:.4f}, FAR={result['FAR']:.4f}")

# ------------------------------
# Results from Grid Search
# Testing combination: nu=0.01, gamma=scale
