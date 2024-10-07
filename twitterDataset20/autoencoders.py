import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textstat import flesch_reading_ease
from gensim.models import Word2Vec
import nltk

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

# Step 3: Autoencoder Training and Evaluation with K-Fold Cross Validation
authors = df['author'].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

overall_false_rejection_count = 0
overall_false_acceptance_count = 0
overall_samples_tested = 0
overall_samples_train = 0

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

for author in authors:
    author_data = X[df['author'] == author]
    
    false_rejection_count = 0
    false_acceptance_count = 0
    samples_tested = 0
    samples_train = 0

    for train_index, test_index in kf.split(author_data):
        X_train, X_test = author_data[train_index], author_data[test_index]

        # Build and train the autoencoder
        autoencoder = build_autoencoder(input_dim=X_train.shape[1])
        autoencoder.fit(X_train, X_train, epochs=50, batch_size=16, shuffle=True, validation_split=0.1, verbose=0)

        # Predict reconstruction on train and test data
        reconstructed_train = autoencoder.predict(X_train)
        reconstructed_test = autoencoder.predict(X_test)

        # Calculate reconstruction error (Mean Squared Error)
        train_mse = np.mean(np.power(X_train - reconstructed_train, 2), axis=1)
        test_mse = np.mean(np.power(X_test - reconstructed_test, 2), axis=1)

        # Set threshold for inliers based on training reconstruction error
        threshold = np.percentile(train_mse, 95)  # Using 95th percentile of train errors as threshold

        # Calculate false rejections and false acceptances
        false_rejection_count += np.sum(test_mse > threshold)  # Legit samples incorrectly marked as outliers
        false_acceptance_count += np.sum(train_mse > threshold)  # Training samples incorrectly marked as outliers

        samples_tested += len(test_mse)
        samples_train += len(train_mse)
    
    # Accumulate results across all authors
    overall_false_rejection_count += false_rejection_count
    overall_false_acceptance_count += false_acceptance_count
    overall_samples_tested += samples_tested
    overall_samples_train += samples_train

# Calculate overall False Rejection Rate (FRR) and False Acceptance Rate (FAR)
overall_FRR = overall_false_rejection_count / overall_samples_tested
overall_FAR = overall_false_acceptance_count / overall_samples_train

print(f"Overall False Rejection Rate (FRR): {overall_FRR:.2f}")
print(f"Overall False Acceptance Rate (FAR): {overall_FAR:.2f}")

# ------------------------------
# Explanation of Autoencoder Approach:
# Feature Extraction:

# The same feature engineering steps as the previous model are used to create a feature matrix (X), including TF-IDF, LDA, stylometric features, and Word2Vec embeddings.
# Autoencoder Architecture:

# An autoencoder is created using Keras (from TensorFlow).
# The encoder consists of three dense layers (128, 64, and 32 units).
# The decoder mirrors the encoder to reconstruct the input.
# The mean squared error (MSE) is used as the loss function to minimize reconstruction errors.
# Training the Autoencoder:

# The autoencoder is trained on the author’s data to learn to reconstruct the features.
# The threshold for classifying a sample as an outlier is set based on the 95th percentile of the reconstruction errors from the training set.
# Evaluation:

# During testing, reconstruction errors are computed for both train and test samples.
# False Acceptance Rate (FAR) and False Rejection Rate (FRR) are calculated based on how well the autoencoder can reconstruct known inliers.
# Improvements and Expected Outcomes:
# The autoencoder learns to compress and reconstruct the data, making it particularly sensitive to features that are specific to an author's style.
# By setting a threshold on the reconstruction error, we can identify whether a given sample matches the learned writing style.
# This approach, along with the enriched feature set, aims to reduce both FAR and FRR below 10% by learning a more nuanced representation of the author’s style.