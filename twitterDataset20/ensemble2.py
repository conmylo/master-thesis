import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textstat import flesch_reading_ease
from gensim.models import Word2Vec
from keras import Model, Input, Dense # comment this line before running
# from tensorflow.keras.models import Model # uncomment this line before running
# from tensorflow.keras.layers import Input, Dense # uncomment this line before running

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

# Stratified K-Fold Cross Validation
y = df['author']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

overall_false_rejection_count = 0
overall_false_acceptance_count = 0
overall_samples_tested = 0
overall_samples_train = 0

# Build Autoencoder Model
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

# Perform Stratified K-Fold Cross Validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    for author in y_train.unique():
        # Get training and testing data for the current author
        author_train_data = X_train[y_train == author]
        author_test_data = X_test[y_test == author]

        # Train One-Class SVM
        ocs_model = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.05)
        ocs_model.fit(author_train_data)

        # Train Isolation Forest
        iso_forest_model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        iso_forest_model.fit(author_train_data)

        # Train Autoencoder
        autoencoder = build_autoencoder(input_dim=author_train_data.shape[1])
        autoencoder.fit(author_train_data, author_train_data, epochs=50, batch_size=16, shuffle=True, validation_split=0.1, verbose=0)

        # Make Predictions and Aggregate Results
        y_pred_test_ocsvm = ocs_model.predict(author_test_data)
        y_pred_test_iso_forest = iso_forest_model.predict(author_test_data)

        reconstructed_test = autoencoder.predict(author_test_data)
        test_mse = np.mean(np.power(author_test_data - reconstructed_test, 2), axis=1)
        threshold = np.percentile(test_mse, 95)
        y_pred_test_autoencoder = np.array([1 if error <= threshold else -1 for error in test_mse])

        # Aggregate Predictions (Voting)
        votes = np.vstack((y_pred_test_ocsvm, y_pred_test_iso_forest, y_pred_test_autoencoder)).T
        y_pred_ensemble = np.apply_along_axis(lambda x: 1 if np.sum(x == 1) > 1 else -1, axis=1, arr=votes)

        # Calculate False Rejections and False Acceptances
        false_rejection_count = np.sum(y_pred_ensemble == -1)  # Legit samples incorrectly marked as outliers
        false_acceptance_count = np.sum(y_pred_ensemble == 1)  # Training samples incorrectly marked as inliers

        samples_tested = len(y_pred_ensemble)
        samples_train = len(author_train_data)

        # Accumulate results across all authors in the fold
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
# Results got us: 
# Overall False Rejection Rate (FRR): 0.03
# Overall False Acceptance Rate (FAR): 0.24