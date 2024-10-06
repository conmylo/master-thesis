import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

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
    # Tokenize
    tokens = word_tokenize(tweet)
    # Removing stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Preprocessing the content column
df['processed_text'] = df['content'].apply(preprocess_tweet)

# Step 2: Feature Extraction (Stylometric Features)
# Stylometric features can include TF-IDF scores, average word length, and punctuation counts.

# Extracting TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=300)  # Limiting to top 300 features for simplicity
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text']).toarray()

# Adding some additional stylometric features
def stylometric_features(tweet):
    words = word_tokenize(tweet)
    num_words = len(words)
    avg_word_len = np.mean([len(word) for word in words]) if num_words > 0 else 0
    num_punctuation = len(re.findall(r'[.,!?;]', tweet))
    return [num_words, avg_word_len, num_punctuation]

stylometric_data = df['processed_text'].apply(stylometric_features)
stylometric_df = pd.DataFrame(stylometric_data.tolist(), columns=['num_words', 'avg_word_len', 'num_punctuation'])

# Combine TF-IDF and other stylometric features
X = np.hstack((X_tfidf, stylometric_df.values))

# Step 3: Training One-Class SVM for each user
authors = df['author'].unique()
models = {}

for author in authors:
    author_data = X[df['author'] == author]
    
    # Train-Test Split for each author to evaluate model performance
    X_train, X_test = train_test_split(author_data, test_size=0.3, random_state=42)

    # One-Class SVM Training
    model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    model.fit(X_train)
    models[author] = model

    # Predicting on test data
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluating model: +1 means inlier, -1 means outlier
    false_rejection = np.sum(y_pred_test == -1)  # False rejection (legit samples marked as outliers)
    false_acceptance = np.sum(y_pred_train == -1)  # False acceptance (training samples marked as outliers)

    FRR = false_rejection / len(y_pred_test)
    FAR = false_acceptance / len(y_pred_train)
    
    print(f"Author: {author}")
    print(f"False Rejection Rate (FRR): {FRR:.2f}")
    print(f"False Acceptance Rate (FAR): {FAR:.2f}")
    print("-" * 30)
