import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def extract_features(text):
    """Extract stylometric features from a given text."""
    features = {}
    words = nltk.word_tokenize(text)
    
    # 1. Average word length
    features['avg_word_length'] = np.mean([len(word) for word in words])

    # 2. Character distribution
    char_count = len(text)
    whitespace_count = text.count(' ')
    features['whitespace_ratio'] = whitespace_count / char_count if char_count > 0 else 0

    # 3. Uppercase to lowercase ratio
    features['upper_to_lower_ratio'] = sum(1 for c in text if c.isupper()) / (sum(1 for c in text if c.islower()) + 1)

    # 4. Vocabulary richness
    unique_words = set(words)
    features['vocabulary_richness'] = len(unique_words) / len(words) if words else 0

    # 5. Function word frequency (using stop words as proxies for function words)
    vectorizer = CountVectorizer(stop_words='english')
    function_word_count = vectorizer.fit_transform([text]).sum()
    features['function_word_count'] = function_word_count / len(words) if words else 0

    # 6. N-grams (bigrams) frequency
    bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
    bigrams = bigram_vectorizer.fit_transform([text]).sum()
    features['bigrams'] = bigrams

    # 7. POS tagging distribution
    pos_tags = nltk.pos_tag(words)
    pos_counts = nltk.FreqDist(tag for (word, tag) in pos_tags)
    features['noun_ratio'] = pos_counts['NN'] / len(words) if words else 0

    # 8. Punctuation style (counting punctuations)
    punctuation_count = sum(1 for c in text if c in '.,!?')
    features['punctuation_ratio'] = punctuation_count / len(words) if words else 0

    # 9. Sentence length (in words)
    sentences = nltk.sent_tokenize(text)
    features['sentence_length'] = np.mean([len(nltk.word_tokenize(sentence)) for sentence in sentences]) if sentences else 0

    # 10. Contraction usage
    contraction_count = sum(1 for word in words if "'" in word)
    features['contraction_usage'] = contraction_count / len(words) if words else 0

    return features
