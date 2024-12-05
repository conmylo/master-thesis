# src/extract_features.py
import numpy as np
import nltk
from collections import Counter
from nltk.corpus import stopwords
import textstat

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def extract_features(text):
    """
    Extracts various linguistic and stylistic features from the input text
    to uniquely profile each user. Returns a list of feature values.
    """
    features = {}  # Dictionary to store feature values
    words = nltk.word_tokenize(text)  # Tokenize text into words
    char_count = len(text) if len(text) > 0 else 1  # Total character count, avoid division by zero

    # 1. Character Count (normalized)
    features['char_count_norm'] = char_count / 100  # Normalized to a base of 100 characters

    # 2. Character N-grams (3-grams) Ratio
    # Measures the ratio of 3-character sequences relative to total characters
    features['char_ngrams_3_ratio'] = len([text[i:i+3] for i in range(len(text) - 2)]) / char_count

    # 3. Stop Word Frequency Ratio
    # Ratio of stop words (e.g., "the", "is") to total words
    stop_words = set(stopwords.words('english'))
    stop_word_count = sum(1 for word in words if word.lower() in stop_words)
    features['stop_word_freq_ratio'] = stop_word_count / len(words) if words else 0

    # 4. Word Length Average (normalized)
    # Average word length, normalized by a typical maximum word length of 15
    features['word_length_avg_norm'] = np.mean([len(word) for word in words]) / 15 if words else 0

    # 5. Type-Token Ratio (lexical diversity)
    # Ratio of unique words (types) to total words (tokens) in the text
    features['type_token_ratio'] = len(set(words)) / len(words) if words else 0

    # 6. Uppercase Letter and Digit Proportions
    # Proportion of uppercase letters and digits in the text
    features['uppercase_proportion'] = sum(1 for c in text if c.isupper()) / char_count
    features['digit_proportion'] = sum(1 for c in text if c.isdigit()) / char_count

    # 7. Punctuation Proportion
    # Proportion of punctuation marks (.,!?;:) in the text
    punctuation_count = sum(1 for c in text if c in '.,!?;:')
    features['punctuation_proportion'] = punctuation_count / char_count

    # 8. Part-of-Speech Ratios (Adjectives, Nouns, Verbs, Adverbs)
    # Ratios of various parts of speech, useful for capturing stylistic tendencies
    pos_tags = nltk.pos_tag(words)
    pos_counts = Counter(tag for word, tag in pos_tags)
    features['adjective_ratio'] = pos_counts.get('JJ', 0) / len(words) if words else 0
    features['noun_ratio'] = pos_counts.get('NN', 0) / len(words) if words else 0
    features['verb_ratio'] = sum(pos_counts.get(tag, 0) for tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']) / len(words) if words else 0
    features['adverb_ratio'] = pos_counts.get('RB', 0) / len(words) if words else 0

    # 9. Average Sentence Length
    # Average number of words per sentence
    sentences = nltk.sent_tokenize(text)
    features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0

    # 10. Pronoun Usage Proportion
    # Proportion of pronouns (e.g., "I", "you", "he") to total words
    pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'us', 'him', 'her', 'them'}
    pronoun_count = sum(1 for word in words if word.lower() in pronouns)
    features['pronoun_usage_proportion'] = pronoun_count / len(words) if words else 0

    # 11. Function Word Ratio
    # Ratio of common function words (e.g., "the", "is", "in") to total words
    function_words = {'the', 'is', 'in', 'it', 'of', 'and', 'to', 'a', 'that', 'with', 'as', 'for', 'on', 'at', 'by'}
    function_word_count = sum(1 for word in words if word.lower() in function_words)
    features['function_word_ratio'] = function_word_count / len(words) if words else 0

    # 12. Past Tense Ratio
    # Proportion of past-tense verbs to total words
    past_tense_verbs = pos_counts.get('VBD', 0)
    features['past_tense_ratio'] = past_tense_verbs / len(words) if words else 0

    # 13. Readability Score (scaled Flesch Reading Ease)
    # Measure of readability, scaled to 0-1 range
    readability_score = textstat.flesch_reading_ease(text)
    features['readability_score'] = readability_score / 100

    # 14. Cohesion and Complexity Indicators
    # Average syllable count per word and proportion of polysyllabic words
    features['syllable_avg'] = textstat.syllable_count(text) / len(words) if words else 0
    features['polysyllabic_word_ratio'] = textstat.polysyllabcount(text) / len(words) if words else 0

    # 15. Formality Measure
    # Ratio of nouns/adjectives to pronouns/verbs, indicating formality of language
    features['formality'] = (features['noun_ratio'] + features['adjective_ratio']) / (features['pronoun_usage_proportion'] + features['verb_ratio'] + 0.01)

    # Return feature values as a list
    return list(features.values())
