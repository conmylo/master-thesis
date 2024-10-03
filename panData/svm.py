import os
import pandas as pd

# Define path to dataset
base_path = 'panData\data\pan12-authorship-attribution-test-dataset-problem-j-2015-10-20'  # Update this to the correct path

# Initialize lists to store data
documents = []
authors = []

# Iterate over each author folder
for author_folder in os.listdir(base_path):
    author_path = os.path.join(base_path, author_folder)
    
    if os.path.isdir(author_path):
        # Iterate over each document for the author
        for document_name in os.listdir(author_path):
            document_path = os.path.join(author_path, document_name)
            
            # Ensure that only files are read (skip subdirectories, if any)
            if os.path.isfile(document_path):
                # Read the document content
                with open(document_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    documents.append(content)
                    authors.append(author_folder)  # Author folder name as author ID

# Create a DataFrame with the collected data
data = pd.DataFrame({'author': authors, 'text': documents})

# Example: Extract features from each document
FEATURE_NAMES = ['avg_word_length', 'function_word_ratio', 'punctuation_usage', 'pronoun_usage', 'lexical_diversity', 'prompt_length']

def extract_stylometric_features(text):
    words = text.split()
    unique_words = set(words)
    
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    lexical_diversity = len(unique_words) / len(words) if words else 0
    function_words = {'the', 'and', 'is', 'in', 'on', 'at', 'by', 'with', 'a', 'an'}
    function_word_ratio = sum(1 for word in words if word in function_words) / len(words) if words else 0
    punctuation_count = sum(1 for char in text if char in '.,!?')
    punctuation_usage = punctuation_count / len(text) if text else 0
    pronoun_usage = sum(1 for word in words if word.lower() in ['i', 'he', 'she', 'we', 'they']) / len(words) if words else 0
    prompt_length = len(text)

    return [avg_word_length, function_word_ratio, punctuation_usage, pronoun_usage, lexical_diversity, prompt_length]

# Apply feature extraction to each document
features = data['text'].apply(extract_stylometric_features)
features_df = pd.DataFrame(features.tolist(), columns=FEATURE_NAMES)
features_df['author'] = data['author']

# View the resulting DataFrame
print(features_df.head())
