import pandas as pd
import re

def remove_urls(text):
    """Remove URLs from content."""
    return re.sub(r'http\S+', '', text)

def preprocess_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    df['content'] = df['content'].apply(remove_urls)
    return df
