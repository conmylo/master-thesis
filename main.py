from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk
import enchant

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize spell checker
spell_checker = enchant.Dict("en_US")


def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())  # Convert text to lowercase for consistency

    # Perform part-of-speech tagging
    pos_tags = pos_tag(tokens)

    return tokens, pos_tags


def calculate_spelling_accuracy(tokens):
    # Calculate the spelling accuracy percentage
    total_words = len(tokens)
    correctly_spelled_words = sum(1 for word in tokens if spell_checker.check(word))
    spelling_accuracy = correctly_spelled_words / total_words if total_words > 0 else 0

    return spelling_accuracy


def calculate_prompt_length(tokens):
    # Calculate the prompt length
    return len(tokens)


def calculate_pos_percentage(pos_tags, target_pos):
    # Calculate the percentage of specific parts of speech (e.g., nouns, verbs, adjectives)
    total_words = len(pos_tags)
    target_words = sum(1 for _, pos in pos_tags if pos.startswith(target_pos))
    pos_percentage = target_words / total_words if total_words > 0 else 0

    return pos_percentage


# Example usage
text = "This is a sample sentence."
tokens, pos_tags = preprocess_text(text)
spelling_accuracy = calculate_spelling_accuracy(tokens)
prompt_length = calculate_prompt_length(tokens)
noun_percentage = calculate_pos_percentage(pos_tags, 'NN')

print("Tokens:", tokens)
print("Spelling Accuracy:", spelling_accuracy)
print("Prompt Length:", prompt_length)
print("Noun Percentage:", noun_percentage)
