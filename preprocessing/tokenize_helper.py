import string
import re
from nltk import word_tokenize
from nltk.corpus import stopwords


def tokenizer(input_text):
    """
    Helper function to clean, process, and tokenize a raw string of text

    Args:
        input_text: (String) Raw text string

    Returns: List of cleaned, tokenized, non-stopword words from original raw string
    """

    stripped = re.sub(r"[^a-zA-Z0-9]", " ", input_text)
    # Remove whitespace and newline breaks
    stripped = stripped.strip().replace("\n", "")
    # Convert to lowercase
    stripped = stripped.lower()
    # Remove punctuation
    stripped = stripped.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers
    stripped = "".join([i for i in stripped if not i.isdigit()])
    # Tokenize
    tokens = word_tokenize(stripped)
    # Remove stopwords
    stop_words = stopwords.words("english")
    X_test = [w for w in tokens if not w.lower() in stop_words]

    return X_test
