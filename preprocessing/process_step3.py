import json
import os
import re
import string
from csv import DictWriter
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams


def process_and_tokenize(element):
    """
    Function to clean, process, normalize, and process each element of a larger text corpus.

    Args:
        element: This is the raw text file to clean, process, and normalize

    Returns: List of lists with word tokens
    """
    stripped = re.sub(r"[^a-zA-Z0-9]", " ", element)
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
    filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
    # Return filtered tokens
    return filtered_tokens


def parse_review(input_json):
    """
    Function to parse a text chunk for text content and outcome.

    Args:
        input_json: JSON segment that represents a single observation

    Returns: Raw text content and outcome in two outputs
    """

    text = input_json["free_text"]
    diagnoses = input_json["diagnosis"]
    cleaned_text = process_and_tokenize(text)

    return cleaned_text, diagnoses


def process_data(input_file, output_df_path):
    """
    Function to parse input_json files and produce CSV with text and labels.

    Args:
        input_file: (String) Filepath of text corpus in JSON format
        output_df_path: (String) Filepath of CSV output

    Returns: Saved CSV file to output_df_path
    """
    # Read in CSV data and convert to JSON
    input_data = pd.read_csv(input_file)
    input_data = input_data.sample(frac=1, random_state=12)
    drugs = input_data.to_json(orient="records")
    parsed = json.loads(drugs)

    i = 1
    for drug in parsed:
        try:
            text_cleaned, diagnoses = parse_review(drug)
            # Create dictionary and append to CSV
            output_dict = {"unigrams": text_cleaned, "diagnoses": diagnoses}
            fieldnames = ["unigrams", "diagnoses"]
            with open(output_df_path, "a") as file_obj:
                dictwriterobject = DictWriter(file_obj, fieldnames=fieldnames)
                dictwriterobject.writerow(output_dict)
                file_obj.close()
        except:
            pass

        print(str(round(i / len(parsed) * 100, 2)) + "% complete...")
        i += 1


if __name__ == "__main__":

    # Check if filepath for output exists, remove and refresh if it already exists
    output_performance_path = "../processed_data/cleaned_data_dict.csv"
    try:
        os.remove(output_performance_path)
    except FileNotFoundError:
        print(
            "Output file has not been created, will create new file at:",
            output_performance_path,
        )

    # Add header to csv row
    field_names = ["unigrams", "diagnoses"]

    with open(output_performance_path, "a") as file:
        dictwriter_object = DictWriter(file, fieldnames=field_names)
        dictwriter_object.writeheader()
        file.close()

    process_data(
        input_file="../intermediate_data/joined_table_50_diagnoses.csv",
        output_df_path=output_performance_path,
    )
