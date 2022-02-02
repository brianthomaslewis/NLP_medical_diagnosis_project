from googleapiclient.discovery import build
import pprint
import pandas as pd
import json
import re
import time

API_KEY = open("../raw_data/search_api_key.txt").read()
CSE_ID = open("../raw_data/search_cse_id.txt").read()


def google_search(search_term, api_key=API_KEY, cse_id=CSE_ID, **kwargs):
    """
    Basic function to utilize Google's API client for its "Custom Search" product.

    Args:
        search_term: (String) Search term or phrase that will act as text input into Google's "Custom Search" API.
        api_key: (String) API key associated with Google Cloud account.
        cse_id: (String) CSE ID associated with Google Cloud account.

    Returns: JSON dictionary of search results from "Custom Search" API.
    """
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    try:
        res_dict = res["items"]
    except:
        res_dict = {}
    return res_dict


def api_loop(phrase_list, json_stem_path):
    """
    Function to iteratively loop through a list of search phrases using `google_search` and save them as
    raw JSON output.

    Args:
        phrase_list: (List) List of distinct search terms through which to iterate
        json_stem_path: (String) Stem of filepath wherein to store the raw JSON output data

    Returns: Saved raw JSON output data to the file paths with a prefix from `json_stem_path`.
    """
    iterator = 0
    for keyword in phrase_list:
        search_term = keyword
        search_phrase = f"what is {search_term} used for?"
        json_output = {}

        results = google_search(search_phrase, API_KEY, CSE_ID, num=10)

        json_iterator = 0
        for result in results:
            try:
                json_output[json_iterator] = result["snippet"]

            except KeyError as e:
                json_output[json_iterator] = ""

            json_iterator += 1

        with open(f"{json_stem_path}{i}.json", "w") as fp:
            json.dump(json_output, fp)
        iterator += 1
        time.sleep(1)


def pre_process_step_2a(mimic_path, json_stem_path):
    """
    Function to extract distinct list of drug names and iteratively query+save data to specified filepaths.

    Args:
        mimic_path: (String) Filepath of processed diagnosis+prescriptions CSV data
        json_stem_path: (String) Filepath prefix for where to save raw JSON output files from API

    Returns: Raw JSON data output saved within the file paths prefixed by `json_stem_path`
    """

    # Load in output data from prior step, narrow to existing drug list
    data = pd.read_csv(mimic_path)

    # Shorten the list as needed
    drug_list = data.drug_clean.str.strip().unique().tolist()
    drug_list.sort()

    # Run the queries
    api_loop(drug_list, json_stem_path)


def pre_process_step_2b(json_dump_path, output_path):
    """
    Function to parse raw JSON output, clean and process parsed data, and merge with existing joined data.

    Args:
        json_dump_path: (String) Filepath prefix of raw JSON data output files
        output_path: (String) Filepath for where to save merged data containing parsed JSON and existing data

    Returns: A saved CSV output file containing 'id', 'drug_clean', 'free_text', 'diagnosis' fields
    """

    # Pull out relevant data from JSON results
    result_list = []

    for i in range(0, len(drug_list)):

        # Open each JSON associated with the test_list
        with open(f"{json_dump_path}{i}.json", "r") as fp:
            element_list = []
            json_list = json.load(fp)

        # Pull out relevant content of the API call results
        for key in json_list:
            element_list.append(json_list[key])

        # Join together all the relevant contents and clean them to create mini-corpus
        output_joined = " ".join(element_list)
        r2 = re.sub("[\n,]+", "", output_joined)
        r3 = re.sub("[\n,—\xa0.:;0-9()?]+", "", r2)
        r4 = re.sub(" mg+| mcg+", "", r3)
        r5 = re.sub(" mL+| /mL+", "", r4)
        r6 = re.sub(" +|-", " ", r5)
        r7 = re.sub("[^A-Za-z0-9 ]+", "", r6)
        r8 = re.sub(" +", " ", r7)
        r9 = r8.strip().lower()

        # Join mini-corpora for the full test_list
        result_list.append(r9)

    zip_iter = zip(drug_list, result_list)
    result_dc = dict(zip_iter)

    df = pd.DataFrame.from_dict(result_dc, orient="index").reset_index()
    df.set_axis(["drug_clean", "free_text"], axis=1, inplace=True)
    data2 = data.merge(df, how="inner", on="drug_clean")
    data2.sort_values("drug_clean")
    output = data2[["id", "drug_clean", "free_text", "diagnosis"]].copy()
    output.to_csv(output_path, index=False)


if __name__ == "__main__":
    pre_process_step_2a(
        mimic_path="../intermediate_data/mimic_cleaned_50_diagnoses.csv",
        json_stem_path="../raw_data/api_results/json_",
    )
    pre_process_step_2b(
        json_dump_path="..raw_data/api_results/json_",
        output_path="../intermedia_data/joined_table_50_diagnoses.csv",
    )
