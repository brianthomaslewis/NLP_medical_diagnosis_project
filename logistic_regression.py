import os
import numpy as np
import pandas as pd
import re
import string
import pickle
from csv import DictWriter
import csv
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, chi2


def logistic_reg(
    cleaned_filepath,
    tdidf_min=2,
    norm_type="l2",
    ngram_base=1,
    ngram_max=1,
    n_feat=100,
    output_filepath="output/performance_logistic.csv",
):
    """
    Function to build, train, and evaluate a logistic regression model on a cleaned text corpus.

    Args:
        cleaned_filepath: Filepath of cleaned .csv text corpus
        tdidf_min: Max threshold of tdidf value to exclude words
        norm_type: Linear regression regularization norm type, e.g. L1 or L2
        ngram_base: Base number of grams to use in regression, e.g. 1 or 2
        ngram_max: Max number of grams to use in regression, e.g. 1 or 2
        n_feat: Number of features (k) to select while tuning hyperparameters
        output_filepath: Filepath for performance metric output

    Returns: Printed .txt output of results from model run to output_filepath
    """
    data_df = pd.read_csv(cleaned_filepath)

    X, X_test, y, y_test = train_test_split(
        data_df["unigrams"], data_df["diagnoses"], test_size=0.2, random_state=6
    )

    # Oversampling
    ros = RandomOverSampler(sampling_strategy="minority")
    X_train, y_train = ros.fit_resample(X.values.reshape(-1, 1), y)

    # # Undersampling
    # rus = RandomUnderSampler(sampling_strategy='majority')
    # X_train, y_train = rus.fit_resample(X.values.reshape(-1, 1), y)

    X_train = X_train.flatten()

    model = Pipeline(
        [
            (
                "vect",
                TfidfVectorizer(
                    min_df=tdidf_min,
                    stop_words="english",
                    sublinear_tf=True,
                    norm=norm_type,
                    ngram_range=(ngram_base, ngram_max),
                ),
            ),
            ("kbest", SelectKBest(score_func=chi2, k=n_feat)),
            ("classif", LogisticRegression(random_state=6)),
        ]
    ).fit(X_train, y_train)

    ytest = np.array(y_test)

    # Built-in and calculated performance metrics
    conf_matrix = confusion_matrix(ytest, model.predict(X_test))

    # Calculate accuracy metrics for dictionary
    precision = np.nanmean(np.diag(conf_matrix) / np.sum(conf_matrix, axis=0))
    recall = np.nanmean(np.diag(conf_matrix) / np.sum(conf_matrix, axis=1))
    f1_score = np.nanmean(2 * np.multiply(precision, recall) / (precision + recall))
    micro_f1_score = np.sum(np.diag(conf_matrix)) / np.sum(np.sum(conf_matrix, axis=0))

    # Print out existing metrics as well
    with open("output/logistic_confusion_matrices.csv", "a") as file:
        writer_obj = csv.writer(file)
        writer_obj.writerows(classification_report(ytest, model.predict(X_test)))
        file.close()

    # Create dictionary and append to CSV
    output_dict = {
        "max_td_idf": tdidf_min,
        "norm_type": norm_type,
        "ngram_base": ngram_base,
        "ngram_max": ngram_max,
        "num_of_features": n_feat,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "micro_f1_score": micro_f1_score,
    }
    field_names = [
        "max_td_idf",
        "norm_type",
        "ngram_base",
        "ngram_max",
        "num_of_features",
        "precision",
        "recall",
        "f1_score",
        "micro_f1_score",
    ]
    with open(output_filepath, "a") as file:
        dictwriter_object = DictWriter(file, fieldnames=field_names)
        dictwriter_object.writerow(output_dict)
        file.close()

    return model


if __name__ == "__main__":

    # Check if filepath for output exists, remove and refresh if it already exists
    output_performance_path = "output/performance_logistic.csv"
    try:
        os.remove(output_performance_path)
    except FileNotFoundError:
        print(
            "performance_logistic.csv file has not been created, will create new file at: ",
            output_performance_path,
        )

    # Add header to csv row
    field_names = [
        "max_td_idf",
        "norm_type",
        "ngram_base",
        "ngram_max",
        "num_of_features",
        "precision",
        "recall",
        "f1_score",
        "micro_f1_score",
    ]

    with open(output_performance_path, "a") as file:
        dictwriter_object = DictWriter(file, fieldnames=field_names)
        dictwriter_object.writeheader()
        file.close()

    # Build out models and assign them to objects

    m1 = logistic_reg(
        "processed_data/cleaned_data_dict.csv",
        tdidf_min=2,
        norm_type="l1",
        ngram_base=1,
        ngram_max=1,
        n_feat="all",
    )
    m2 = logistic_reg(
        "processed_data/cleaned_data_dict.csv",
        tdidf_min=2,
        norm_type="l1",
        ngram_base=1,
        ngram_max=2,
        n_feat="all",
    )
    m3 = logistic_reg(
        "processed_data/cleaned_data_dict.csv",
        tdidf_min=2,
        norm_type="l1",
        ngram_base=2,
        ngram_max=2,
        n_feat="all",
    )

    m4 = logistic_reg(
        "processed_data/cleaned_data_dict.csv",
        tdidf_min=2,
        norm_type="l2",
        ngram_base=1,
        ngram_max=1,
        n_feat="all",
    )
    m5 = logistic_reg(
        "processed_data/cleaned_data_dict.csv",
        tdidf_min=2,
        norm_type="l2",
        ngram_base=1,
        ngram_max=2,
        n_feat="all",
    )
    m6 = logistic_reg(
        "processed_data/cleaned_data_dict.csv",
        tdidf_min=2,
        norm_type="l2",
        ngram_base=2,
        ngram_max=2,
        n_feat="all",
    )

    # Write model artifacts to appropriate filepaths
    model_list = [m1, m2, m3, m4, m5, m6]
    i = 0
    for model in model_list:
        i += 1
        mod_string = "m" + str(i)
        path = f"model_artifacts/logistic_{mod_string}.pickle"
        with open(path, "wb") as f:
            pickle.dump(model, f)
