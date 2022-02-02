import numpy as np
import pandas as pd
import seaborn as sns
import re
from matplotlib import pyplot as plt
import requests
import urllib
import time
from requests_html import HTML
from requests_html import HTMLSession


# ### Step 1: Merge `diagnoses_icd` with `d_icd_diagnoses` table for full view of diagnosis information


def pre_process_step_1(
    diag_path, d_icd_path, pres_path, code_lookup_path, count_path, out_path
):
    """
    Function to read, clean, and process raw diagnosis and prescription drug data.

    Args:
        diag_path: (String) Filepath of raw diagnosis CSV data
        d_icd_path: (String) Filepath of raw diagnosis names CSV data
        pres_path: (String) Filepath of raw prescriptions CSV data
        code_lookup_path: (String) Filepath to output intermediate CSV dataset of diagnoses+name lookups
        count_path: (String) Filepath to output intermediate CSV dataset of diagnosis value counts
        out_path: (String) Filepath to output processed and joined diagnosis+prescriptions dataset

    Returns: Several CSV outputs of intermediate, cleaned data merging diagnosis and prescriptions data
    """

    # Create diagnosis preliminary drug dataset
    dg = pd.read_csv(diag_path)
    dg2 = pd.read_csv(d_icd_path)
    diagnoses = dg.merge(dg2, how="left", on=["icd_code", "icd_version"])
    diagnoses.drop_duplicates(inplace=True)
    diagnoses.reset_index(inplace=True)
    diagnoses["diagnosis"] = (
        diagnoses["icd_version"].astype(str) + "_" + diagnoses["icd_code"].astype(str)
    )

    # Create code lookup
    code_lookup = diagnoses[["diagnosis", "long_title"]].drop_duplicates()
    code_lookup.reset_index(inplace=True, drop=True)
    code_lookup.to_csv(code_lookup_path, index=False)

    # ### Step 2: Weight diagnoses based on `seq_num`
    #
    # #### From MIMIC's table description (https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd/) :
    #
    # `seq_num`
    #
    # The priority assigned to the diagnoses.
    # The priority can be interpreted as a ranking of which diagnoses are “important”,
    # but many caveats to this broad statement exist. For example, patients who are diagnosed with sepsis must
    # have sepsis as their 2nd billed condition. The 1st billed condition must be the infectious agent.
    # There’s also less importance placed on ranking low priority diagnoses “correctly”
    # (as there may be no correct ordering of the priority of the 5th - 10th diagnosis codes, for example).
    #
    # #### How can one deal with multiple diagnoses based on this information?
    #
    # Following the lead of other papers ("Natural language processing of MIMIC-III clinical notes for
    # identifying diagnosis and procedures with neural networks"),
    # I have **only selected the most important diagnosis, where `seq_num` == 1**.

    # Most common 50 diagnoses among those that are the highest-priority in EMR
    common_50 = pd.DataFrame(top_diags.diagnosis.value_counts().keys()[0:50])
    common_50.rename(columns={common_50.columns[0]: "diagnosis"}, inplace=True)

    # Create cleaned dataset
    diagnoses_c = top_diags.merge(common_50[["diagnosis"]], how="inner", on="diagnosis")
    diagnoses_c["id"] = (
        diagnoses_c.subject_id.astype(str) + "_" + diagnoses_c.hadm_id.astype(str)
    )
    diagnoses_clean = diagnoses_c[["id", "diagnosis"]]
    diagnoses_clean.drop_duplicates(inplace=True)
    diagnoses_clean.reset_index(inplace=True, drop=True)

    # What are the value counts?
    diag_counts = diagnoses_clean["diagnosis"].value_counts()
    dcount = pd.DataFrame(diag_counts).reset_index()
    dcount.rename(columns={"index": "diagnosis", "diagnosis": "count"}, inplace=True)
    dcount.to_csv(count_path, index=False)

    # Read in prescriptions.csv
    pscr = pd.read_csv(pres_path, dtype=str)
    pscr["id"] = pscr.subject_id.astype(str) + "_" + pscr.hadm_id.astype(str)
    pscr.drop(columns=["subject_id", "hadm_id"], inplace=True)

    # Trim prescription dataset to relevant columns
    pscr_tr = pscr[["id", "drug"]].copy()
    pscr_tr.drop_duplicates(inplace=True)
    pscr_tr.reset_index(inplace=True, drop=True)

    # Merge main diagnosis on to prescription data
    output = pscr_tr.merge(diagnoses_clean, how="inner", on="id")

    # Clean drug name field slightly and export
    output["drug_clean"] = output["drug"].str.replace(r"\*NF\*", "").str.strip()
    output["drug_clean"] = output["drug_clean"].copy().str.replace(r",", "").str.strip()
    output.drop(columns=["drug"], inplace=True)
    output.to_csv(out_path, index=False)


if __name__ == "__main__":
    pre_process_step_1(
        diag_path="../raw_data/diagnoses_icd.csv",
        d_icd_path="../raw_data/d_icd_diagnoses.csv",
        pres_path="../raw_data/prescriptions.csv",
        code_lookup_path="../intermediate_data/code_lookup.csv",
        count_path="../intermediate_data/diagnoses_by_count_50.csv",
        out_path="../intermediate_data/mimic_cleaned_50_diagnoses.csv",
    )
