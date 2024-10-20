"""
Script for extracting llm summary using spacy
"""

import os
import asyncio
import sys
import logging
import argparse
import torch
import numpy as np
import pandas as pd
import scipy
import spacy
from spacy import displacy
import en_core_sci_sm
from tqdm import tqdm


def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--in-dataset-file", type=str, help="csv of the data we want to learn concepts for")
    parser.add_argument("--spacy-outputs-file", type=str, help="csv file with concepts")
    parser.add_argument("--log-file", type=str, default="_output/log_extract.txt")
    args = parser.parse_args()
    return args

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if os.path.exists(args.spacy_outputs_file):
        data_df = pd.read_csv(args.spacy_outputs_file, header=0)
    else:
        data_df = pd.read_csv(args.in_dataset_file, index_col=0, header=0)
        data_df['spacy_output'] = [""] * len(data_df)

    concepts_is_missing = (data_df['spacy_output'].str.len() == 0) | data_df.spacy_output.isna()
    missing_idxs = np.where(concepts_is_missing)[0]
    logging.info("missing %s", missing_idxs)
    print("missing", missing_idxs)
    if missing_idxs.size == 0:
        return
    
    new_data_df = data_df.iloc[missing_idxs]

    nlp_sm = en_core_sci_sm.load()
    spacy_outputs = []
    for text in tqdm(new_data_df.sentence):
        doc = nlp_sm(text)
        entities = [str(ent).strip() for ent in doc.ents]
        spacy_outputs.append(",".join(entities))
    fill_in_idxs = missing_idxs[:len(spacy_outputs)]
    data_df["spacy_output"].iloc[fill_in_idxs] = spacy_outputs
    data_df.to_csv(args.spacy_outputs_file, index=False)
        

if __name__ == "__main__":
    main(sys.argv[1:])
