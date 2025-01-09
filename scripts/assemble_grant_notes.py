"""
Creates notes for concept extraction from the zsfg dataset
"""
import os
import sys
import logging
import argparse
import pandas as pd
import medspacy
from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

sys.path.append(os.getcwd())
import src.common as common

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--max-section-tokens", type=int, default=2500, help="max section tokens extracted")
    parser.add_argument("--sections-to-keep",
                        nargs="*",
                        type=str,
                        default=["summary_of_hospitalization", "discharge_medications", "discharge_condition"],
                        help="categories of sections to use from the discharge summary"
                        )
    parser.add_argument("--in-csv", type=str,
                        help="the name of the output csv")
    parser.add_argument("--out-csv", type=str,
                        help="the name of the output csv")
    # parser.add_argument("--log-file", type=str, default="_output/zsfg_log.txt",
    #                     help="log file")
    args = parser.parse_args()
    return args

def main(args):
    args = parse_args(args)
    np.random.seed(args.seed)

    notes_df = pd.read_csv(args.in_csv)
    notes_df = notes_df.rename({'y_true':'y'}, axis=1)
    
    nlp = medspacy.load()
    # Note: max_section_length denotes the token length of section. These tokens are computed by medspacy (NOT the tokenizer from
    # the LLM). More info can be found here:
    # https://github.com/medspacy/medspacy/blob/664128ce0af4481a8312e03ed4fdf7b96e092a73/medspacy/section_detection/sectionizer.py#L83
    section_rules_path = os.path.join(os.getcwd(), "note_section_rules/clinical_notes.json")
    config = {"max_section_length": args.max_section_tokens, "rules": section_rules_path}
    sectionizer = nlp.add_pipe("medspacy_sectionizer", config=config)

    section_notes_data = []
    for idx, row in tqdm(notes_df.iterrows(), total=len(notes_df)):
        doc = nlp(row['note_text'])
        text_to_keep = []
        categories = [sec.category for sec in doc._.sections]
        for category, sect in zip(categories, doc._.section_spans):
            text_to_keep += [str(sect) for section in args.sections_to_keep if section == str(category).lower()]

        new_note = " ".join(text_to_keep)
        new_row = row.copy()
        new_row['sentence'] = new_note
        print(new_note)
        section_notes_data.append(new_row)

    # these are medspacy note sections
    sect_notes_df = pd.DataFrame(data=section_notes_data)
    sect_notes_df = sect_notes_df.replace("", np.nan)
    sect_notes_df = sect_notes_df[sect_notes_df['sentence'].notna()]
    sect_notes_df = sect_notes_df.reset_index()
    sect_notes_df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main(sys.argv[1:])
