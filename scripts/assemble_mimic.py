"""
Creates notes for concept extraction from the mimic dataset
"""
import os
import sys
import argparse
import pandas as pd
import medspacy
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

mappings = {
    "label_employment": {"func": lambda x: convert_yes_no(x), "data_type": "category"},
    "label_education":  {"func": lambda x: convert_true_false(x), "data_type": "bool"},
    "label_community_absent": {"func": lambda x: convert_true_false(x), "data_type": "bool"},
    "label_community_present": {"func": lambda x: convert_true_false(x), "data_type": "bool"},
    "label_housing": {"func": lambda x: convert_yes_no(x), "data_type": "category"},
    "label_alcohol": {"func": lambda x: convert_category(x), "data_type": "category"},
    "label_tobacco": {"func": lambda x: convert_category(x), "data_type": "category"},
    "label_drugs": {"func": lambda x: convert_category(x), "data_type": "category"}
}


def convert_true_false(value):
    if value == 0:
        return False
    elif value == 1:
        return True


def convert_yes_no(value):
    if value == 1:
        return 'True'
    elif value == 2:
        return 'False'
    else:
        return 'None'


def convert_category(value):
    if value == 0:
        return 'None'
    elif value == 1:
        return 'Present'
    elif value == 2:
        return 'Past'
    elif value == 3:
        return 'Never'
    else:
        return 'Unsure'


def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dataset-file",
                        type=str,
                        default="exp_mimic/data/mimic_social_history.csv",
                        help="location of your mimic data"
                        )
    parser.add_argument("--sections-to-keep",
                        nargs="*",
                        type=str,
                        default=['brief_hospital_course', 'social_history'],
                        help="sections to use from the discharge summary"
                        )
    parser.add_argument("--out-csv", type=str,
                        help="the name of the output train csv")
    parser.add_argument("--max-obs", type=int, default=1000,
                        help="max number of observations")
    args = parser.parse_args()
    return args


def main(args):
    args = parse_args(args)
    if args.max_obs > 0:
        notes_df = pd.read_csv(args.in_dataset_file)[:args.max_obs]
    else:
        notes_df = pd.read_csv(args.in_dataset_file)

    for col, info in mappings.items():
        notes_df[col] = notes_df[col].apply(info["func"])
        notes_df[col] = notes_df[col].astype(info["data_type"])

    category_cols = notes_df[mappings.keys()].select_dtypes(
        include=['category']).columns
    try:
        encoder = OneHotEncoder(sparse=False)
    except:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_data = encoder.fit_transform(notes_df[category_cols])
    notes_df = notes_df.drop(columns=category_cols)
    encoded_df = pd.DataFrame(
        encoded_data, columns=encoder.get_feature_names_out())
    notes_df = pd.concat((notes_df, encoded_df), axis=1)

    nlp = medspacy.load()
    # Note: max_section_length denotes the token length of section. These tokens are computed by medspacy (NOT the tokenizer from
    # the LLM). More info can be found here:
    # https://github.com/medspacy/medspacy/blob/664128ce0af4481a8312e03ed4fdf7b96e092a73/medspacy/section_detection/sectionizer.py#L83
    config = {"max_section_length": 2500}
    sectionizer = nlp.add_pipe("medspacy_sectionizer", config=config)

    label_cols = [col for col in notes_df.columns if "label" in col]
    sections_to_keep = [section.replace("_", " ")
                        for section in args.sections_to_keep]
    section_notes_data = []
    for idx, row in tqdm(notes_df.iterrows(), total=len(notes_df)):
        doc = nlp(row['note_text'])
        text_to_keep = []
        for sect_title, sect in zip(doc._.section_titles, doc._.section_spans):
            text_to_keep += [str(sect)
                             for section in sections_to_keep if section in str(sect_title).lower()]

        new_note = " ".join(text_to_keep)
        new_row = row[label_cols].copy()
        new_row['sentence'] = new_note
        section_notes_data.append(new_row)

    # these are medspacy note sections
    sect_notes_df = pd.DataFrame(data=section_notes_data)
    sect_notes_df.to_csv(args.out_csv)


if __name__ == "__main__":
    main(sys.argv[1:])
