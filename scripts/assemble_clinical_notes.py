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
    parser.add_argument("--num-chf-train", type=int, default=100, help="max number of observations")
    parser.add_argument("--nonchf-dataset-file",
                        type=str,
                        help="location of your zsfg data for nonchf"
                        )
    parser.add_argument("--chf-dataset-file",
                        type=str,
                        help="location of your zsfg data for chf"
                        )
    parser.add_argument("--sections-to-keep",
                        nargs="*",
                        type=str,
                        default=["summary_of_hospitalization", "discharge_medications", "discharge_condition"],
                        help="categories of sections to use from the discharge summary"
                        )
    parser.add_argument("--out-csv", type=str,
                        help="the name of the output csv")
    parser.add_argument("--log-file", type=str, default="_output/zsfg_log.txt",
                        help="log file")
    args = parser.parse_args()
    return args

def fit_rf(nonchf_df: pd.DataFrame, chf_df: pd.DataFrame, num_chf_train: int = 0):
    # prepare nonchf data
    nonchf_df = nonchf_df.iloc[:,1:].astype(float)
    col_mask = nonchf_df.columns.str.startswith("y_true") | nonchf_df.columns.str.startswith("flowsheet_value_") | nonchf_df.columns.str.startswith("lab_results_value_")
    nonchf_df = nonchf_df.iloc[:,col_mask]
    X_cols = nonchf_df.columns[1:]

    # random select some subset of chf to train model on
    chf_train_idxs = np.random.choice(chf_df.index, size=num_chf_train, replace=False)

    train_X_df = pd.concat([nonchf_df[X_cols], chf_df[X_cols].iloc[chf_train_idxs]])
    train_Y = pd.concat([nonchf_df.y_true, chf_df.y_true.iloc[chf_train_idxs]])
    
    rf_nonchf = RandomForestClassifier(n_estimators=2000, n_jobs=-1, oob_score=True, max_depth=4)
    rf_nonchf.fit(nonchf_df[X_cols], nonchf_df.y_true)
    rf_mix = RandomForestClassifier(n_estimators=2000, n_jobs=-1, oob_score=True, max_depth=4)
    rf_mix.fit(train_X_df, train_Y)
    print("OOB SCORE", rf_mix.oob_score_)
    train_pred_prob = rf_mix.predict_proba(train_X_df)[:,1]
    train_auc = roc_auc_score(y_true=train_Y, y_score=train_pred_prob)
    logging.info("train_auc_mix %f", train_auc)

    # Prepare output for note-based error diagnosis
    eval_df = chf_df.drop(chf_train_idxs)
    eval_df['pred_prob_nonchf'] = rf_nonchf.predict_proba(eval_df[X_cols])[:,1]
    eval_df['pred_logit_nonchf'] = common.get_safe_logit(eval_df.pred_prob_nonchf)
    eval_df['pred_prob_mix'] = rf_mix.predict_proba(eval_df[X_cols])[:,1]
    eval_df['pred_logit_mix'] = common.get_safe_logit(eval_df.pred_prob_mix)

    auc_nonchf = roc_auc_score(y_true=eval_df.y_true, y_score=eval_df.pred_logit_nonchf)
    logging.info("test AUC nonchf %f", auc_nonchf)
    auc_mix = roc_auc_score(y_true=eval_df.y_true, y_score=eval_df.pred_logit_mix)
    logging.info("test AUC mix %f", auc_mix)
    print("aucs test", auc_nonchf, auc_mix)
    return eval_df


def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    np.random.seed(args.seed)

    nonchf_df = pd.read_csv(args.nonchf_dataset_file, index_col=0).reset_index(drop=True)
    print(nonchf_df)
    chf_df = pd.read_csv(args.chf_dataset_file, index_col=0).reset_index(drop=True)
    chf_df = chf_df.drop_duplicates()
    print(chf_df)

    notes_df = fit_rf(nonchf_df, chf_df, args.num_chf_train)
    notes_df = notes_df.rename({'y_true':'y'}, axis=1)
    
    notes_df = notes_df[['pred_prob_nonchf', 'pred_logit_nonchf', 'pred_prob_mix', 'pred_logit_mix', 'y', 'note_text']]

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
