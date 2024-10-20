"""
Creates a dataset of sentence, label where the sentence is a one-liner from clinical notes and the label is is_female
"""
import os
import sys
import argparse
import pandas as pd
import duckdb
import csv

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument( "--db-file", 
                        type=str, 
                        default = "/mnt/efs/home/hadoop/zsfg_database.db", 
                        help="location of your zsfg database"
                        ) 
    parser.add_argument( "--out-csv", type=str, help="the name of the output csv") 
    parser.add_argument( "--num-pats", default=700, type=int, help="number of unique patients to pull notes for") 
    args = parser.parse_args()
    return args

convert_to_sql = lambda ids: "(" + ",".join(["'%s'" % e for e in ids]) + ")"

def pull_oneliners(db , num_pats: int) -> pd.DataFrame:
    pat_query = f"""
    SELECT DISTINCT pat_id_surrogate 
    FROM notes
    ORDER BY pat_id_surrogate
    LIMIT {num_pats}
    ;
    """
    pat_ids_df = db.execute(pat_query).df()
    pat_ids_sql = convert_to_sql(pat_ids_df.pat_id_surrogate.values)

    query = f"""
        SELECT 
            notes.pat_id_surrogate,
            notes.note_id,
            notes.note_text,
            dem.patient_sex
        FROM notes
        JOIN demographics AS dem on notes.pat_id_surrogate = dem.pat_id_surrogate
        WHERE notes.pat_id_surrogate IN {pat_ids_sql} AND 
            note_type = 'Discharge Summary' AND 
            note_text like '%Brief History Leading to this Hospitalization%'
        ORDER BY notes.pat_id_surrogate, notes.note_id
        ;
    """
    oneliner_df = db.execute(query).df()
    return oneliner_df

def get_oneliner(note_text: str) -> str:
    split_note_text = note_text.split("    ") 
    oneliner = [text for text in split_note_text if "Brief History Leading to this Hospitalization" in text]
    if oneliner:
        text = oneliner[0]
        cleaned_oneliner = text.replace("\xa0", " ").replace("      ", " ")
        cleaned_oneliner = cleaned_oneliner.replace("Brief History Leading to this Hospitalization", "").strip()
        return str(cleaned_oneliner)
    else:
        return np.nan

def main(args):
    args = parse_args(args)
    db = duckdb.connect(args.db_file, read_only = True)

    df = pull_oneliners(db, args.num_pats)

    df = df.drop_duplicates(subset=['pat_id_surrogate'], keep='first')
    df['sentence'] = df['note_text'].apply(get_oneliner)

    df['sentence'] = df['sentence'].astype(str) 
    df = df[~df.sentence.apply(lambda text: text == '')]

    df['label'] = df['patient_sex'].apply(lambda gender: 1 if gender == 'Female' else 0)
    df = df.drop(columns = ['patient_sex', 'pat_id_surrogate', 'note_text', 'note_id'])

    df.to_csv(args.out_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)

if __name__ == "__main__":
    main(sys.argv[1:])
