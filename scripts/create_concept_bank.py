import sys
import logging
import argparse
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
import faiss 
import duckdb
from tqdm import tqdm
import numpy as np

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=str, default="cache_concepts.txt",
                        help="log file")
    parser.add_argument("--database-file", type=str, default="bc_llm_database.db")
    parser.add_argument("--embeddings-file", type=str, default="concept_embeddings.bin")
    parser.add_argument("--embedding-model", 
                        type=str, 
                        choices=["google-bert/bert-base-uncased"], 
                        default="google-bert/bert-base-uncased",
                        help="the model to use for computing embeddings"
                        )
    parser.add_argument("--force", action="store_true", default=False,
                        help="will force rerun generating embeddings for the concept bank, otherwise will not overwrite")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)

    db = duckdb.connect(args.database_file)

    model = SentenceTransformer(args.embedding_model)
    embed_dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(embed_dim)

    tables = db.execute("PRAGMA show_tables").df()
    if not "concept_bank" in tables.name or args.force:
        db.execute("""
            CREATE OR REPLACE TABLE concept_bank (
                idx INTEGER PRIMARY KEY,
                text TEXT
            )
           """
        ) 

        words = []
        for idx, word in tqdm(enumerate(wn.words())):
            word = word.replace("_", " ")
            db.execute("INSERT INTO concept_bank (idx, text) VALUES (?, ?)", (idx, word))
            words.append(word)

        logging.info("Finished inserting words into db")

        embeddings = np.array(model.encode(words))

        logging.info("Finished creating embeddings for all words")
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        logging.info("Added all words to embedding vector db")
        breakpoint()

        faiss.write_index(index, args.embeddings_file)
        print("Saved embeddings")
        logging.info("Saved embeddings")
    else:
        print("concept bank is already created")
        logging.info("concept bank is already created")

if __name__ == "__main__":
    main(sys.argv[1:])
