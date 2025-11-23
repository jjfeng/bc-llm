"""
Script for evaluating bootstrap model coverage
"""
import os
import sys
import logging
import argparse
import torch
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from scripts.train_concept import make_dataset, setup
from scripts.train_lasso import prep_lasso_data
from src.common import load_model

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-concepts", type=int, default=1)
    parser.add_argument("--in-dataset-file", type=str, help="csv of the test data")
    parser.add_argument("--embeddings-file", type=str, help="npz of the embeddings")
    parser.add_argument("--learner-type", type=str)
    parser.add_argument("--in-mdl-file", type=str, help="the learned model")
    parser.add_argument("--log-file", type=str, default="_output/log_plot.txt")
    parser.add_argument("--plot-file", type=str, default="_output/test_scatter.png")
    parser.add_argument("--bootstrap-params", type=str, help="file that stores the values for the bootstrapped parameters for the concept model")
    # currently the code works with llama models
    parser.add_argument(
            "--llm-model-type",
            type=str,
            default="meta-llama/Meta-Llama-3-8B-Instruct",
            choices=["meta-llama/Meta-Llama-3-8B-Instruct"]
            )
    args = parser.parse_args()
    return args

def eval_pred_probs(pred_probs, test_df, plot_file=None):
    # eval trained model
    if "true_prob" in test_df.columns:
        brier = np.mean(np.power(pred_probs - test_df.true_prob.to_numpy().flatten(), 2))
        logging.info("brier against true prob %s", brier)
        oracle_auc = roc_auc_score(
            test_df.y.to_numpy().flatten(),
            test_df.true_prob.to_numpy().flatten(),
            )
        logging.info("oracle auc %s", oracle_auc)

        if plot_file is not None:
            sns.scatterplot(x=test_df.true_prob, y=pred_probs)
            plt.savefig(plot_file)

    concept_mdl_auc = roc_auc_score(
        test_df.y.to_numpy().flatten(),
        pred_probs,
        )
    logging.info("model auc %s", concept_mdl_auc)
    print("model auc", concept_mdl_auc)


def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)

    test_df = pd.read_csv(args.in_dataset_file, index_col=0)
    if args.learner_type.startswith("direct") or args.learner_type.startswith("lasso"):
        # load data for LLM probs
        device, english_filter = setup(args)
        dataset, llm_token_embed = make_dataset(args, english_filter, device)

    # Load the trained model and get predictions
    pred_prob_all = []
    if args.learner_type.startswith("direct"):
        dataloader = DataLoader(dataset, batch_size=len(list(dataset)), shuffle=False)
        cl_model = load_model(args.learner_type, english_filter, llm_token_embed, num_concepts=args.num_concepts, device=device, file=args.in_mdl_file, logging=logging)
        for llm_probs, y in dataloader:
            pred_probs = cl_model(llm_probs)
            pred_probs = pred_probs.detach().cpu().numpy().flatten()
        pred_prob_all.append(pred_probs)

        if args.bootstrap_params is not None:
            for i, state_dict in enumerate(torch.load(args.bootstrap_params)):
                cl_model.load_state_dict(state_dict)
                for llm_probs, y in dataloader:
                    pred_probs = cl_model(llm_probs)
                    pred_probs = pred_probs.detach().cpu().numpy().flatten()
                pred_prob_all.append(pred_probs)
    elif args.learner_type.startswith("lasso"):
        with open(args.in_mdl_file, "rb") as f:
            mdl = pickle.load(f)
        lasso_X = prep_lasso_data(args.learner_type, dataset.tensors[0][:,0,:].cpu().numpy())
        pred_probs = mdl.predict_proba(lasso_X)[:,1]
        pred_prob_all = [pred_probs]
    elif args.learner_type == "oracle":
        with open(args.in_mdl_file, "rb") as f:
            mdl = pickle.load(f)
        pred_probs = mdl.predict_proba(test_df.iloc[:, mdl.oracle_columns])[:,1]
        pred_prob_all = [pred_probs]
    else:
        with open(args.in_mdl_file, "rb") as f:
            mdl_dict = pickle.load(f)
            mdl = mdl_dict["model"]
            vectorizer = mdl_dict["vectorizer"]
        X_vectorized = vectorizer.transform(test_df['sentence'])
        pred_probs = mdl.predict_proba(X_vectorized)[:,1]
        pred_prob_all = [pred_probs]

    pred_prob_all = np.concatenate([probs.reshape((-1,1)) for probs in pred_prob_all], axis=1)

    for i in range(3):
        print("OBS", i, np.sort(pred_prob_all[i,:]))

    print("pred prob all", pred_prob_all.shape)
    print("pred prob all", pred_prob_all)
    min_pred_probs = np.min(pred_prob_all, axis=1)
    max_pred_probs = np.max(pred_prob_all, axis=1)
    print("PRED PROBS", min_pred_probs.shape, max_pred_probs.shape)
    print("MIN", min_pred_probs)
    print("MAX", max_pred_probs)

    true_probs = test_df.true_prob.to_numpy()
    is_below = (true_probs >= min_pred_probs)
    logging.info("percent below %f", np.mean(is_below))
    is_above = (true_probs <= max_pred_probs)
    logging.info("percent above %f", np.mean(is_above))
    logging.info("percent below AND above %f", np.mean(is_above & is_below))

    print("CI WIDTH", max_pred_probs - min_pred_probs)

    logging.info("average CI width %f", np.mean(max_pred_probs - min_pred_probs))

if __name__ == "__main__":
    main(sys.argv[1:])

