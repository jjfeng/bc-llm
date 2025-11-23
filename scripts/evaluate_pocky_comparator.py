"""
Script for evaluating CBM on pocky or OOD bird images
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

from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from sklearn.calibration import CalibrationDisplay

sys.path.append(os.getcwd())
from src.training_history import TrainingHistory
from scripts.train_bayesian import load_data_partition, load_llms
from scripts.evaluate_pocky import get_ci, load_images

import src.common as common
import src.utils as utils

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-concept-size", type=int, default=20)
    parser.add_argument("--method-name", type=str)
    parser.add_argument("--max-obs", type=int, default=-1)
    parser.add_argument("--num-posterior-iters", type=int, default=1)
    parser.add_argument("--prompt-concepts-file", type=str, default="exp_multi_concept/prompts/concept_questions.txt")
    parser.add_argument("--indices-csv", type=str, help="csv of train/test indices")
    parser.add_argument("--cache-file", type=str)
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--training-history-file", type=str, help="the learned model")
    parser.add_argument("--is-image", action="store_true", default=False)
    parser.add_argument("--dataset-folder", type=str, help="folder of CUB data")
    parser.add_argument("--trained-classes", nargs="*", type=int, default=None)
    parser.add_argument("--max-section-length", type=int, default=None)
    parser.add_argument(
            "--llm-model-type",
            type=str,
            default=None,
            choices=[
                "gpt-4o-mini",
                "versa-gpt-4o-2024-05-13",
                "meta-llama/Meta-Llama-3.1-8B-Instruct", 
                "meta-llama/Meta-Llama-3.1-70B-Instruct", 
                "meta-llama/Llama-3.2-11B-Vision-Instruct" 
                ]
            )
    parser.add_argument(
            "--llm-iter-type",
            type=str,
            default=None,
            choices=[
                "gpt-4o-mini",
                "versa-gpt-4o-2024-05-13",
                "meta-llama/Meta-Llama-3.1-8B-Instruct", 
                "meta-llama/Meta-Llama-3.1-70B-Instruct", 
                "meta-llama/Llama-3.2-11B-Vision-Instruct" 
                ]
            )
    parser.add_argument(
            "--llm-extraction-type",
            type=str,
            default=None,
            choices=[
                "gpt-4o-mini",
                "versa-gpt-4o-2024-05-13",
                "meta-llama/Meta-Llama-3.1-8B-Instruct", 
                "meta-llama/Meta-Llama-3.1-70B-Instruct", 
                "meta-llama/Llama-3.2-11B-Vision-Instruct" 
                ]
            )
    parser.add_argument("--log-file", type=str, default="_output/log_plot.txt")
    parser.add_argument("--csv-file", type=str, default="_output/ood.csv")
    args = parser.parse_args()
    args.partition = "test"
    assert args.cache_file is not None
    return args

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    if args.trained_classes is not None:
        args.seed = args.seed + args.trained_classes[0]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    history = TrainingHistory().load(args.training_history_file)
    test_data_df, target_domain = load_images(args)
    llm_dict = load_llms(args)

    start_iter = max(0, history.num_iters - args.num_posterior_iters)
    print("start_iter", start_iter, history.num_iters)
    if args.method_name == "boosting":
        iter_list = [history.num_iters - 1]
    elif args.method_name == "baseline":
        iter_list = [0]
    else:
        iter_list = list(range(start_iter, history.num_iters))
    all_concepts_to_extract = [concept_dict for iter in iter_list for concept_dict in history._concepts[iter]]
    
    all_extracted_features_dict = common.extract_features_by_llm_grouped(
        llm_dict['extraction'],
        test_data_df,
        meta_concept_dicts=all_concepts_to_extract,
        prompt_file=args.prompt_concepts_file, 
        batch_size=args.batch_size,
        batch_concept_size=args.batch_concept_size,
        max_new_tokens=3000,
        group_size=1,
        is_image=args.is_image,
        max_section_length=args.max_section_length
    )

    pred_probs = []
    for i in iter_list:
        concept_dicts = history._concepts[i]
        logging.info("ITER %d %s", i, [c['concept'] for c in concept_dicts])
        logging.info("ITER %d %s %s", i, history._coefs[i], history._intercepts[i])
        extracted_features = common.get_features(
            concept_dicts=concept_dicts,
            all_extracted_features=all_extracted_features_dict,
            dset=test_data_df,
            force_keep_columns=history.force_keep_cols if hasattr(history, 'force_keep_cols') else None
        )
        model = history.get_model(index=i)
        pred_probs_i = model.predict_proba(extracted_features)[np.newaxis,:]
        pred_probs.append(pred_probs_i)
    posterior_pred_probs = np.mean(np.concatenate(pred_probs, axis=0), axis=0)
    entropies = -np.sum(np.log(posterior_pred_probs) * posterior_pred_probs, axis=1)

    logging.info("entropies first two %s", entropies[:2])

    logging.info("entropy CI: %s", get_ci(entropies))
    entropy_res = get_ci(entropies)
    pd.DataFrame({
        "method": [args.method_name],
        "trained_classes": [" ".join(map(str, args.trained_classes))],
        "target_domain": [target_domain],
        "entropy": [entropy_res[0]],
        "entropy_lower_ci": [entropy_res[1]],
        "entropy_upper_ci": [entropy_res[2]],
    }).to_csv(args.csv_file)

if __name__ == "__main__":
    main(sys.argv[1:])
