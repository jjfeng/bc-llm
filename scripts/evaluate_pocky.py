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
from src.llm.llm_api import LLMApi
from src.llm.llm_local import LLMLocal
import src.common as common
import src.utils as utils

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--method-name", type=str)
    parser.add_argument("--max-obs", type=int, default=-1)
    parser.add_argument("--num-posterior-iters", type=int, default=1)
    parser.add_argument("--max-num-concepts-extract", type=int, default=10)
    parser.add_argument("--prompt-concepts-file", type=str, default="exp_multi_concept/prompts/concept_questions.txt")
    parser.add_argument("--indices-csv", type=str, help="csv of train/test indices")
    parser.add_argument("--out-extractions", type=str)
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--training-history-file", type=str, help="the learned model")
    parser.add_argument("--is-image", action="store_true", default=False)
    parser.add_argument("--dataset-folder", type=str, help="folder of CUB data")
    parser.add_argument("--bird-class", nargs="*", type=int, default=None)
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
    args = parser.parse_args()
    args.partition = "test"
    return args

def get_ci(entropies):
    mean_ent = np.mean(entropies)
    std_dev_ent = np.sqrt(np.var(entropies))
    return np.array([
        mean_ent,
        mean_ent - 1.96 * std_dev_ent,
        mean_ent + 1.96 * std_dev_ent
    ])

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    history = TrainingHistory().load(args.training_history_file)
    if args.bird_class is None:
        test_data_df = pd.DataFrame({
            "image_path": ["exp_cub_birds_existing/data/pocky.jpg", "exp_cub_birds_existing/data/pocky2.jpg"]
        })
    else:
        images_file = os.path.join(args.dataset_folder, "CUB_200_2011/images.txt")
        image_labels_file = os.path.join(args.dataset_folder, "CUB_200_2011/image_class_labels.txt")
        image_folder = os.path.join(args.dataset_folder, "CUB_200_2011/images")
        labels_df = pd.read_csv(image_labels_file, delim_whitespace=True, index_col=0, header=None, names=["orig_bird_label"])
        images_df = pd.read_csv(images_file, delim_whitespace=True, names=["image_idx", "image_path"])
        images_df['image_path'] = image_folder + '/' + images_df.image_path
        images_df = pd.concat([images_df, labels_df], axis=1)
        test_data_df = images_df[images_df.orig_bird_label.isin(args.bird_class)].iloc[:args.max_obs]

    # load any past extractions
    all_extracted_features_dict = {}
    if os.path.exists(args.out_extractions):
        with open(args.out_extractions, "rb") as f:
            all_extracted_features_dict = pickle.load(f)

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
    
    all_extracted_features_dict = common.extract_features_by_llm(
        llm_dict['extraction'],
        test_data_df,
        meta_concept_dicts=all_concepts_to_extract,
        all_extracted_features_dict=all_extracted_features_dict,
        prompt_file=args.prompt_concepts_file, 
        batch_size=args.batch_size,
        extraction_file=args.out_extractions,
        max_new_tokens=1000, # TODO: do not hard code?
        is_image=args.is_image,
        max_num_concepts_extract=args.max_num_concepts_extract,
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
        if test_data_df.shape[0] < 5:
            logging.info("posterior pred prob (iter %d) %s", i, pd.DataFrame(posterior_pred_probs).to_latex(float_format='%.2f'))
        else:
            logging.info("entropy (iter %d) %s", i, get_ci(entropies))

if __name__ == "__main__":
    main(sys.argv[1:])
