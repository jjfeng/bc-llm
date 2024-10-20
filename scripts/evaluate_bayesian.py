"""
Script for evaluating trained bayesian model
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
    parser.add_argument("--in-dataset-file", type=str, help="csv of data")
    parser.add_argument("--indices-csv", type=str, help="csv of train/test indices")
    parser.add_argument("--out-extractions", type=str)
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--training-history-file", type=str, help="the learned model")
    parser.add_argument("--is-image", action="store_true", default=False)
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
    parser.add_argument("--result-csv-file", type=str, default="_output/res.csv")
    parser.add_argument("--calib-plot-file", type=str, default="_output/calib.png")
    args = parser.parse_args()
    args.partition = "test"
    return args

def get_bootstrap_ci(y_true, y_pred, eval_func, n_bootstraps = 1000):
    eval_value = eval_func(y_true, y_pred)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = eval_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    return eval_value, np.quantile(bootstrapped_scores, [0.025, 0.975])

def to_table_str(bootstrap_res):
    return f"{bootstrap_res[0]:.3f} ({bootstrap_res[1][0]:.3f}, {bootstrap_res[1][1]:.3f})"

def eval_ensemble(y_test, posterior_pred_prob, true_prob = None, is_multiclass=False):
    posterior_pred_prob = common.get_safe_prob(posterior_pred_prob)
    if is_multiclass:
        test_auc = get_bootstrap_ci(y_test, posterior_pred_prob, lambda x1, x2: roc_auc_score(y_true=x1, y_score=x2, multi_class="ovr"))
    else:
        test_auc = get_bootstrap_ci(y_test, posterior_pred_prob[:,1], lambda x1, x2: roc_auc_score(y_true=x1, y_score=x2))
    logging.info(f'test auc ensemble: {test_auc}')
    print(f'test auc ensemble: {test_auc}')
    ref_val = true_prob if true_prob is not None else y_test
    
    calib_display = None
    brier_score = (np.nan, (np.nan, np.nan))
    log_lik = get_bootstrap_ci(y_test, posterior_pred_prob, lambda x1, x2: np.mean(common.get_log_liks(x1, x2, is_multiclass=True)))
    if not is_multiclass:
        brier_score = get_bootstrap_ci(ref_val, posterior_pred_prob[:,1], lambda x1, x2: np.mean(np.power(x1 - x2, 2)))
        logging.info(f'test brier ensemble: {brier_score}')
        print(f'test brier ensemble: {brier_score}')
        log_lik = get_bootstrap_ci(ref_val, posterior_pred_prob[:,1], lambda x1, x2: np.mean(common.get_log_liks(x1, x2, is_multiclass=False)))

        prob_true_bin, prob_pred_bin = utils.get_calibration_curve(ref_val, posterior_pred_prob[:,1], n_bins=10)
        calib_display = CalibrationDisplay(prob_true_bin, prob_pred_bin, posterior_pred_prob[:,1])
    
    logging.info(f'test log lik ensemble: {log_lik}')
    print(f'test log lik ensemble: {log_lik}')
    
    return calib_display, pd.DataFrame({
        "auc": [test_auc[0]],
        "auc_str": [to_table_str(test_auc)],
        "brier": [brier_score[0]],
        "brier_str": [to_table_str(brier_score)],
        "log_lik": [log_lik[0]],
        "log_lik_str": [to_table_str(log_lik)],
    })

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    history = TrainingHistory().load(args.training_history_file)
    if not hasattr(history, "force_keep_cols"):
        history.force_keep_cols = None
    test_data_df = load_data_partition(args)
    
    # load any past extractions
    all_extracted_features_dict = {}
    if os.path.exists(args.out_extractions):
        with open(args.out_extractions, "rb") as f:
            all_extracted_features_dict = pickle.load(f)

    llm_dict = load_llms(args)

    start_iter = max(0, history.num_iters - args.num_posterior_iters)
    print("start_iter", start_iter, history.num_iters)
    all_concepts_to_extract = [concept_dict for concept_dicts in history._concepts[start_iter:history.num_iters] for concept_dict in concept_dicts]
    
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

    fig, ax = plt.subplots()
    pred_probs = []
    y_test = test_data_df['y'].to_numpy().flatten()
    is_multiclass = np.unique(y_test).size > 2
    true_prob = test_data_df['true_prob'].to_numpy().flatten() if 'true_prob' in test_data_df.columns else None
    for i in range(start_iter, history.num_iters):
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
        pred_prob_i = model.predict_proba(extracted_features)
        _, indiv_res_df = eval_ensemble(y_test, pred_prob_i, true_prob=true_prob, is_multiclass=is_multiclass)
        logging.info(f'test auc index {i} {indiv_res_df.auc[0]}')
        print(f'test auc index {i} {indiv_res_df.auc[0]}')
        pred_probs.append(pred_prob_i[np.newaxis,:])
        posterior_pred_prob = np.mean(np.concatenate(pred_probs, axis=0), axis=0)

        if (i - start_iter) % 4 == 0:
            print(f"EVAL ITER {i}")
            calib_display, res_df = eval_ensemble(y_test, posterior_pred_prob, true_prob=true_prob, is_multiclass=is_multiclass)
            if calib_display is not None:
                calib_display.plot(ax=ax, name=f"iter{i}")

    calib_display, res_df = eval_ensemble(y_test, posterior_pred_prob, true_prob=true_prob, is_multiclass=is_multiclass)
    if calib_display is not None:
        calib_display.plot(ax=ax, name=f"iter{i}")
        plt.legend()
        plt.savefig(args.calib_plot_file)

    res_df['method'] = args.method_name
    res_df['seed'] = args.seed
    res_df.to_csv(args.result_csv_file, index=False)

if __name__ == "__main__":
    main(sys.argv[1:])
