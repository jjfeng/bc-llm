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

from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import pearsonr
from sklearn.calibration import CalibrationDisplay
from sklearn.calibration import calibration_curve

sys.path.append(os.getcwd())
from src.training_history import TrainingHistory
from scripts.train_bayesian import load_data_partition, load_llms
import src.common as common
import src.utils as utils

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-concept-size", type=int, default=20)
    parser.add_argument("--batch-obs-size", type=int, default=1)
    parser.add_argument("--cache-file", type=str, default="cache.db")
    parser.add_argument("--method-name", type=str)
    parser.add_argument("--max-obs", type=int, default=-1)
    parser.add_argument("--num-posterior-iters", type=int, default=1)
    parser.add_argument("--max-num-concepts-extract", type=int, default=10)
    parser.add_argument("--prompt-concepts-file", type=str)
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
            )
    parser.add_argument(
            "--llm-iter-type",
            type=str,
            )
    parser.add_argument(
            "--llm-extraction-type",
            type=str,
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
        # if np.unique(y_true[indices]).size != y_pred.shape[1]:
        #     continue

        try:
            score = eval_func(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
        except:
            continue
    return eval_value, np.quantile(bootstrapped_scores, [0.025, 0.975]) if bootstrapped_scores else None

def to_table_str(bootstrap_res):
    return f"{bootstrap_res[0]:.3f} ({bootstrap_res[1][0]:.3f}, {bootstrap_res[1][1]:.3f})"

def brier_multi(targets, probs):
    if targets.dtype == int:
        target_onehot = np.zeros((targets.size, targets.max() + 1))
        target_onehot[np.arange(targets.size), targets] = 1
        return np.mean(np.sum((probs - target_onehot)**2, axis=1))
    else:
        return np.mean((targets - probs[:,1])**2)

def get_ece(
    y_true,
    y_prob,
    pos_label=None,
    n_bins=10,
    strategy="uniform",
):
    y_true = y_true == pos_label
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.searchsorted(bins[1:-1], y_prob)
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))[:-1]
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))[:-1]
    bin_total = np.bincount(binids, minlength=len(bins))[:-1]

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    return np.sum(np.abs(prob_true - prob_pred) * bin_total[nonzero]/np.sum(bin_total))


def get_all_eces(targets, probs, n_bins=5):
    uniq_pos_labels = np.sort(np.unique(targets))
    if uniq_pos_labels.size == 2:
        return get_ece(targets, probs[:,1], n_bins=n_bins, strategy="uniform")
    else:
        all_eces = []
        for pos_label in uniq_pos_labels:
            probs_label = probs[:, pos_label]
            ece = get_ece(targets, probs_label, pos_label=pos_label, n_bins=n_bins, strategy="uniform")
            all_eces.append(ece)
        return np.mean(all_eces)


def eval_ensemble(y_test, posterior_pred_prob, true_prob = None, is_multiclass=False, classes=None):
    posterior_pred_prob = common.get_safe_prob(posterior_pred_prob)
    if classes is not None:
        test_f1 = get_bootstrap_ci(y_test, posterior_pred_prob, lambda x1, x2: f1_score(x1, classes[np.argmax(x2, axis=1)], average="macro"))
        test_acc = get_bootstrap_ci(y_test, posterior_pred_prob, lambda x1, x2: np.mean(classes[np.argmax(x2, axis=1)] == x1))
    else:
        test_f1 = get_bootstrap_ci(y_test, posterior_pred_prob, lambda x1, x2: f1_score(x1, np.argmax(x2, axis=1), average="macro"))
        test_acc = get_bootstrap_ci(y_test, posterior_pred_prob, lambda x1, x2: np.mean(np.argmax(x2, axis=1) == x1))
    logging.info(f'test acc ensemble: {test_acc}')
    print(f'test acc ensemble: {test_acc}')
    logging.info(f'test f1 ensemble: {test_f1}')
    print(f'test f1 ensemble: {test_f1}')
    
    if np.unique(y_test).size == posterior_pred_prob.shape[1]:
        if is_multiclass:
            test_auc = get_bootstrap_ci(y_test, posterior_pred_prob, lambda x1, x2: roc_auc_score(y_true=x1, y_score=x2, multi_class="ovr"))
        else:
            test_auc = get_bootstrap_ci(y_test, posterior_pred_prob[:,1], lambda x1, x2: roc_auc_score(y_true=x1, y_score=x2))
    else:
        y_uniq_idxs = np.unique(y_test)
        sub_posterior_pred_prob = posterior_pred_prob[:, y_uniq_idxs]
        sub_posterior_pred_prob /= sub_posterior_pred_prob.sum(axis=1, keepdims=True)
        test_auc = get_bootstrap_ci(y_test, sub_posterior_pred_prob, lambda x1, x2: roc_auc_score(y_true=x1, y_score=x2, multi_class="ovr"))
    logging.info(f'test auc ensemble: {test_auc}')
    print(f'test auc ensemble: {test_auc}')

    ref_val = true_prob if true_prob is not None else y_test
    calib_display = None
    brier_score = (np.nan, (np.nan, np.nan))
    log_lik = get_bootstrap_ci(y_test, posterior_pred_prob, lambda x1, x2: np.mean(common.get_log_liks(x1, x2, is_multiclass=True)))
    brier_score = get_bootstrap_ci(ref_val, posterior_pred_prob, lambda x1, x2: brier_multi(targets=x1, probs=x2))
    ece_score = get_bootstrap_ci(y_test, posterior_pred_prob, lambda x1, x2: get_all_eces(targets=x1, probs=x2))
    logging.info(f'test brier ensemble: {brier_score}')
    print(f'test brier ensemble: {brier_score}')
    logging.info(f'test ece ensemble: {ece_score}')
    print(f'test ece ensemble: {ece_score}')
    if not is_multiclass:
        log_lik = get_bootstrap_ci(ref_val, posterior_pred_prob[:,1], lambda x1, x2: np.mean(common.get_log_liks(x1, x2, is_multiclass=False)))

        prob_true_bin, prob_pred_bin = utils.get_calibration_curve(ref_val, posterior_pred_prob[:,1], n_bins=10)
        calib_display = CalibrationDisplay(prob_true_bin, prob_pred_bin, posterior_pred_prob[:,1])
    
    logging.info(f'test log lik ensemble: {log_lik}')
    print(f'test log lik ensemble: {log_lik}')
    
    return calib_display, pd.DataFrame({
        "acc": [test_acc[0]],
        "acc_str": [to_table_str(test_acc)],
        "auc": [test_auc[0]],
        "auc_str": [to_table_str(test_auc)],
        "brier": [brier_score[0]],
        "brier_str": [to_table_str(brier_score)],
        "ece": [ece_score[0]],
        "ece_str": [to_table_str(ece_score)],
        "log_lik": [log_lik[0]],
        "log_lik_str": [to_table_str(log_lik)],
        "f1": [test_f1[0]],
        "f1_str": [to_table_str(test_f1)],
    })

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    history = TrainingHistory().load(args.training_history_file)
    print("NUM HISTORY ITERS", history.num_iters)

    if not hasattr(history, "force_keep_cols"):
        history.force_keep_cols = None
    test_data_df = load_data_partition(args)
    
    # load any past extractions
    all_extracted_features_dict = {}
    if args.out_extractions is not None and os.path.exists(args.out_extractions):
        with open(args.out_extractions, "rb") as f:
            all_extracted_features_dict = pickle.load(f)

    llm_dict = load_llms(args)

    start_iter = max(0, history.num_iters - args.num_posterior_iters)
    print("start_iter", start_iter, history.num_iters)
    all_concepts_to_extract = [concept_dict for concept_dicts in history._concepts[start_iter:history.num_iters] for concept_dict in concept_dicts]
    
    all_extracted_features_dict = common.extract_features_by_llm_grouped(
        llm_dict['extraction'],
        test_data_df,
        meta_concept_dicts=all_concepts_to_extract,
        prompt_file=args.prompt_concepts_file, 
        batch_size=args.batch_size,
        batch_concept_size=args.batch_concept_size,
        group_size=args.batch_obs_size,
        is_image=args.is_image,
        max_section_length=args.max_section_length
    )

    fig, ax = plt.subplots()
    pred_probs = []
    y_test = test_data_df['y'].to_numpy().flatten()
    is_multiclass = np.unique(y_test).size > 2
    true_prob = test_data_df['true_prob'].to_numpy().flatten() if 'true_prob' in test_data_df.columns else None
    for i in range(start_iter, history.num_iters):
        concept_dicts = history._concepts[i]
        logging.info("ITER %d %s (%d)", i, [c['concept'] for c in concept_dicts][:10], len(concept_dicts))
        # logging.info("ITER %d %s %s", i, history._coefs[i], history._intercepts[i])
        extracted_features = common.get_features(
            concept_dicts=concept_dicts,
            all_extracted_features=all_extracted_features_dict,
            dset=test_data_df,
            force_keep_columns=history.force_keep_cols if hasattr(history, 'force_keep_cols') else None
        )
        model = history.get_model(index=i)
        pred_prob_i = model.predict_proba(extracted_features)
        logging.info("===INDIV===")
        _, indiv_res_df = eval_ensemble(y_test, pred_prob_i, true_prob=true_prob, is_multiclass=is_multiclass, classes=model.classes_)
        logging.info(f'test auc index {i} {indiv_res_df.auc[0]}')
        print(f'test auc index {i} {indiv_res_df.auc[0]}')
        pred_probs.append(pred_prob_i[np.newaxis,:])
        posterior_pred_prob = np.mean(np.concatenate(pred_probs, axis=0), axis=0)
        
        # if (i - start_iter) % 4 == 0:
        print(f"EVAL ITER {i}")
        logging.info("===ENSEMBLE===")
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

    with open(args.out_extractions, "wb") as f:
        pickle.dump(all_extracted_features_dict, f)

if __name__ == "__main__":
    main(sys.argv[1:])
