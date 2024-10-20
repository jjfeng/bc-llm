import os
import sys
import argparse
import torch
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

sys.path.append(os.getcwd())
from scripts.train_bayesian import load_data_partition
from scripts.train_image_blackbox import ImageDataset, get_data
from scripts.evaluate_bayesian import eval_ensemble

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-obs", type=int, default=-1)
    parser.add_argument("--in-dataset-file", type=str, help="csv of the labelled training data")
    parser.add_argument("--indices-csv", type=str, help="csv of training indices")
    parser.add_argument("--in-mdl", type=str, help="the learned concept model")
    parser.add_argument("--results-csv", type=str, default="_output/log_results.csv")
    parser.add_argument(
            "--image-weights",
            type=str,
            default='IMAGENET1K_V2',
            )
    args = parser.parse_args()
    args.partition = "test"
    return args

def main(args):
    args = parse_args(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_df = load_data_partition(args, init_concepts_file=None, text_summary_column=None)
    num_classes = len(data_df.y.unique())

    dataset = ImageDataset(data_df)
    dataloader = DataLoader(
            dataset, 
            batch_size=len(dataset),
            shuffle=True, 
            num_workers=torch.cuda.device_count()
            )

    with open(args.in_mdl, "rb") as f:
        model = pickle.load(f)

    X_test, y_test = get_data(args.image_weights, device, dataloader)

    pred_prob = model.predict_proba(X_test)
    is_multiclass = len(np.unique(y_test)) > 2
    _, res_df = eval_ensemble(y_test, pred_prob, is_multiclass=is_multiclass)
    print(res_df)
    res_df['method'] = 'ResNet'
    res_df['seed'] = args.seed
    res_df.to_csv(args.results_csv, index=False)

if __name__ == "__main__":
    main(sys.argv[1:])
