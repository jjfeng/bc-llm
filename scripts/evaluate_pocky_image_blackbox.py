import os
import sys
import argparse
import torch
import pickle
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

sys.path.append(os.getcwd())
from scripts.train_bayesian import load_data_partition
from scripts.train_image_blackbox import ImageDataset, get_data
import src.common as common
from scripts.evaluate_bayesian import eval_ensemble
from scripts.evaluate_pocky import get_ci, load_images

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method-name", type=str)
    parser.add_argument("--in-mdl", type=str, help="the learned concept model")
    parser.add_argument(
            "--image-weights",
            type=str,
            default='IMAGENET1K_V2',
            )
    parser.add_argument("--max-obs", type=int, default=-1)
    parser.add_argument("--trained-classes", nargs="*", type=int, default=None)
    parser.add_argument("--dataset-folder", type=str, help="folder of CUB data")
    parser.add_argument("--log-file", type=str, default="_output/log_plot.txt")
    parser.add_argument("--csv-file", type=str, default="_output/ood.csv")
    args = parser.parse_args()
    args.partition = "test"
    return args

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    if args.trained_classes is not None:
        args.seed = args.seed + args.trained_classes[0]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_df, target_domain = load_images(args)
    data_df["y"] = 0
    
    dataset = ImageDataset(data_df)
    dataloader = DataLoader(
            dataset, 
            batch_size=len(dataset),
            shuffle=True, 
            num_workers=torch.cuda.device_count()
            )

    with open(args.in_mdl, "rb") as f:
        model = pickle.load(f)

    X_test, _ = get_data(args.image_weights, device, dataloader)

    pred_probs = model.predict_proba(X_test)
    entropies = -np.sum(np.log(pred_probs) * pred_probs, axis=1)

    logging.info("entropies first two %s", entropies[:2])
    
    logging.info("entropy CI: %s", get_ci(entropies))
    entropy_res = get_ci(entropies)
    pd.DataFrame({
        "method": ["ResNet"],
        "trained_classes": [" ".join(map(str, args.trained_classes))],
        "target_domain": [target_domain],
        "entropy": [entropy_res[0]],
        "entropy_lower_ci": [entropy_res[1]],
        "entropy_upper_ci": [entropy_res[2]],
    }).to_csv(args.csv_file)

if __name__ == "__main__":
    main(sys.argv[1:])
