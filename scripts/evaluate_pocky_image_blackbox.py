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

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--in-mdl", type=str, help="the learned concept model")
    parser.add_argument(
            "--image-weights",
            type=str,
            default='IMAGENET1K_V2',
            )
    parser.add_argument("--log-file", type=str, default="_output/log_plot.txt")
    args = parser.parse_args()
    args.partition = "test"
    return args

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_df = pd.DataFrame({
        "image_path": ["exp_cub_birds_existing/data/pocky.jpg", "exp_cub_birds_existing/data/pocky2.jpg"],
        "y": [0,0] # fake labels
    })

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
    
    posterior_pred_prob = pd.DataFrame(pred_probs)
    logging.info("pred prob %s", posterior_pred_prob.to_latex(float_format='%.2f'))

if __name__ == "__main__":
    main(sys.argv[1:])
