import os
import sys
import logging
import argparse
import numpy as np
import torch
import pickle

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV

sys.path.append(os.getcwd())
from scripts.train_bayesian import load_data_partition
import src.common as common

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
RESNET_IMAGE_SIZE = 224

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-obs", type=int, default=-1)
    parser.add_argument("--in-dataset-file", type=str, help="csv of the labelled training data")
    parser.add_argument("--indices-csv", type=str, help="csv of training indices")
    parser.add_argument("--out-mdl", type=str, help="the learned concept model")
    parser.add_argument("--log-file", type=str, default="_output/log_train_concept.txt")
    parser.add_argument(
            "--image-weights",
            type=str,
            default='IMAGENET1K_V2',
            )
    args = parser.parse_args()
    args.partition = "train"
    return args

class ImageDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(RESNET_IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        label = self.df.iloc[idx]["y"] 
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        return image, label

def get_data(image_weights, device, dataloader):
    model = models.resnet50(weights=image_weights)
    model.to(device)

    with torch.no_grad(): # Disable gradients for faster inference
        for images, labels in dataloader:
            images.to(device)
            X = model(images).detach().cpu().numpy()
            y = labels.detach().cpu().numpy()

    return X, y

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    logging.info(args)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_df = load_data_partition(args, init_concepts_file=None, text_summary_column=None)
    num_classes = len(data_df.y.unique())

    dataset = ImageDataset(data_df)
    dataloader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            num_workers=torch.cuda.device_count()
            )

    X_train, y_train = get_data(args.image_weights, device, dataloader)

    model = LogisticRegressionCV(max_iter=10000, penalty="l2",Cs=20, solver="saga", multi_class="multinomial")
    model.fit(X_train, y_train)

    pred_prob = model.predict_proba(X_train)

    auc = roc_auc_score(y_score=pred_prob, y_true=y_train, multi_class="ovr")

    print(f"Res net train AUC {auc}")
    logging.info("Res net train AUC %f", auc)

    with open(args.out_mdl, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main(sys.argv[1:])
