"""
Script for assembling imagenette
"""
import os
import sys
import logging
import argparse
import pickle
import pandas as pd
import numpy as np

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-num-obs", type=int, default=20)
    parser.add_argument("--labelled-data-file", type=str, help="csv file with the necessary data elements for training model")
    args = parser.parse_args()
    return args

def main(args):
  args = parse_args(args)
  np.random.seed(args.seed)

  metadata_df = pd.read_csv('notebooks/data/imagenette/imagenette2-320/noisy_imagenette.csv')
  metadata_df['img_path'] = f'notebooks/data/imagenette/imagenette2-320/' + metadata_df['path']
  y_map = {y: i for i, y in enumerate(metadata_df.noisy_labels_0.unique())}
  metadata_df['y'] = metadata_df['noisy_labels_0'].map(y_map)
  metadata_df = metadata_df[metadata_df.is_valid == 0]
  print(metadata_df)
  
  filtered_dfs = []
  for category in range(10):
      print(category)
      labels_i_df = metadata_df[metadata_df.y == category]
      print(labels_i_df)
      if labels_i_df.shape[0] < args.max_num_obs:
        img_list = labels_i_df.img_path.to_numpy()
      else:
        img_list = np.random.choice(labels_i_df.img_path, args.max_num_obs, replace=False)
      print(img_list)
      cls_df = pd.DataFrame({
          'y': [category] * img_list.size,
          'image_path': img_list
      })
      filtered_dfs.append(cls_df)
  filtered_dfs = pd.concat(filtered_dfs).reset_index(drop=True)
      
  print(filtered_dfs.y)
  if args.labelled_data_file:
    filtered_dfs.to_csv(args.labelled_data_file, index=False)

if __name__ == "__main__":
    main(sys.argv[1:])
