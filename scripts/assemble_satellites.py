"""
Script for assembling satellite data
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
import torchvision.transforms as transforms

CLASS_DICT = {
  "aeroway": [
    "airport",
    "airport_hangar",
    "airport_terminal",
    "helipad",
    "runway"
  ],
  "amenity": [
    "educational_institution",
    "hospital",
    "fire_station",
    "police_station",
    "place_of_worship",
    "fountain",
    "parking_lot_or_garage",
    "waste_disposal",
    "water_treatment_facility",
    "gas_station"
  ],
  "barrier": [
    "border_checkpoint",
    "toll_booth"
  ],
  "building": [
    "barn",
    "tower",
    "single-unit_residential",
    "multi-unit_residential",
    "office_building"
  ],
  "highway": [
    "road_bridge",
    "tunnel_opening",
    "interchange",
    "flooded_road"
  ],
  "historic": [
    "archaeological_site",
    "burial_site",
    "lighthouse"
  ],
  "agriculture_landuse": [
    "crop_field",
    "aquaculture"
  ],
  "industrial_landuse": [
    "surface_mine",
    "factory_or_powerplant",
    "smokestack",
    "storage_tank",
  ],
  "leisure": [
    "amusement_park",
    "golf_course",
    "stadium",
    "race_track",
    "swimming_pool",
    "recreational_facility",
    "park",
    "zoo"
  ],
  "hazards_construction": [
    "construction_site",
    "debris_or_rubble"
  ],
  "government_and_defense": [
    "military_facility",
    "space_facility",
  ],
  "natural": [
    "lake_or_pond"
  ],
  "power": [
    "electric_substation",
    "nuclear_powerplant",
    "oil_or_gas_facility",
    "solar_farm",
    "wind_farm"
  ],
  "public_transport": [
    "ground_transportation_station"
  ],
  "railway": [
    "railway_bridge"
  ],
  "shop": [
    "car_dealership",
    "shopping_mall"
  ],
  "waterway": [
    "dam",
    "port",
    "shipyard"
  ]
}


def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-all-classes", action="store_true", default=False)
    parser.add_argument("--max-num-obs", type=int, default=20)
    parser.add_argument("--orig-country-code", type=str, default='USA')
    parser.add_argument("--country-code", type=str, default='USA')
    parser.add_argument("--labelled-data-file", type=str, help="csv file with the necessary data elements for training model")
    args = parser.parse_args()
    return args

def main(args):
    args = parse_args(args)
    np.random.seed(args.seed)

    # Load the full dataset, and download it if necessary
    labels_df = pd.read_csv('data/fmow_v1.1/rgb_metadata.csv')
    labels_df = labels_df[labels_df.split != 'seq'] 
    full_idxs = np.arange(len(labels_df))
    labels_df['img_path'] = [f'data/fmow_v1.1/images/rgb_img_{idx}.png' for idx in full_idxs]
    labels_df = labels_df[labels_df.split == 'train']

    filtered_dfs = []
    if not args.use_all_classes:
      labels_df = labels_df[labels_df.country_code == args.country_code]
      print(CLASS_DICT.keys())
      for i, (cls_name, labels) in enumerate(CLASS_DICT.items()):
          print(labels)
          labels_i_df = labels_df[labels_df.category.isin(labels)]
          img_list = np.random.choice(labels_i_df.img_path, args.max_num_obs, replace=False)
          print(img_list)
          cls_df = pd.DataFrame({
              'y': [i] * img_list.size,
              'image_path': img_list
          })
          filtered_dfs.append(cls_df)
    else:
      orig_labels_df = labels_df[labels_df.country_code == args.orig_country_code]
      labels_df = labels_df[labels_df.country_code == args.country_code]
      uniq_classes = orig_labels_df.category.unique()
      for i, category in enumerate(uniq_classes):
        print(category)
        labels_i_df = labels_df[labels_df.category == category]
        if labels_i_df.shape[0] < args.max_num_obs:
          img_list = labels_i_df.img_path.to_numpy()
        else:
          img_list = np.random.choice(labels_i_df.img_path, args.max_num_obs, replace=False)
        print(img_list)
        cls_df = pd.DataFrame({
            'y': [i] * img_list.size,
            'category': [category] * img_list.size,
            'image_path': img_list
        })
        filtered_dfs.append(cls_df)
      print(filtered_dfs)
    filtered_dfs = pd.concat(filtered_dfs).reset_index(drop=True)
    filtered_dfs['y'] = filtered_dfs.y.astype(int)
        
    print(filtered_dfs.y)
    if args.labelled_data_file:
      filtered_dfs.to_csv(args.labelled_data_file, index=False)

if __name__ == "__main__":
    main(sys.argv[1:])
