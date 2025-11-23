"""
Script for assembling CUB data
"""
import os
import sys
import logging
import argparse
import pickle
import pandas as pd
import numpy as np

def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-folder", type=str, help="folder of CUB data")
    parser.add_argument(
        "--attributes",
        nargs="*",
        type=str,
        help="the columns in the data to use for making the labels",
        default=[]) #"has_bill_shape::dagger", "has_wing_color::blue", "has_wing_shape::rounded-wings", "has_wing_pattern::solid"])
    parser.add_argument("--out-extractions", type=str, default="_output/note_extractions.pkl")
    parser.add_argument("--keep-classes", nargs="*", type=int)
    parser.add_argument("--log-file", type=str, help="log file")
    parser.add_argument("--max-obs", type=int, default=-1)
    parser.add_argument("--labelled-data-file", type=str, help="csv file with the necessary data elements for training model")
    args = parser.parse_args()
    args.attributes_file = os.path.join(args.dataset_folder, "attributes.txt")
    args.image_attribute_file = os.path.join(args.dataset_folder, "CUB_200_2011/attributes/image_attribute_labels_clean.txt")
    args.images_file = os.path.join(args.dataset_folder, "CUB_200_2011/images.txt") 
    args.image_labels_file = os.path.join(args.dataset_folder, "CUB_200_2011/image_class_labels.txt") 
    args.image_folder = os.path.join(args.dataset_folder, "CUB_200_2011/images") 
    return args

def assemble_images(image_labels_file, images_file, attributes_file, image_folder, image_attribute_file):
    labels_df = pd.read_csv(image_labels_file, delim_whitespace=True, index_col=None, header=None, names=["image_idx", "orig_bird_label"])
    images_df = pd.read_csv(images_file, delim_whitespace=True, index_col=None, header=None, names=["image_idx", "image_path"])
    images_df['image_path'] = image_folder + '/' + images_df.image_path
    images_df = images_df.merge(labels_df, on="image_idx")
    attributes_df = pd.read_csv(attributes_file, delim_whitespace=True, header=None, names=["attribute_idx", "attribute"])
    image_attr_df = pd.read_csv(
        image_attribute_file,
        delim_whitespace=True,
        index_col=None,
        header=None,
        names=["image_idx", "attribute_idx", "attribute_value", "certainty", "time"],
        on_bad_lines="warn")
    
    image_attr_df = image_attr_df[image_attr_df.attribute_idx.isin(attributes_df.attribute_idx)]
    image_attr_df = image_attr_df.pivot(index='image_idx', columns='attribute_idx', values='attribute_value')
    image_attr_df = image_attr_df.fillna(0)
    image_attr_df.columns = attributes_df.attribute.to_list()
    images_df = images_df.merge(image_attr_df, on="image_idx")
    return images_df

def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    np.random.seed(args.seed)
    print(args)

    labels_df = pd.read_csv(args.image_labels_file, delim_whitespace=True, index_col=None, header=None, names=["image_idx", "orig_bird_label"])
    images_df = pd.read_csv(args.images_file, delim_whitespace=True, index_col=None, header=None, names=["image_idx", "image_path"])
    images_df['image_path'] = args.image_folder + '/' + images_df.image_path
    images_df = images_df.merge(labels_df, on="image_idx")
    attributes_df = pd.read_csv(args.attributes_file, delim_whitespace=True, header=None, names=["attribute_idx", "attribute"])
    image_attr_df = pd.read_csv(
        args.image_attribute_file,
        delim_whitespace=True,
        index_col=None,
        header=None,
        names=["image_idx", "attribute_idx", "attribute_value", "certainty", "time"],
        on_bad_lines="warn")
    
    image_attr_df = image_attr_df[image_attr_df.attribute_idx.isin(attributes_df.attribute_idx)]
    image_attr_df = image_attr_df.pivot(index='image_idx', columns='attribute_idx', values='attribute_value')
    image_attr_df = image_attr_df.fillna(0)
    image_attr_df.columns = image_attr_df.columns[:-len(args.attributes)].to_list() + attributes_df.attribute.to_list()
    images_df = images_df.merge(image_attr_df, on="image_idx")
    print(images_df)
    

    if args.keep_classes:
        images_df = images_df[images_df.orig_bird_label.isin(args.keep_classes)]
        if len(args.keep_classes) == 2:
            images_df['y'] = (images_df.orig_bird_label == args.keep_classes[1]).astype(int)
        else:
            images_df['y'] = images_df.orig_bird_label.map({label: idx for idx, label in enumerate(args.keep_classes)})

    if args.max_obs > 0 and args.max_obs < len(images_df):
        images_df = images_df.sample(args.max_obs, replace=False).reset_index()
    
    images_df.to_csv(args.labelled_data_file, index=True)

if __name__ == "__main__":
    main(sys.argv[1:])
