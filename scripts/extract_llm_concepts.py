"""
Script for extracting llm summary
"""


import os
import asyncio
import sys
import logging
import argparse
import torch
import numpy as np
import pandas as pd
import scipy

import transformers
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoTokenizer
from datasets import Dataset

sys.path.append(os.getcwd())

from src.utils import convert_to_json
from src.llm.llm_api import LLMApi
from src.llm.llm_local import LLMLocal
from src.llm.dataset import TextDataset, ImageDataset
import src.common as common


def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-file", type=str,
                        help="file with prompt for extracting concepts")
    parser.add_argument("--in-dataset-file", type=str,
                        help="csv of the data we want to learn concepts for")
    parser.add_argument("--indices-file", type=str,
                        help="csv of training indices")
    parser.add_argument("--is-image", action="store_true", default=False)
    parser.add_argument("--llm-outputs-file", type=str,
                        help="csv file with llm concepts")
    parser.add_argument("--log-file", type=str,
                        default="_output/log_extract.txt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-new-tokens", type=int, default=300,
                        help="the number of new tokens to generate")
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--max-section-length", type=int, default=None)
    parser.add_argument(
        "--llm-model-type",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        choices=[
                "versa-gpt-4o-2024-05-13",
                "gpt-4o-mini",
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "meta-llama/Meta-Llama-3.1-70B-Instruct",
                "meta-llama/Llama-3.2-11B-Vision-Instruct" 
                ]
            )
    args = parser.parse_args()
    return args


def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s",
                        filename=args.log_file, level=logging.INFO)
    logging.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if os.path.exists(args.llm_outputs_file):
        data_df = pd.read_csv(args.llm_outputs_file, header=0)
    else:
        data_df = pd.read_csv(args.in_dataset_file, index_col=0, header=0)
        data_df['llm_output'] = [""] * len(data_df)

    concepts_is_missing = (
        data_df['llm_output'].str.len() == 0) | data_df.llm_output.isna()
    missing_idxs = np.where(concepts_is_missing)[0]
    logging.info("missing raw %s", missing_idxs.size)
    if args.indices_file is not None:
        indices_df = pd.read_csv(args.indices_file, header=0)
        print(indices_df)
        train_idxs = indices_df[indices_df.partition == "train"].idx.to_numpy()
        missing_idxs = np.intersect1d(missing_idxs, train_idxs)
        print("inTERSECTIOn", missing_idxs)
    logging.info("missing final %s %d", missing_idxs[:10], missing_idxs.size)
    print("missing", missing_idxs.size)
    if missing_idxs.size == 0:
        return

    new_data_df = data_df.iloc[missing_idxs]

    cur_dir = os.getcwd()
    prompt_file = os.path.abspath(os.path.join(cur_dir, args.prompt_file))
    with open(prompt_file, 'r') as file:
        prompt_template = file.read()

    if args.is_image:
        group_ids = np.arange(new_data_df.shape[0])
        dataset = ImageDataset(
            new_data_df.image_path.tolist(),
            prompt_template
        )
    else:
        group_ids, sentences = common.split_sentences_by_id(
            new_data_df,
            args.max_section_length
        )
        print("GROUP IDS", group_ids, len(new_data_df), len(group_ids))

        dataset = TextDataset(
            sentences,
            prompt_template,
            text_to_replace="{note}",
        )

    def write_llm_outputs(llm_outputs):
        tot_num_llm_outs = len(llm_outputs)
        grp_llm_output_list = []
        for grp_id in np.unique(group_ids[:tot_num_llm_outs]):
            match_idxs = np.where(group_ids == grp_id)[0]
            try:
                grp_llm_output = ','.join([",".join(list(convert_to_json(llm_outputs[match_idx]).values(
                ))) for match_idx in match_idxs if match_idx < tot_num_llm_outs])

            except Exception as e:
                # print(e)
                # breakpoint()
                grp_llm_output = ''
            grp_llm_output_list.append(grp_llm_output)

        fill_in_idxs = missing_idxs[:len(grp_llm_output_list)]
        data_df["llm_output"].iloc[fill_in_idxs] = grp_llm_output_list
        data_df.to_csv(args.llm_outputs_file, index=False)

    if args.use_api:
        llm = LLMApi(args.seed, args.llm_model_type, logging)
        llm_outputs = asyncio.run(llm.get_outputs(
            dataset,
            max_new_tokens=args.num_new_tokens,
            batch_size=args.batch_size,
            is_image=args.is_image,
            validation_func=lambda x: [
                convert_to_json(elem, logging) for elem in x],
            callback=write_llm_outputs
        ))
    else:
        llm = LLMLocal(args.seed, args.llm_model_type, logging)
        llm_outputs = llm.get_outputs(
            dataset,
            max_new_tokens=args.num_new_tokens,
            batch_size=args.batch_size,
            is_image=args.is_image,
            validation_func=lambda x: [
                convert_to_json(elem, logging) for elem in x],
            callback=write_llm_outputs
        )


if __name__ == "__main__":
    main(sys.argv[1:])
