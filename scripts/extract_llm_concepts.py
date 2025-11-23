"""
Script for extracting llm summary
"""


import os
import asyncio
import sys
import json
import logging
import argparse
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import scipy
from typing import List

import transformers
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoTokenizer

sys.path.append(os.getcwd())

from src.utils import convert_to_json
from src.llm_response_types import ProtoConceptExtract
import src.common as common

sys.path.append('llm-api-main')
from lab_llm.constants import LLMModel, OpenAi, convert_to_llm_type
from lab_llm.llm_api import LLMApi
from lab_llm.llm_cache import LLMCache
from lab_llm.duckdb_handler import DuckDBHandler
from lab_llm.error_callback_handler import ErrorCallbackHandler
from lab_llm.dataset import TextDataset, ImageDataset


def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-file", type=str, default='cache.db')
    parser.add_argument("--config-file", type=str,
                        help="file with json config for replacing strings in template")
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
    parser.add_argument("--batch-size", type=int, default=4, help="number of llm queries to run in a batch")
    parser.add_argument("--batch-obs-size", type=int, default=1, help="number of observations to batch together annotation for")
    parser.add_argument("--num-new-tokens", type=int, default=300,
                        help="the number of new tokens to generate")
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--max-section-length", type=int, default=None)
    parser.add_argument(
        "--llm-model-type",
        type=str,
        )
    args = parser.parse_args()
    assert args.use_api
    return args


def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s",
                        filename=args.log_file, level=logging.INFO)
    logging.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_df = pd.read_csv(args.in_dataset_file, index_col=0, header=0)
    data_df['llm_output'] = ""
    data_df['llm_reasoning'] = ""

    missing_idxs = np.arange(data_df.shape[0])
    logging.info("missing raw %s", missing_idxs.size)
    if args.indices_file is not None:
        indices_df = pd.read_csv(args.indices_file, header=0)
        train_idxs = indices_df[indices_df.partition == "train"].idx.to_numpy()
        missing_idxs = np.intersect1d(missing_idxs, train_idxs)
    logging.info("missing final %s %d", missing_idxs[:10], missing_idxs.size)
    print("missing", missing_idxs.size)

    new_data_df = data_df.iloc[missing_idxs]
    
    cur_dir = os.getcwd()
    prompt_file = os.path.abspath(os.path.join(cur_dir, args.prompt_file))
    with open(prompt_file, 'r') as file:
        prompt_template = file.read()

    if args.config_file:
        with open(args.config_file, "r") as f:
            config_dict = json.load(f)
        for k, v in config_dict.items():
            prompt_template = prompt_template.replace(k, v)
    
    # Setup LLM cache + API
    logger = logging.getLogger(__name__)
    load_dotenv()
    args.cache = LLMCache(DuckDBHandler(args.cache_file))
    args.llm_model_type = convert_to_llm_type(args.llm_model_type)
    llm = LLMApi(args.cache, seed=10, model_type=args.llm_model_type, error_handler=ErrorCallbackHandler(logger), logging=logging)

    if args.batch_obs_size > 1:
        # Get outputs grouped
        obs_group_idxs = []
        if not is_image:
            raise NotImplementedError("havent implemented grouped extractions for text data yet")
        else:
            group_ids = np.arange(dset_train.shape[0])
            image_paths_list = []
            for i in range(0, dset_train.shape[0], group_size):
                group_df = dset_train.image_path.iloc[i:i + group_size]
                image_paths_list.append(group_df.tolist())
                obs_group_idxs.append([j for j in range(i, i + group_df.shape[0])])

            dataset = ImageGroupDataset(
                image_paths_list,
                prompt_template
            )

        grouped_llm_outputs = asyncio.run(llm.get_outputs(
            dataset,
            max_new_tokens=args.num_new_tokens,
            batch_size=args.batch_size,
            is_image=args.is_image,
            response_model=ProtoConceptExtractGrouped,
        ))

        # ungroup responses
        llm_output_strs = [""] * len(missing_idxs)
        for llm_output_group, idxs in zip(grouped_llm_outputs, obs_group_idxs):
            if llm_output_group is not None:
                for idx, keyphrases in enumerate(llm_output.keyphrases[:len(idxs)]):
                    llm_output_strs[idxs[j]] = keyphrases
            else:
                logging.info(f"warning: llm output was missing for idxs {idxs}")
    else:
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
            prompts = [prompt_template.replace("{note}", s) for s in sentences]
            dataset = TextDataset(
                prompts,
            )

        llm_outputs = asyncio.run(llm.get_outputs(
            dataset,
            max_new_tokens=args.num_new_tokens,
            batch_size=args.batch_size,
            is_image=args.is_image,
            response_model=ProtoConceptExtract,
        ))

        llm_output_strs = [""] * len(missing_idxs)
        llm_reasoning_strs = [""] * len(missing_idxs)
        for grp_id in np.unique(group_ids):
            match_idxs = np.where(group_ids == grp_id)[0]
            grp_llm_output = ','.join(
                [",".join(llm_outputs[match_idx].keyphrases) for match_idx in match_idxs]
            )
            llm_output_strs[grp_id] = grp_llm_output
            llm_reasoning_strs[grp_id] = llm_outputs[match_idxs[0]].reasoning
    
    data_df["llm_output"].iloc[missing_idxs] = llm_output_strs
    data_df["llm_reasoning"].iloc[missing_idxs] = llm_reasoning_strs
    print("not na", (data_df.llm_output != "").sum())
    data_df.to_csv(args.llm_outputs_file, index=False)

if __name__ == "__main__":
    main(sys.argv[1:])
