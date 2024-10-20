"""
Script to train a model on LLM concepts with boosting similar to: https://arxiv.org/pdf/2310.19660
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
import pickle
import base64
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import re


sys.path.append(os.getcwd())
from src.llm.llm_api import LLMApi
from src.utils import convert_to_json
from src.llm.llm_local import LLMLocal
import src.common as common
from src.training_history import TrainingHistory
from scripts.train_bayesian import load_data_partition


def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-obs", type=int, default=50000)
    parser.add_argument("--in-dataset-file", type=str,
                        help="csv of the labelled training data")
    parser.add_argument("--keep-x-cols", type=str, nargs="*",
                        help="tabular columns to force keep")
    parser.add_argument("--indices-csv", type=str,
                        help="csv of training indices")
    parser.add_argument("--max-num-concepts", type=int, default=15)
    parser.add_argument("--log-file", type=str,
                        default="_output/log_train_boosting_concept.txt")
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--max-section-length", type=int, default=None)
    parser.add_argument("--text-summary-column", type=str,
                        default="llm_output", choices=['llm_output', 'sentence'])
    parser.add_argument("--boosting-prompt-file", type=str,
                        default="exp_multi_concept/prompts/boosting_iter.txt")
    parser.add_argument("--prompt-concepts-file", type=str,
                        default="exp_multi_concept/prompts/concept_questions.txt")
    parser.add_argument("--num-boost-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--in-training-history-file", type=str, default=None)
    parser.add_argument("--out-training-history-file", type=str, default=None)
    parser.add_argument("--out-extractions", type=str,
                        default="_output/boosting_extractions.pkl")
    parser.add_argument("--is-image", action="store_true", default=False)
    parser.add_argument("--num-iters", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=.0)
    parser.add_argument(
        "--llm-model-type",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        choices=[
                "gpt-4o-mini",
                "versa-gpt-4o-2024-05-13",
                "meta-llama/Meta-Llama-3.1-8B-Instruct", 
                "meta-llama/Meta-Llama-3.1-70B-Instruct", 
                "meta-llama/Llama-3.2-11B-Vision-Instruct" 
                ]
            )
    args = parser.parse_args()
    args.partition = "train"
    args.keep_x_cols = pd.Series(
        args.keep_x_cols) if args.keep_x_cols is not None else None
    return args


def create_sample_text(data_df, data_col):
    samples_text = ""
    for idx, row in data_df.iterrows():
        samples_text += f"""
        Text: {row[data_col]} 
        True label: {row['y']}
        """
    return samples_text


def create_sample_images(data_df, data_col):
    samples_text = ""
    payload = []
    for idx, (_, row) in enumerate(data_df.iterrows()):
        image_path = row[data_col]
        samples_text += f"Image {idx + 1}, True label: {row['y']}\n"
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        # img = mpimg.imread(image_path)
        # imgplot = plt.imshow(img)
        # plt.show()

        payload.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    return samples_text, payload


def generate_prompt(
        log_liks,
        args,
        concept_dicts,
        data_df,
        y_train,
        rejected_concept_dicts,
        num_samples=4,
        top_log_liks=50
):
    data_col = 'image_path' if args.is_image else 'sentence'
    with open(args.boosting_prompt_file, 'r') as file:
        prompt_template = file.read()

    print("===============", len(concept_dicts))

    existing_concepts = ""
    rejected_concepts = ""
    if len(concept_dicts):
        # TODO: change this to worst samples of differing classes?
        worst_log_liks = np.argsort(log_liks)[:top_log_liks]
        random_idxs = np.random.choice(
            worst_log_liks, size=num_samples, replace=False)
        samples_df = data_df.iloc[random_idxs]

        for concept_dict in concept_dicts:
            existing_concepts += f"- {concept_dict['concept']}\n"

        for concept_dict in rejected_concept_dicts:
            rejected_concepts += f"- {concept_dict['concept']}\n"
    else:
        num_per_class = max(1, int(num_samples / len(np.unique(y_train))))
        samples_df = data_df.groupby('y').apply(
            lambda x: x.sample(n=num_per_class)
        ).reset_index(drop=True)[[data_col, 'y']]

    prompt_template = prompt_template.replace(
        "{existing concepts}", existing_concepts)
    prompt_template = prompt_template.replace(
        "{rejected concepts}", rejected_concepts)

    if args.is_image:
        samples_text, image_payload = create_sample_images(
            samples_df, data_col)
        payload = [
            {
                "type": "text",
                "text": prompt_template.replace("{examples}", samples_text)
            }
        ] + image_payload
        return payload
    else:
        samples_text = create_sample_text(samples_df, data_col)
        prompt_template = prompt_template.replace("{examples}", samples_text)
        return prompt_template


def get_candidate_concept(llm_output) -> dict:
    try:
        logging.info("candidate concept summary =================")
        concept_dict = convert_to_json(llm_output)
        assert 'concept' in concept_dict
        logging.info("Candidate concept %s", concept_dict)
        return concept_dict
    except Exception as e:
        try:
            # search for just a question
            # Regular expression to find sentences that end with a question mark
            assert '?' in llm_output
            match = re.search(r'[^?.!]*\?', llm_output)
            return {'concept': match.group(0).strip()}
        except:
            logging.info("ERROR in extracting candidate concept %s", e)
            return None


def get_X_train(all_extracted_features, concept_dicts):
    if len(concept_dicts):
        return np.concatenate(
            [all_extracted_features[concept_dict["concept"]]
                for concept_dict in concept_dicts],
            axis=1
        )
    else:
        return None


def extract_features_and_train(llm, data_df, concept_dicts, all_extracted_features, args, history, y_train, force_keep_cols: pd.Series = None):
    all_extracted_features = common.extract_features_by_llm(
        llm,
        data_df,
        meta_concept_dicts=concept_dicts,
        all_extracted_features_dict=all_extracted_features,
        prompt_file=args.prompt_concepts_file,
        batch_size=args.batch_size,
        extraction_file=args.out_extractions,
        is_image=args.is_image,
        max_section_length=args.max_section_length,
    )

    X_train = common.get_features(
        concept_dicts, all_extracted_features, data_df, force_keep_columns=args.keep_x_cols)
    model_results = common.train_LR(X_train, y_train, penalty=None)

    return all_extracted_features, model_results


def main(args):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s",
                        filename=args.log_file, level=logging.INFO)
    logging.info(args)
    np.random.seed(args.seed)
    history = TrainingHistory(force_keep_cols=args.keep_x_cols)

    data_df = load_data_partition(args)

    y_train = data_df['y'].to_numpy().flatten()
    is_multiclass = np.unique(y_train).size > 2
    logging.info("y train prevalence %f", y_train.mean())
    logging.info("data_df shape %s", data_df.shape)

    all_extracted_features = {}
    if os.path.exists(args.out_extractions):
        with open(args.out_extractions, "rb") as f:
            all_extracted_features = pickle.load(f)

    concept_dicts = []
    if args.in_training_history_file is not None:
        history = history.load(args.in_training_history_file)
        print(history._concepts)
        concept_dicts = history.get_last_concepts()
        history.add_concepts(concept_dicts)

    if args.use_api:
        llm = LLMApi(args.seed, args.llm_model_type, logging)
    else:
        llm = LLMLocal(args.seed, args.llm_model_type, logging)

    concept_questions = [concept_dict["concept"]
                         for concept_dict in concept_dicts]
    logging.info("Initial concepts %s", concept_questions)

    iters = 0
    rejected_concept_dicts = []
    while len(concept_dicts) < args.max_num_concepts and iters < args.num_iters:
        llm_prompt = generate_prompt(
            history.get_last_log_liks(),
            args,
            concept_dicts,
            data_df,
            y_train,
            rejected_concept_dicts,
            num_samples=args.num_boost_samples,
        )
        print(llm_prompt)
        llm_response = llm.get_output(
            llm_prompt, max_new_tokens=2500, is_image=args.is_image)
        print(llm_response)
        new_concept_dict = get_candidate_concept(llm_response)
        proposal_concept_dicts = concept_dicts + [new_concept_dict]
        # breakpoint()
        all_extracted_features, model_results = extract_features_and_train(
            llm,
            data_df,
            proposal_concept_dicts,
            all_extracted_features,
            args,
            history,
            y_train,
            force_keep_cols=args.keep_x_cols,
        )
        log_liks = common.get_log_liks(
            y_train, model_results['y_pred'], is_multiclass=is_multiclass)

        # if the new concept increases model performance above a threshold keep it
        # TODO: check improvement in log lik instead of AUC?
        if len(concept_dicts) == 0 or (model_results["auc"] - history.get_last_auc() > args.threshold):
            concept_dicts = proposal_concept_dicts
            history.add_auc(model_results["auc"])
            history.add_model(model_results["model"])
            history.add_coef(model_results["coef"])
            history.add_intercept(model_results["intercept"])
            history.add_concepts(concept_dicts)
            history.add_log_liks(log_liks)
            logging.info("Iteration %d AUC %f", iters, model_results["auc"])
            logging.info("Iteration %d accepted proposed concept %s",
                         iters, new_concept_dict["concept"])
        else:
            rejected_concept_dicts.append(new_concept_dict)
            logging.info("Iteration %d rejected proposed concept %s",
                         iters, new_concept_dict["concept"])

        iters += 1
        history.save(args.out_training_history_file)
        concept_questions = [concept_dict["concept"]
                             for concept_dict in concept_dicts]
        logging.info("Iteration %d concepts %s", iters, str(concept_questions))


if __name__ == "__main__":
    main(sys.argv[1:])
