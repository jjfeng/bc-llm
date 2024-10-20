import os
import logging
import sys
import pickle
import asyncio

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import label_binarize

sys.path.append(os.getcwd())
from src.llm.dataset import TextDataset, ImageDataset
from src.utils import convert_to_json
from src.llm.llm import LLM

TABULAR_PREFIX = "Tabular feature: "


def query_and_parse_llm(llm, llm_prompt, extract_func, times_to_retry=4, default_response=None):
    llm_response = llm.get_output(llm_prompt, max_new_tokens=2500)
    print(llm_response)
    while times_to_retry > 0:
        try:
            llm_extraction = extract_func(llm_response)
            return llm_extraction
        except Exception:
            logging.info(
                "llm call failed. Retrying query. here is the response %s", llm_response)
            llm_response = llm.get_output(llm_prompt, max_new_tokens=2500)
            times_to_retry -= 1
    return default_response


def is_tabular(concept):
    return concept.startswith(TABULAR_PREFIX)


def get_features(concept_dicts, all_extracted_features, dset, force_keep_columns: pd.Series = None):
    concept_vals = []
    for concept_dict in concept_dicts:
        concept = concept_dict["concept"]
        concept_vals.append(all_extracted_features[concept])
    if force_keep_columns is not None:
        for col_name in force_keep_columns:
            concept_vals = [dset[col_name].to_numpy().reshape(
                (-1, 1))] + concept_vals
    return np.concatenate(concept_vals, axis=1)


def extract_features_by_llm(
        llm: LLM,
        dset_train,
        meta_concept_dicts: list[dict],
        all_extracted_features_dict: dict[str, np.ndarray],
        prompt_file,
        extraction_file: str,
        batch_size=1,
        max_new_tokens=150,
        is_image=False,
        max_retries: int = 2,
        max_num_concepts_extract: int = 11,
        max_section_length: int = None
) -> dict[str, np.ndarray]:
    logging.info("LLM %s", llm.model_type)
    # Determine which new concepts to extract
    prior_concepts = set(all_extracted_features_dict.keys())
    concepts_to_extract = []
    concepts_dicts_to_extract = []
    for concept_dict in meta_concept_dicts:
        concept = concept_dict["concept"]
        if not is_tabular(concept) and concept not in prior_concepts and concept not in concepts_to_extract:
            concepts_to_extract.append(concept)
            concepts_dicts_to_extract.append(concept_dict)

    if concepts_dicts_to_extract:
        batched_concepts = np.array_split(concepts_dicts_to_extract, max(
            1, len(concepts_dicts_to_extract)//max_num_concepts_extract))
        for concept_dicts_batch in batched_concepts:
            all_extracted_features_dict = _extract_features_by_llm_batch(
                llm,
                dset_train,
                concept_dicts_batch,
                all_extracted_features_dict,
                prompt_file,
                extraction_file,
                batch_size,
                max_new_tokens,
                is_image,
                max_retries,
                max_section_length
            )
    return all_extracted_features_dict


def split_sentences_by_id(
        dset_train: pd.DataFrame,
        max_section_length: int = None
) -> (np.array, np.array):
    """
    If texts are too long, split it into sections and return group_id as well as the sections
    """
    if max_section_length is None:
        return np.arange(dset_train.shape[0]), dset_train.sentence.to_numpy()
    else:
        sentences = []
        ids = []
        for idx, row in dset_train.iterrows():
            sentence = row.sentence
            if len(sentence) > max_section_length:
                for start in range(0, len(sentence), max_section_length):
                    end = min(start + max_section_length, len(sentence))
                    ids.append(idx)
                    sentences.append(sentence[start:end])
            else:
                sentences.append(sentence)
                ids.append(idx)
        return np.array(ids), np.array(sentences)


def _extract_features_by_llm_batch(
        llm,
        dset_train,
        meta_concept_dicts: list[dict],
        all_extracted_features_dict: dict[str, np.ndarray],
        prompt_file,
        extraction_file: str,
        batch_size=1,
        max_new_tokens=150,
        is_image=False,
        max_retries: int = 3,
        max_section_length: int = None
) -> dict[str, np.ndarray]:
    # Determine which new concepts to extract
    concepts_in_meta = [concept_dict["concept"]
                        for concept_dict in meta_concept_dicts if not is_tabular(concept_dict["concept"])]
    concepts_to_extract = list(
        set(concepts_in_meta) - set(all_extracted_features_dict.keys()))

    num_to_extract = len(concepts_to_extract)
    if num_to_extract == 0:
        return all_extracted_features_dict

    # Load prompt for extracting concepts
    prompt_file = os.path.abspath(os.path.join(os.getcwd(), prompt_file))
    with open(prompt_file, 'r') as file:
        prompt_template = file.read()

    prompt_questions = ""
    for idx, concept in enumerate(concepts_to_extract):
        prompt_questions += f"{idx} - {concept}" + "\n"
    prompt_template = prompt_template.replace(
        "{prompt_questions}", prompt_questions)

    if is_image:
        group_ids = np.arange(dset_train.shape[0])
        dataset = ImageDataset(dset_train.image_path.tolist(), prompt_template)
    else:
        group_ids, sentences = split_sentences_by_id(
            dset_train,
            max_section_length
        )
        dataset = TextDataset(
            sentences,
            prompt_template,
            text_to_replace="{sentence}"
        )

    if llm.is_api:
        llm_outputs = asyncio.run(
            llm.get_outputs(
                dataset,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                is_image=is_image,
                temperature=0.5,
                validation_func=lambda x: [_extract_features_json(
                    elem, logging, num_concepts=num_to_extract) for elem in x],
                max_retries=max_retries,
            )
        )
    else:
        llm_outputs = llm.get_outputs(
            dataset,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            is_image=is_image,
            temperature=0.5,
            validation_func=lambda x: [_extract_features_json(
                elem, logging, num_concepts=num_to_extract) for elem in x]
        )
    extracted_llm_outputs = _collate_extractions_by_group(
        llm_outputs, group_ids, num_to_extract)

    # Parse the json from the LLM
    new_extracted_feature_dict = {}
    for idx, concept in enumerate(concepts_to_extract):
        extracted_features = extracted_llm_outputs[:, idx:idx + 1]
        logging.info("concept %s, prevalence %f",
                     concept, np.mean(extracted_features))
        new_extracted_feature_dict[concept] = extracted_features

    # Save down the new extractions and the old ones
    all_extracted_features_dict = all_extracted_features_dict | new_extracted_feature_dict

    directory = os.path.dirname(extraction_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(extraction_file, "wb") as f:
        pickle.dump(all_extracted_features_dict, f)
    return all_extracted_features_dict


def _collate_extractions_by_group(llm_outputs, group_ids, num_to_extract: int = 6) -> np.ndarray:
    """
    Combines the extracted feature vectors if they have the same group ids
    """
    extracted_llm_outputs = []
    for grp_id in np.unique(group_ids):
        match_idxs = np.where(group_ids == grp_id)[0]
        id_llm_outputs = np.concatenate(
            [
                np.array(_extract_features_json(
                    llm_outputs[match_idx], logging, num_concepts=num_to_extract)).reshape((1, -1))
                for match_idx in match_idxs
            ],
            axis=0
        )

        assert id_llm_outputs.shape == (len(match_idxs), num_to_extract)
        concepts_for_id = np.max(id_llm_outputs, axis=0)
        assert concepts_for_id.shape == (num_to_extract,)
        extracted_llm_outputs.append(concepts_for_id)
    extracted_llm_outputs = np.vstack(extracted_llm_outputs)
    return extracted_llm_outputs


def _extract_features_json(llm_output, logging, num_concepts):
    json_response = convert_to_json(llm_output, logging)
    features_output = []
    for idx in range(num_concepts):
        try:
            features_output.append(json_response[str(idx)])
        except:
            features_output.append(0)
    return features_output


def get_auc_and_probs(mdl, X, y) -> (float, list):
    is_multi_class = len(np.unique(y)) > 2
    if is_multi_class:
        classes = np.unique(y)
        y_pred = mdl.predict_proba(X)
        y = label_binarize(y, classes=classes)
        auc = roc_auc_score(y, y_pred, multi_class="ovr")
    else:
        y_pred = mdl.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_pred)

    return auc, y_pred

# to avoid numerical underflow/overflow


def get_safe_prob(y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return y_pred


def get_safe_logit(y_pred, eps=1e-15):
    y_pred = get_safe_prob(y_pred, eps)
    return np.log(y_pred/(1 - y_pred))


def get_log_liks(y, pred_prob, is_multiclass=False):
    safe_pred_prob = get_safe_prob(pred_prob)
    if not is_multiclass:
        # binary
        return y * np.log(safe_pred_prob) + (1 - y) * np.log(1 - safe_pred_prob)
    else:
        # multiclass
        return np.log(safe_pred_prob[np.arange(safe_pred_prob.shape[0]), y])


def train_LR(X_train, y_train, penalty=None) -> dict:
    _, class_counts = np.unique(y_train, return_counts=True)
    cv = 2 if np.any(class_counts < 3) else 3
    is_multi_class = len(np.unique(y_train)) > 2
    args = {"cv": cv,
            "scoring": "roc_auc_ovr" if is_multi_class else "roc_auc",
            "n_jobs": -1,
            "multi_class": "multinomial" if is_multi_class else "auto",
            }

    if penalty is None:
        model = LogisticRegression(penalty=None)
    elif penalty == 'l2':
        model = LogisticRegressionCV(
            max_iter=10000, Cs=20, solver="lbfgs", **args)
    elif penalty == 'l1':
        solver = "saga" if is_multi_class else "liblinear"
        model = LogisticRegressionCV(
            max_iter=6000, penalty='l1', solver=solver, Cs=20, **args)
    elif penalty == 'elasticnet':
        model = LogisticRegressionCV(
            max_iter=10000,
            penalty='elasticnet',
            solver='saga',
            Cs=20,
            l1_ratios=[0, 0.1, 0.5, 0.8, 1],
            **args
        )
    model.fit(X_train, y_train)
    if penalty is not None:
        print("SHAPE", X_train.shape, y_train.shape)
        logging.info("CV SCORES %s", model.scores_)
        logging.info("CV SCORES Cs %s", model.Cs_)
        logging.info("CV SCORES best C %s", model.C_)
        logging.info("CV SCORES L1 ratios %s", model.l1_ratios_)
        logging.info("CV SCORES coefs_paths_ %s", model.coefs_paths_[1][0])
        logging.info("CV SCORES coefs_paths_ %s",
                     model.coefs_paths_[1][0].shape)
    logging.info("NUM NONZERO %d", np.sum(model.coef_ != 0))
    if penalty == "elasticnet":
        print("L1 ratio", model.l1_ratio_)
        logging.info("L1 ratio %f", model.l1_ratio_)
    train_auc, y_pred = get_auc_and_probs(model, X_train, y_train)
    train_logit = get_safe_logit(y_pred)
    return {
        "model": model,
        "auc": train_auc,
        "logit": train_logit,
        "coef": model.coef_,
        "intercept": model.intercept_,
        "y_pred": y_pred
    }
