import os
import logging
import sys
import pickle
import asyncio

from typing import List
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.preprocessing import label_binarize

sys.path.append(os.getcwd())
from src.logistic import LogisticRegressionTorch
from src.utils import convert_to_json
from src.llm_response_types import ExtractResponseList, GroupedExtractResponses

sys.path.append('llm-api-main')
from lab_llm.llm_api import LLMApi
from lab_llm.dataset import TextDataset, ImageDataset, ImageGroupDataset

TABULAR_PREFIX = "Tabular feature: "


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


def _collate_extractions_by_group(llm_outputs: List, group_ids, num_to_extract: int = 6) -> np.ndarray:
    """
    Combines the extracted feature vectors if they have the same group ids
    @param num_to_extract: this is the number of concepts in a batch, assumes you are extracting the same number of features per concept-batch
    """
    extracted_llm_outputs = []
    for grp_id in np.unique(group_ids):
        match_idxs = np.where(group_ids == grp_id)[0]
        id_llm_outputs = np.concatenate(
            [
                _extract_features(llm_outputs[match_idx], num_concepts=num_to_extract).reshape((1, -1))
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


def _extract_features(llm_output: ExtractResponseList, num_concepts: int):
    features_output = [0] * num_concepts
    if llm_output is not None:
        for extraction_resp in llm_output.extractions:
            if extraction_resp.question <= num_concepts:
                features_output[(extraction_resp.question - 1)] = extraction_resp.answer
            else:
                logging.warn("question ID does not exist... %d %s", extraction_resp.question, extraction_resp.reasoning)
    return np.array(features_output)


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

def train_LR(X_train, y_train, penalty=None, use_acc:bool=False, seed:int = 0) -> dict:
    _, class_counts = np.unique(y_train, return_counts=True)
    cv = 2 if np.any(class_counts < 3) else 5
    is_multi_class = len(np.unique(y_train)) > 2
    args = {
        "cv": StratifiedKFold(n_splits=cv),
        "scoring": "accuracy" if use_acc else ("roc_auc_ovr" if is_multi_class else "roc_auc"),
        "n_jobs": -1,
    }
    
    if penalty is None:
        final_model = LogisticRegression(penalty=None)
        final_model.fit(X_train, y_train)
    elif penalty == "l1":
        param_grid = {
            'lambd': [0.01,0.001,0.0001, 1e-5],
            'num_epochs': [7000]
        }
        model = GridSearchCV(estimator=LogisticRegressionTorch(seed=seed), param_grid=param_grid, **args)
        model.fit(X_train, y_train)
        print("CV SCORES ACC/AUC", model.best_score_)
        logging.info("CV SCORES best_score_ %s", model.best_score_)
        logging.info("CV RESULTS %s", model.cv_results_)
        final_model = model.best_estimator_
        logging.info("NUM NONZERO 1e-2 %d", np.sum(~np.isclose(np.abs(final_model.coef_).max(axis=0), 0, atol=1e-2)))
    elif penalty == "l1_sklearn":
        final_model = LogisticRegressionCV(
            max_iter=10000,
            Cs=10,
            solver="saga",
            penalty="l1",
            **args)
        final_model.fit(X_train, y_train)
    elif penalty == "l2":
        final_model = LogisticRegressionCV(
            max_iter=10000,
            Cs=10,
            solver="lbfgs",
            **args)
        final_model.fit(X_train, y_train)
    
    train_auc, y_pred = get_auc_and_probs(final_model, X_train, y_train)
    y_assigned_class = final_model.predict(X_train)
    train_logit = get_safe_logit(y_pred)
    return {
        "model": final_model,
        "auc": train_auc, # train auc
        "logit": train_logit,
        "coef": final_model.coef_,
        "intercept": final_model.intercept_,
        "y_pred": y_pred,
        "y_assigned_class": y_assigned_class,
        "acc": accuracy_score(y_train, y_assigned_class), # train acc
        "f1": f1_score(y_train, y_assigned_class, average="macro"), # train acc
    }

def extract_features_by_llm_grouped(
        llm: LLMApi,
        dset_train,
        meta_concept_dicts: list[dict],
        prompt_file,
        all_extracted_features_dict: dict = dict(),
        extraction_file: str = None,
        batch_size=1,
        batch_concept_size = 20,
        max_new_tokens=5000,
        is_image=False,
        group_size:int=1,
        max_retries: int = 1,
        max_section_length: int = None
) -> dict[str, np.ndarray]:
    prior_concepts = set(all_extracted_features_dict.keys())
    print("all_extracted_features_dict", prior_concepts)
    concepts_to_extract = []
    for concept_dict in meta_concept_dicts:
        concept = concept_dict["concept"]
        if not is_tabular(concept) and concept not in concepts_to_extract and concept not in prior_concepts:
            concepts_to_extract.append(concept)
    
    print("LEN CONCEPTS", len(concepts_to_extract))
    
    print("dset_train", dset_train.shape)
    for i in range(0, len(concepts_to_extract), batch_concept_size):
        # Get the batch of concepts to annotate
        batch_concepts_to_extract = concepts_to_extract[i:i + batch_concept_size]

        # Load prompt for extracting concepts
        prompt_file = os.path.abspath(os.path.join(os.getcwd(), prompt_file))
        with open(prompt_file, 'r') as file:
            prompt_template = file.read()

        # Fill in concept questions
        prompt_questions = ""
        for idx, concept in enumerate(batch_concepts_to_extract):
            prompt_questions += f"{idx + 1} - {concept}" + "\n"
        prompt_template = prompt_template.replace(
            "{prompt_questions}", prompt_questions)
        print(prompt_template)

        print("dset_train.shape[0]", dset_train.shape[0])
        print("group_size", group_size)
        if group_size > 1:
            # group together observations for extractions
            obs_group_idxs = []
            if is_image:
                group_ids = np.arange(dset_train.shape[0])
                image_paths_list = []
                for i in range(0, dset_train.shape[0], group_size):
                    group_df = dset_train.image_path.iloc[i:i + group_size]
                    image_paths_list.append(group_df.tolist())
                    obs_group_idxs.append([j for j in range(i, i + group_df.shape[0])])

                dataset = ImageGroupDataset(image_paths_list, prompt_template=prompt_template)
                print("dataset1", len(dataset))
            else:
                raise NotImplementedError("have not yet implemented grouped-extraction of text data")

            grouped_llm_outputs = asyncio.run(
                llm.get_outputs(
                    dataset,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    is_image=is_image,
                    temperature=0,
                    max_retries=max_retries,
                    response_model=GroupedExtractResponses,
                )
            )
            print("grouped_llm_outputs", len(grouped_llm_outputs), dset_train.shape[0])
            # ungroup responses
            llm_outputs = [None] * dset_train.shape[0]
            for group_obs_idxs, grouped_output in zip(obs_group_idxs, grouped_llm_outputs):
                if grouped_output is not None:
                    for j, extraction in enumerate(grouped_output.all_extractions[:len(group_obs_idxs)]):
                        llm_outputs[group_obs_idxs[j]] = extraction
                else:
                    logging.info(f"warning: llm output was missing for idxs {group_obs_idxs}")
            print("llm_outputs llm_outputs", len(llm_outputs), dset_train.shape[0])
        else:
            if is_image:
                group_ids = np.arange(dset_train.shape[0])
                dataset = ImageDataset(dset_train.image_path.tolist(), prompt_template)
                print("dataset2", len(dataset))
            else:
                group_ids, sentences = split_sentences_by_id(
                    dset_train,
                    max_section_length
                )
                prompts = [prompt_template.replace("{sentence}", s) for s in sentences]
                dataset = TextDataset(
                    prompts,
                )
            llm_outputs = asyncio.run(
                llm.get_outputs(
                    dataset,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    is_image=is_image,
                    temperature=0,
                    max_retries=max_retries,
                    response_model=ExtractResponseList,
                )
            )

            # for llm_out, img_path in zip(llm_outputs, dset_train.image_path.tolist()):
            #     logging.info(img_path)
            #     logging.info(llm_out)

        # extract responses
        extracted_llm_outputs = _collate_extractions_by_group(
            llm_outputs, group_ids, len(batch_concepts_to_extract))

        # fill in dictionary with extractions        
        for idx, concept in enumerate(batch_concepts_to_extract):
            extracted_features = extracted_llm_outputs[:, idx:idx + 1]
            logging.info("concept %s, prevalence %f",
                        concept, np.mean(extracted_features))
            all_extracted_features_dict[concept] = extracted_features

        if extraction_file is not None:
            with open(extraction_file, "wb") as f:
                pickle.dump(all_extracted_features_dict, f)

    return all_extracted_features_dict