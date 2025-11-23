import time
import re
import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import asyncio
import scipy
from typing import List, Optional

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

sys.path.append('llm-api-main')
from lab_llm.llm_api import LLMApi
from src.utils import convert_to_json, convert_to_list_of_jsons
from src.training_history import TrainingHistory
import src.common as common
from src.llm_response_types import PriorResponse, CandidateConcepts


class ConceptLearnerModel:
    """
    Class for our (Bayesian) concept learner

    Can also learn concepts greedily
    """
    default_prior = 0.1
    penalty_downweight_factor = 1000
    # Number of observations to show each iter
    num_show_obs = 3
    # number of candidate concepts to test each iter
    keep_num_candidates = 20
    # max candidate concepts to consider per iter
    max_candidates_to_consider = 10
    def __init__(
            self,
            init_history: TrainingHistory,
            llm_iter: LLMApi,
            llm_extraction: LLMApi,
            num_classes: int,
            num_meta_concepts: int,
            prompt_iter_type: str,
            prompt_iter_file: str,
            config: dict,
            prompt_concepts_file: str,
            prompt_prior_file: str,
            out_extractions_file: str,
            residual_model_type: str,
            final_learner_type: str,
            inverse_penalty_param: float, # inverse regularization for the l2 penalty for logistic regression
            train_frac: float = 0.5,
            num_greedy_epochs: int = 0,
            max_epochs: int = 10,
            batch_size: int = 4,
            batch_concept_size: int = 20,
            batch_obs_size: int = 1,
            num_greedy_holdout: int = 1,
            do_greedy: bool = False,
            is_image: bool = False,
            all_extracted_features_dict = {},
            max_section_length: int = None,
            force_keep_columns: pd.Series = None,
            max_new_tokens: int = 5000,
            num_top: int = 40,
            num_minibatch: int = None,
            is_greedy_metric_acc: bool = False
            ):
        self.init_history = init_history
        self.llm_iter = llm_iter
        self.llm_extraction = llm_extraction
        self.num_classes = num_classes
        self.is_multiclass = num_classes > 2
        self.num_meta_concepts = num_meta_concepts
        self.prompt_iter_type = prompt_iter_type
        self.num_greedy_holdout = num_greedy_holdout
        assert self.prompt_iter_type == "conditional"
        self.config = config
        self.prompt_iter_file = prompt_iter_file
        self.prompt_concepts_file = prompt_concepts_file
        self.prompt_prior_file = prompt_prior_file
        self.out_extractions_file = out_extractions_file
        self.residual_model_type = residual_model_type
        self.inverse_penalty_param = inverse_penalty_param
        self.train_frac = train_frac
        self.num_greedy_epochs = num_greedy_epochs
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.batch_obs_size = batch_obs_size
        self.all_extracted_features_dict = all_extracted_features_dict
        self.is_image = is_image
        self.do_greedy = do_greedy
        self.is_greedy_metric_acc = is_greedy_metric_acc
        self.max_section_length = max_section_length
        self.force_keep_columns = force_keep_columns
        self.max_new_tokens = max_new_tokens
        self.num_minibatch = num_minibatch
        self.num_top = num_top
        self.batch_concept_size = batch_concept_size
        self.final_learner_type = final_learner_type

        assert self.num_minibatch is None
        assert self.num_meta_concepts % self.num_greedy_holdout == 0
    
    def fit(self,
            data_df,
            X_features,
            feat_names,
            training_history_file: str,
            aucs_plot_file: str):
        history = self.init_history
        # initialize concepts
        meta_concept_dicts = history.get_last_concepts()[:self.num_meta_concepts]

        # do posterior inference
        minibatch_extracted_features = {}
        all_extracted_features = {}
        for i in range(self.max_epochs):
            st_time = time.time()
            is_iter_greedy = self.do_greedy or (i < self.num_greedy_epochs)
            num_concept_holdout = self.num_greedy_holdout if is_iter_greedy else 1
            for j in range(0, self.num_meta_concepts, num_concept_holdout):
                # If minibatch, then subset data first
                minibatch_df = data_df
                minibatch_X_features = X_features
                minibatch_y_train = minibatch_df['y'].to_numpy().flatten()

                all_extracted_features = common.extract_features_by_llm_grouped(
                    self.llm_extraction,
                    minibatch_df,
                    meta_concept_dicts=meta_concept_dicts[:self.num_meta_concepts],
                    all_extracted_features_dict=all_extracted_features,
                    prompt_file=self.prompt_concepts_file,
                    batch_size=self.batch_size,
                    max_new_tokens=8000,
                    batch_concept_size=self.batch_concept_size,
                    group_size=self.batch_obs_size,
                    is_image=self.is_image,
                    max_section_length=self.max_section_length,
                )

                # randomly pick portion of data for generating LLM prior
                train_size = int(minibatch_df.shape[0] * self.train_frac)
                train_idx, test_idx = train_test_split(np.arange(minibatch_df.shape[0]), train_size=train_size, stratify=minibatch_y_train)
                
                num_iter = i * self.num_meta_concepts + j
                # All extracted concepts
                history.add_concepts(meta_concept_dicts)
                logging.info("Iteration %d concepts %s (tot %d)", num_iter, [c['concept'] for c in meta_concept_dicts[:max(num_concept_holdout * 2, 10)]], len(meta_concept_dicts))
                all_concept_extract_feat = common.get_features(meta_concept_dicts, all_extracted_features, minibatch_df, force_keep_columns=self.force_keep_columns)
                train_results = common.train_LR(
                    all_concept_extract_feat,
                    minibatch_y_train,
                    penalty=self.final_learner_type, # this technically should be none from a bayesian perspective
                    use_acc=self.is_greedy_metric_acc)
                logging.info("All extracted concepts AUC %f", train_results["auc"])
                logging.info("All extracted concepts ACC %f", train_results["acc"])
                # coefficients should follow from bayesian model. But it is ok for now...
                history.add_auc(train_results["auc"])
                history.add_coef(train_results["coef"])
                logging.info('All extracted concepts coef %s', train_results["coef"])
                history.add_intercept(train_results["intercept"])
                history.add_model(train_results["model"])
                history.save(training_history_file)
                history.plot_aucs(aucs_plot_file)

                # Extracted features minus the held out concept
                concept_subset_dicts = meta_concept_dicts[:(self.num_meta_concepts - num_concept_holdout)]
                concept_subset = []
                for k, c in enumerate(concept_subset_dicts):
                    print("CURRENT META-CONCEPT", k, c["concept"])
                    concept_subset.append(c['concept'])

                # generate LLM prior, dropping one concept from the mix
                extracted_features = common.get_features(concept_subset_dicts, all_extracted_features, minibatch_df, force_keep_columns=self.force_keep_columns)

                X_scrubbed, feat_names_scrubbed = self.scrub_vectorized_sentences(minibatch_X_features, feat_names, concept_subset_dicts)
                concepts_to_replace = meta_concept_dicts[-num_concept_holdout:]
                iter_llm_prompt, meta_concepts_text, top_features_text, top_feat_names = self.make_new_concept_prompt(
                    X_extracted=extracted_features[train_idx],
                    X_words=X_scrubbed[train_idx],
                    y = minibatch_y_train[train_idx],
                    data_df = minibatch_df.iloc[train_idx],
                    extract_feature_names=concept_subset,
                    feat_names=feat_names_scrubbed,
                    num_replace=num_concept_holdout,
                    )
                print('END OF PROMPT\n\n\n\n')

                # ask for candidates
                raw_candidate_concept_dicts = self.query_for_new_cand(iter_llm_prompt, top_feat_names, max_new_tokens=self.max_new_tokens)
                print("raw_candidate_concept_dicts", len(raw_candidate_concept_dicts), raw_candidate_concept_dicts)
                
                # extract candidate concepts
                all_extracted_features = common.extract_features_by_llm_grouped(
                    self.llm_extraction,
                    minibatch_df, 
                    raw_candidate_concept_dicts,
                    all_extracted_features_dict=all_extracted_features,
                    prompt_file=self.prompt_concepts_file,
                    batch_size=self.batch_size,
                    max_new_tokens=8000,
                    batch_concept_size=self.batch_concept_size,
                    group_size=self.batch_obs_size,
                    is_image=self.is_image,
                    max_section_length=self.max_section_length,
                    extraction_file=self.out_extractions_file,
                )
                print("ALL KEYS", all_extracted_features.keys())

                if is_iter_greedy:
                    # do greedy selection of new concept
                    selected_concept_dicts = self._do_greedy_step(
                        minibatch_df,
                        extracted_features, 
                        minibatch_y_train,
                        raw_candidate_concept_dicts,
                        all_extracted_features,
                        existing_concept_dicts=concepts_to_replace,
                    )
                else:
                    # Get prior
                    assert len(concepts_to_replace) == 1
                    prior_llm_prompt = self.make_concept_prior_prompt(
                        raw_candidate_concept_dicts,
                        concepts_to_replace[0]["concept"],
                        meta_concepts_text,
                        top_features_text)
                    prior_response = self.llm_iter.get_output(prior_llm_prompt, max_new_tokens=5000, response_model=PriorResponse)
                    all_concept_dicts = concepts_to_replace + raw_candidate_concept_dicts
                    all_concept_dicts = prior_response.fill_candidate_concept_dicts(all_concept_dicts)
                    logging.info("candidate concept dicts %s", all_concept_dicts[1:])
                    
                    # compute posterior and do gibbs-like sampling
                    selected_concept_dict = self._do_acceptance_rejection_step(
                        minibatch_df,
                        extracted_features, 
                        minibatch_y_train,
                        train_idx,
                        test_idx,
                        all_concept_dicts[1:],
                        all_extracted_features,
                        backward_prob=all_concept_dicts[0]['prior'],
                        existing_concept=all_concept_dicts[0],
                    )
                    selected_concept_dicts = [selected_concept_dict]
                meta_concept_dicts = selected_concept_dicts + concept_subset_dicts

                logging.info("-------------------------")
                logging.info("posterior sample: %s", [c["concept"] for c in meta_concept_dicts[:self.num_meta_concepts]])
                for c in meta_concept_dicts[:self.num_meta_concepts]:
                    logging.info("posterior sample iter %d: %s", num_iter, c['concept'])

                logging.info("Time for iteration %d: %d (sec)", j, time.time() - st_time)
                
            logging.info("Time for epoch %d (sec)", time.time() - st_time)
    
    @staticmethod
    def fit_residual(model, word_names, X_extracted, X_words, y_train, penalty_downweight_factor: float, is_multiclass: bool, num_top: int, use_acc: bool = False, seed: int = None):
        if X_extracted is None:
            num_fixed = 0
            word_resid_X = X_words
        else:
            num_fixed = X_extracted.shape[1]
            word_resid_X = np.concatenate([X_extracted * penalty_downweight_factor, X_words], axis=1)
            # word_resid_X = np.concatenate([X_extracted, X_words], axis=1)
        results = common.train_LR(word_resid_X, y_train, penalty=model, use_acc=use_acc, seed=seed)
        print("MODEL ACC AUC", results["acc"], results["auc"])
        logging.info("residual fit AUC: %f", results["auc"])
        logging.info("residual fit ACC: %f", results["acc"])
        logging.info("COEFS fixed %s", results["coef"][:,:num_fixed])
        logging.info("COEFS words %s", np.sort(results["coef"][:, num_fixed:]))
        word_coefs = results["coef"][:,num_fixed:]
        
        # display only top features from the residual model
        if not is_multiclass:
            df = pd.DataFrame(
                {
                    'feature_name': word_names,
                    'freq': X_words.mean(axis=0),
                    'coef': word_coefs[0],
                    'abs_coef': np.abs(word_coefs[0]),
                }).sort_values(["abs_coef"], ascending=False)
            top_df = df[df.abs_coef > 0].reset_index().iloc[:num_top]
        else:
            df = pd.DataFrame(
                {
                    'feature_name': word_names,
                    'freq': X_words.mean(axis=0),
                    'coef': np.abs(word_coefs).max(axis=0),
                }).sort_values(["coef"], ascending=False)
            top_df = df[df.coef > 0].reset_index().iloc[:num_top]
        logging.info("top df %s", top_df)
        print("TOP DF", top_df)
        logging.info("freq sort %s", df.sort_values("freq", ascending=False).iloc[:40])
        
        return top_df

        
    def _do_greedy_step(
            self,
            dataset_df,
            extracted_features, # note this is all the extracted concepts MINUS the existing concept (the one we're trying to replace)
            y, 
            candidate_concept_dicts, 
            all_extracted_feat_dict, 
            existing_concept_dicts,
        ):
        all_concept_dicts = existing_concept_dicts + candidate_concept_dicts
        logging.info("concepts (greedy search) %s", [cdict['concept'] for cdict in all_concept_dicts])
        num_orig_features = extracted_features.shape[1]
        selected_concepts = []
        if self.num_greedy_holdout <= 2:
            # do step-wise selection if only selecting 2 concepts
            for i in range(self.num_greedy_holdout):
                concept_scores = []
                for concept_dict in all_concept_dicts:
                    extracted_candidate = common.get_features([concept_dict], all_extracted_feat_dict, dataset_df)
                    aug_extract = np.concatenate([extracted_features, extracted_candidate], axis=1)
                    train_res = common.train_LR(aug_extract, y, penalty=self.residual_model_type, use_acc=self.is_greedy_metric_acc)
                    if self.is_greedy_metric_acc:
                        # use accuracy
                        candidate_score = np.mean(y == train_res["y_assigned_class"])
                    else:
                        # use AUC
                        candidate_score = roc_auc_score(y, train_res["y_pred"], multi_class="ovr")
                    concept_scores.append(candidate_score)
                max_idxs_options = np.where(np.isclose(concept_scores, np.max(concept_scores)))[0]
                max_idx = np.random.choice(max_idxs_options)
                
                extracted_features = np.concatenate([
                    extracted_features,
                    common.get_features([all_concept_dicts[max_idx]], all_extracted_feat_dict, dataset_df)
                    ], axis=1)

                logging.info("concepts (greedy train) %s", concept_scores)
                logging.info("selected concept (greedy) %s", all_concept_dicts[max_idx]['concept'])
                logging.info("greedy-accept %s", max_idx >= len(existing_concept_dicts))

                selected_concepts.append(all_concept_dicts[max_idx])
                all_concept_dicts.pop(max_idx)
        else:
            # use lasso to do selection if selecting multiple concepts
            extracted_candidates = common.get_features(all_concept_dicts, all_extracted_feat_dict, dataset_df)
            aug_extract = np.concatenate([extracted_features, extracted_candidates], axis=1)
            train_res = common.train_LR(aug_extract, y, penalty=self.residual_model_type, use_acc=self.is_greedy_metric_acc)
            coef_magnitudes = np.max(np.abs(train_res['coef'][:, num_orig_features:]), axis=0)
            # get the concepts with the largest maximum magnitudes
            feat_idx_sorted = np.argsort(-coef_magnitudes)
            max_idxs = feat_idx_sorted[:self.num_greedy_holdout]
            for max_idx in max_idxs:
                logging.info("selected concept (greedy) %s", all_concept_dicts[max_idx]['concept'])
            selected_concepts = [all_concept_dicts[i] for i in max_idxs]
        
        return selected_concepts
    
    def _do_acceptance_rejection_step(
            self,
            data_df,
            extracted_features, # note this is all the extracted concepts MINUS the existing concept (the one we're trying to replace)
            y, 
            train_idx, 
            test_idx, 
            candidate_concept_dicts, 
            all_extracted_feat_dict, 
            backward_prob: float,
            existing_concept: dict
        ):
        """
        Compute the posterior distribution over the candidate concepts
        """
        # For each new concept (gamma_b) calculate its weight
        for concept_dict in candidate_concept_dicts:
            extracted_candidate = common.get_features([concept_dict], all_extracted_feat_dict, data_df)
            concept_dict['prob_gamma_b_given_D'] = self._get_prob_concepts_given_D(
                    extracted_features, 
                    y, 
                    train_idx, 
                    test_idx, 
                    prior=concept_dict["prior"],
                    extracted_candidate=extracted_candidate,
                    )
            concept_dict["forward_weight"] = concept_dict['prob_gamma_b_given_D'] * backward_prob
            logging.info("Candidate concept %s", concept_dict)

        # logging.info("Candidate concept weights %s", candidate_concept_dicts)

        # randomly choose a new concept (gamma_b) based on its weight
        selection_weights = np.array([concept_dict["forward_weight"] for concept_dict in candidate_concept_dicts])
        selection_weights /= np.sum(selection_weights)
        try:
            new_concept_idx = np.random.choice(
                    len(selection_weights), 
                    size=1, 
                    replace=False, 
                    p=selection_weights
                    )[0]
        except Exception as e:
            print(e)
            breakpoint()

        new_concept = candidate_concept_dicts[new_concept_idx]
        logging.info("Selected new candidate concept %s (%sselection_weights)", new_concept, selection_weights)
        # calculate acceptance ratio
        forward_transition_prob_new = np.sum([concept_dict["forward_weight"] for concept_dict in candidate_concept_dicts])
        logging.info("Forward transition probability %s", forward_transition_prob_new)

        # LLM_prior(selected new candidate | held-out concepts)
        existing_extracted_concept = common.get_features([existing_concept], all_extracted_feat_dict, data_df)
        existing_concept['prob_gamma_b_given_D'] = self._get_prob_concepts_given_D(
                extracted_features, 
                y, 
                train_idx, 
                test_idx,
                prior=backward_prob,
                extracted_candidate=existing_extracted_concept
                ) 
        other_candidate_concepts = candidate_concept_dicts[:new_concept_idx] + candidate_concept_dicts[new_concept_idx+1:] + [existing_concept]
        logging.info("Other  candidate concepts %s", other_candidate_concepts)
        for concept_dict in other_candidate_concepts:
            concept_dict["backward_weight"] = concept_dict['prob_gamma_b_given_D'] * new_concept["prior"] 

        backward_transition_prob_new = np.sum([concept_dict["backward_weight"] for concept_dict in other_candidate_concepts])
        logging.info("Backward transition probability %s", backward_transition_prob_new)

        alpha = forward_transition_prob_new/backward_transition_prob_new
        logging.info("alpha_new %s", alpha)

        # accept or reject
        acceptance_ratio = min(1, alpha)
        is_accept = np.random.binomial(1, acceptance_ratio)
        logging.info("MH-accept %d (accept ratio %.4f)", is_accept, acceptance_ratio)
        print("MH-accept %d (accept ratio %.4f)", is_accept, acceptance_ratio)
        return new_concept if is_accept else existing_concept

    def _get_prob_concepts_given_D(
            self,
            extracted_features, 
            y, 
            train_idx, 
            test_idx, 
            prior: float,
            extracted_candidate,
            C=1000 # inverse lambda for ridge penalty
        ):
        if len(extracted_candidate) > 0:
            X_candidate = np.concatenate([extracted_features, extracted_candidate], axis=1)
        else:
            X_candidate = extracted_features
        # D1 and D2
        X = np.hstack((np.ones((X_candidate.shape[0], 1)), X_candidate))
        # D1
        X_1, y_1 = X[train_idx], y[train_idx]
        # D2
        X_2, y_2 = X[test_idx], y[test_idx]

        # fit LR on D1 and D2 with ridge penalty and evaluate on D2
        lik_2, invcov, theta = self.compute_lik_and_invcov_mat(train_X=X, train_y=y, C=C, eval_X=X_2, eval_y=y_2) 
        logging.info("Likelihood of model trained on D1 and D2, evaluated on D2 %s", lik_2)

        # fit LR on D1 with ridge penalty and evaluate on D1
        lik_1, invcov_1, theta_1 = self.compute_lik_and_invcov_mat(train_X=X_1, train_y=y_1, C=C, eval_X=X_1, eval_y=y_1) 
        logging.info("Likelihood of model trained on D1, evaluated on D1 %s", lik_1)

        laplace_prob_theta_D1 = (np.linalg.det(invcov_1) ** .5) * np.exp(-0.5 * (theta - theta_1).T @ invcov_1 @ (theta - theta_1))
        laplace_approx_prob = lik_2 * laplace_prob_theta_D1[0,0] / (np.linalg.det(invcov) ** .5)
        logging.info("Prob concepts given D Laplace approx %s", laplace_approx_prob)
        return laplace_approx_prob * prior
    
    def compute_lik_and_invcov_mat(self, train_X, train_y, C, eval_X, eval_y):
        model = LogisticRegression(penalty='l2', C=C, fit_intercept=False, multi_class="multinomial", max_iter=10000)
        model.fit(train_X, train_y)
        theta = model.coef_.flatten().reshape((-1,1))
        # evaluate on evaluation data
        pred_prob = model.predict_proba(eval_X)
        lik = np.exp(np.sum(common.get_log_liks(
                eval_y,
                pred_prob if self.is_multiclass else pred_prob[:,1],
                is_multiclass=self.is_multiclass)))

        # get covariance matrix
        if not self.is_multiclass:
            pred_prob = model.predict_proba(train_X)[:,1]
            diagonal = np.diag(pred_prob * (1 - pred_prob))
            # phi.T * D * phi + 1/C * I
            invcov_mat = (train_X.T @ diagonal @ train_X) + (1/C * np.identity(train_X.shape[1]))
        else:
            pred_prob = model.predict_proba(train_X)
            scaled_Xs = [train_X * pred_prob[:,i:i+1] for i in range(pred_prob.shape[1])]
            scaled_X_mat = np.concatenate(scaled_Xs, axis=1)
            diag_scaled_X = scipy.linalg.block_diag(*[train_X.T @ scaled_X for scaled_X in scaled_Xs])
            invcov_mat = diag_scaled_X - scaled_X_mat.T @ scaled_X_mat + (1/C * np.identity(scaled_X_mat.shape[1]))

        return lik, invcov_mat, theta
    
    def fill_config(self, template_str: str):
        for k, v in self.config.items():
            template_str = template_str.replace(k, v)
        return template_str
    
    def make_new_concept_prompt(
            self,
            X_extracted,
            X_words,
            y,
            data_df,
            extract_feature_names, 
            feat_names,
            num_replace: int = 1, 
        ):
        """
        Generate prompt to ask LLM for candidate concepts
        """
        with open(self.prompt_iter_file, 'r') as file:
            prompt_template = file.read()

        top_df = self.fit_residual(
            self.residual_model_type,
            feat_names.tolist(),
            X_extracted,
            X_words,
            y,
            penalty_downweight_factor=self.penalty_downweight_factor,
            is_multiclass=self.is_multiclass,
            num_top=self.num_top,
            use_acc=self.is_greedy_metric_acc,
        )
        
        # normalize the coefficients just to make it a bit easier to read for the LLM
        normalization_factor = np.max(np.abs(top_df.coef))
        top_df['coef'] = top_df.coef/normalization_factor if normalization_factor > 0 else top_df.coef
        # Generate the prompt with the top features
        top_features_text = top_df[['feature_name', 'coef']].to_csv(index=False, float_format='%.3f')
        prompt_template = prompt_template.replace("{top_features_df}", top_features_text)

        prompt_template = prompt_template.replace("{num_concepts}", str(self.num_meta_concepts))
        meta_concepts_text = ""
        for i, feat_name in enumerate(extract_feature_names):
            meta_concepts_text += f"* X{i} = {feat_name} \n" 
        prompt_template = prompt_template.replace("{meta_concepts}", meta_concepts_text)
        # prompt_template = prompt_template.replace("{meta_to_replace}", concepts_to_replace)
        prompt_template = prompt_template.replace("{num_concepts_fixed}", str(self.num_meta_concepts - num_replace))
        prompt_template = prompt_template.replace("{num_attributes}", str(top_df.shape[0]))
        
        prompt_template = self.fill_config(prompt_template)
        return prompt_template, meta_concepts_text, top_features_text, top_df.feature_name

    def scrub_vectorized_sentences(self, X_features, feat_names, concept_dicts: list):
        # Remove the words that are too correlated with the concepts from the residual model's inputs
        words_to_scrub = [w for c in concept_dicts if not common.is_tabular(c['concept']) for w in c['words'] if len(w) > 2]
        keep_mask = [
            ~np.any([scrub_word in w for scrub_word in words_to_scrub]) or common.is_tabular(w)
            for w in feat_names]
        return X_features[:, keep_mask], feat_names[keep_mask]
    
    def query_for_new_cand(self, iter_llm_prompt, top_feat_names, times_to_retry=1, max_new_tokens=5000):
        llm_response = self.llm_iter.get_output(iter_llm_prompt, max_new_tokens=max_new_tokens, response_model=CandidateConcepts)
        candidate_concept_dicts = llm_response.to_dicts(default_prior=self.default_prior)
        candidate_concept_dicts += [{
            "concept": feat_name,
            "prior": self.default_prior
        } for feat_name in top_feat_names if common.is_tabular(feat_name)]
        return candidate_concept_dicts[:self.max_candidates_to_consider]

    def make_concept_prior_prompt(self, concept_dicts, concept_to_replace, meta_concepts_text, top_words_text):
        """
        Generate prompt to ask LLM for candidate concepts
        """
        with open(self.prompt_prior_file, 'r') as file:
            prompt_template = file.read()
        prompt_template = prompt_template.replace("{num_concepts}", str(self.num_meta_concepts))
        prompt_template = prompt_template.replace("{num_concepts_fixed}", str(self.num_meta_concepts - 1))
        prompt_template = prompt_template.replace("{meta_concepts}", meta_concepts_text)
        candidate_concepts_text = f"0. {concept_to_replace}\n"
        for i, concept_dict in enumerate(concept_dicts):
            candidate_concepts_text += f"{i + 1}. {concept_dict['concept']}\n" 
        prompt_template = prompt_template.replace("{candidate_list}", candidate_concepts_text)
        prompt_template = prompt_template.replace("{top_features_df}", top_words_text)
        
        prompt_template = self.fill_config(prompt_template)
        return prompt_template