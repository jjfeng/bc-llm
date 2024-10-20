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

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from src.llm.llm import LLM
from src.utils import convert_to_json, convert_to_list_of_jsons
from src.training_history import TrainingHistory
import src.common as common

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
    keep_num_candidates = 10
    def __init__(
            self,
            init_history: TrainingHistory,
            llm_iter: LLM,
            llm_extraction: LLM,
            num_classes: int,
            num_meta_concepts: int,
            prompt_iter_type: str,
            prompt_iter_file: str,
            prompt_concepts_file: str,
            prompt_prior_file: str,
            out_extractions_file: str,
            residual_model_type: str,
            inverse_penalty_param: float, # inverse regularization for the l2 penalty for logistic regression
            train_frac: float = 0.5,
            num_greedy_epochs: int = 0,
            max_epochs: int = 10,
            batch_size: int = 4,
            do_greedy: bool = False,
            is_image: bool = False,
            all_extracted_features_dict = {},
            max_section_length: int = None,
            force_keep_columns: pd.Series = None,
            max_new_tokens: int = 5000,
            num_top: int = 40,
            ):
        self.init_history = init_history
        self.llm_iter = llm_iter
        self.llm_extraction = llm_extraction
        self.num_classes = num_classes
        self.is_multiclass = num_classes > 2
        self.num_meta_concepts = num_meta_concepts
        self.prompt_iter_type = prompt_iter_type
        assert self.prompt_iter_type == "conditional"
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
        self.all_extracted_features_dict = all_extracted_features_dict
        self.is_image = is_image
        self.do_greedy = do_greedy
        self.max_section_length = max_section_length
        self.force_keep_columns = force_keep_columns
        self.max_new_tokens = max_new_tokens
        self.num_top = num_top
    
    def fit(self,
            data_df,
            X_features,
            feat_names,
            training_history_file: str,
            aucs_plot_file: str):
        history = self.init_history
        y_train = data_df['y'].to_numpy().flatten()

        # initialize concepts
        meta_concept_dicts = history.get_last_concepts()[:self.num_meta_concepts]
        # breakpoint()
        
        all_extracted_features = common.extract_features_by_llm(
                self.llm_extraction,
                data_df,
                meta_concept_dicts=meta_concept_dicts[:self.num_meta_concepts],
                all_extracted_features_dict=self.all_extracted_features_dict,
                prompt_file=self.prompt_concepts_file,
                extraction_file=self.out_extractions_file,
                batch_size=self.batch_size,
                is_image=self.is_image,
                max_section_length=self.max_section_length
        )

        # do posterior inference
        for i in range(self.max_epochs):
            st_time = time.time()
            for j in range(self.num_meta_concepts):
                # randomly pick portion of data for generating LLM prior
                train_size = int(data_df.shape[0] * self.train_frac)
                train_idx, test_idx = train_test_split(np.arange(data_df.shape[0]), train_size=train_size, stratify=y_train)
                
                num_iter = i * self.num_meta_concepts + j
                # All extracted concepts
                history.add_concepts(meta_concept_dicts)
                logging.info("Iteration %d concepts %s", num_iter, meta_concept_dicts)
                all_concept_extract_feat = common.get_features(meta_concept_dicts, all_extracted_features, data_df, force_keep_columns=self.force_keep_columns)
                train_results = common.train_LR(all_concept_extract_feat, y_train)
                logging.info("All extracted concepts AUC %f", train_results["auc"])
                # coefficients should follow from bayesian model. But it is ok for now...
                history.add_auc(train_results["auc"])
                history.add_coef(train_results["coef"])
                logging.info('All extracted concepts coef %s', train_results["coef"])
                history.add_intercept(train_results["intercept"])
                history.add_model(train_results["model"])

                # Extracted features minus the held out concept
                concept_subset_dicts = meta_concept_dicts[:(self.num_meta_concepts - 1)]
                concept_subset = []
                for k, c in enumerate(concept_subset_dicts):
                    print("CURRENT META-CONCEPT", k, c["concept"])
                    concept_subset.append(c['concept'])

                # generate LLM prior, dropping one concept from the mix
                extracted_features = common.get_features(concept_subset_dicts, all_extracted_features, data_df, force_keep_columns=self.force_keep_columns)

                X_scrubbed, feat_names_scrubbed = self.scrub_vectorized_sentences(X_features, feat_names, concept_subset_dicts)
                concept_to_replace = meta_concept_dicts[self.num_meta_concepts - 1]
                iter_llm_prompt, meta_concepts_text, top_features_text, top_feat_names = self.make_new_concept_prompt(
                    X_extracted=extracted_features[train_idx],
                    X_words=X_scrubbed[train_idx],
                    y = y_train[train_idx],
                    data_df = data_df.iloc[train_idx],
                    extract_feature_names=concept_subset,
                    concept_to_replace=concept_to_replace["concept"],
                    feat_names=feat_names_scrubbed)
                print(iter_llm_prompt)
                print('END OF PROMPT\n\n\n\n')

                # ask for candidates
                raw_candidate_concept_dicts = self.query_for_new_cand(iter_llm_prompt, top_feat_names, max_new_tokens=self.max_new_tokens)
                print(raw_candidate_concept_dicts)
                
                # extract candidate concepts
                # breakpoint()
                all_extracted_features = common.extract_features_by_llm(
                    self.llm_extraction,
                    data_df, 
                    raw_candidate_concept_dicts,
                    all_extracted_features_dict=all_extracted_features,
                    prompt_file=self.prompt_concepts_file,
                    extraction_file=self.out_extractions_file,
                    batch_size=self.batch_size,
                    is_image=self.is_image,
                    max_section_length=self.max_section_length
                )

                if self.do_greedy or (i < self.num_greedy_epochs):
                    # do greedy selection of new concept
                    selected_concept_dict = self._do_greedy_step(
                        data_df,
                        extracted_features, 
                        y_train,
                        raw_candidate_concept_dicts,
                        all_extracted_features,
                        existing_concept_dict=concept_to_replace,
                    )
                else:
                    # Get prior
                    prior_llm_prompt = self.make_concept_prior_prompt(
                        raw_candidate_concept_dicts,
                        concept_to_replace["concept"],
                        meta_concepts_text,
                        top_features_text)
                    print(prior_llm_prompt)
                    candidate_concept_dicts, backward_llm_prior = common.query_and_parse_llm(
                        self.llm_iter,
                        prior_llm_prompt,
                        extract_func=lambda x: self.extract_concept_prior(x, raw_candidate_concept_dicts),
                        default_response=(raw_candidate_concept_dicts, self.default_prior))
                    concept_to_replace["prior"] = backward_llm_prior
                    print(candidate_concept_dicts)
                    logging.info("candidate concept dicts %s", candidate_concept_dicts)
                    
                    # compute posterior and do gibbs-like sampling
                    selected_concept_dict = self._do_acceptance_rejection_step(
                        data_df,
                        extracted_features, 
                        y_train,
                        train_idx,
                        test_idx,
                        candidate_concept_dicts,
                        all_extracted_features,
                        backward_prob=backward_llm_prior,
                        existing_concept=concept_to_replace,
                    )
                meta_concept_dicts = [selected_concept_dict] + concept_subset_dicts

                logging.info("-------------------------")
                logging.info("posterior sample: %s", [c["concept"] for c in meta_concept_dicts[:self.num_meta_concepts]])
                for c in meta_concept_dicts[:self.num_meta_concepts]:
                    logging.info("posterior sample iter %d: %s", num_iter, c['concept'])
                
                history.save(training_history_file)
                history.plot_aucs(aucs_plot_file)
            logging.info("Time for epoch %d (sec)", time.time() - st_time)
    
    @staticmethod
    def fit_residual(model, word_names, X_extracted, X_words, y_train, penalty_downweight_factor: float, is_multiclass: bool, num_top: int):
        if X_extracted is None:
            num_fixed = 0
            word_resid_X = X_words
        else:
            num_fixed = X_extracted.shape[1]
            word_resid_X = np.concatenate([X_extracted * penalty_downweight_factor, X_words], axis=1)
            # word_resid_X = np.concatenate([X_extracted, X_words], axis=1)
        results = common.train_LR(word_resid_X, y_train, penalty=model)
        print("COEF", results["coef"].shape)
        logging.info("residual fit AUC: %f", results["auc"])
        logging.info("COEFS fixed %s", results["coef"][:,:num_fixed])
        logging.info("COEFS words %s", np.sort(results["coef"][0, num_fixed:]))
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
            existing_concept_dict,
        ):
        concept_scores = []
        all_concept_dicts = candidate_concept_dicts + [existing_concept_dict]
        for concept_dict in all_concept_dicts:
            extracted_candidate = common.get_features([concept_dict], all_extracted_feat_dict, dataset_df)
            aug_extract = np.concatenate([extracted_features, extracted_candidate], axis=1)
            train_res = common.train_LR(aug_extract, y)
            candidate_score = roc_auc_score(y, train_res["y_pred"], multi_class="ovo")
            concept_scores.append(candidate_score)
        max_idxs = np.where(np.isclose(concept_scores, np.max(concept_scores)))[0]
        max_idx = np.random.choice(max_idxs)
        logging.info("concepts (greedy search) %s", [cdict['concept'] for cdict in all_concept_dicts])
        logging.info("concepts (greedy train) %s", concept_scores)
        logging.info("selected concept (greedy) %s", all_concept_dicts[max_idx]['concept'])
        logging.info("greedy-accept %s", max_idx < len(candidate_concept_dicts))
        return all_concept_dicts[max_idx]
    
    def _do_acceptance_rejection_step(
            self,
            data_df,
            extracted_features, # note this is all the extracted concepts MINUS the existing concept (the one we're trying to replace)
            y, 
            train_idx, 
            test_idx, 
            candidate_concept_dicts, 
            all_extracted_feat_dict, 
            backward_prob,
            existing_concept
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

    @staticmethod
    def get_candidate_concepts(llm_output, keep_num_candidates: int = 6, default_prior: float = 1) -> list[dict]:
        #Process the LLM candidates to make sure that we have all the info we need
        #Also shorten the list of candidates if the LLM outputs too many
        
        def _clean_prior_dict(d):
            if not hasattr(d, 'words'):
                for k in ['synonyms', 'phrases']:
                    if k in prior_dicts[i]:
                        d['words'] = d[k]

            return d
            

        try:
            logging.info("candidate concept summary =================")
            try:
                outs = llm_output.split('\n')
                jsons = []
                for out in outs:
                    try:
                        jsons.append(json.loads(out))
                    except:
                        pass           

                # reformat into prior_dicts
                prior_dicts = []
                for i in range(len(jsons)):
                    prior_dicts.append({
                        'concept': list(jsons[i].keys())[0],
                        'words': list(jsons[i].values())[0],
                    })
                # breakpoint()


            except:
                llm_output = convert_to_json(llm_output)
                if 'concepts' in llm_output:
                    prior_dicts = llm_output["concepts"]
                elif 'candidates' in llm_output:
                    prior_dicts = llm_output["candidates"]
                else:
                    prior_dicts = llm_output
                if not isinstance(prior_dicts, list):
                    prior_dicts = convert_to_list_of_jsons(llm_output)
                    for i in range(len(prior_dicts)):
                        if not hasattr('words', prior_dicts[i]):
                            for k in ['synonyms', 'phrases']:
                                if k in prior_dicts[i]:
                                    prior_dicts[i] = _clean_prior_dict(prior_dicts[i])
            # breakpoint()
            for i in range(len(prior_dicts)):
                prior_dicts[i] = _clean_prior_dict(prior_dicts[i])
                concept = prior_dicts[i]['concept']
                assert len(concept)
                if not concept.startswith(common.TABULAR_PREFIX):
                    prior_dicts[i]['words'] = [w.lower().strip() for w in prior_dicts[i]['words'].split(",")]
                # Fill out each concept with a default prior -- to be overwritten as needed
                prior_dicts[i]['prior'] = default_prior
                logging.info("CONCEPT %s %s", prior_dicts[i]['concept'], prior_dicts[i])

            assert len(prior_dicts) > 0
        except Exception as e:
            logging.info("ERROR in extracting candidate concepts %s", e)
            raise ValueError("bad JSON llm response")
            breakpoint()

        logging.info("concept prior dict %s", prior_dicts)

        # only keep the first keep_num candidates, because otherwise this inference procedure will take forever
        top_candidates = prior_dicts[:keep_num_candidates]
        logging.info("top candidates %s", top_candidates)

        return top_candidates

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
    
    def make_new_concept_prompt(
            self,
            X_extracted,
            X_words,
            y,
            data_df,
            extract_feature_names, 
            concept_to_replace, 
            feat_names
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
            num_top=self.num_top
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
        prompt_template = prompt_template.replace("{meta_to_replace}", concept_to_replace)
        prompt_template = prompt_template.replace("{num_concepts_fixed}", str(self.num_meta_concepts - 1))

        return prompt_template, meta_concepts_text, top_features_text, top_df.feature_name

    def scrub_vectorized_sentences(self, X_features, feat_names, concept_dicts: list):
        # Remove the words that are too correlated with the concepts from the residual model's inputs
        words_to_scrub = [w for c in concept_dicts if not common.is_tabular(c['concept']) for w in c['words']]
        keep_mask = [
            ~np.any([scrub_word in w for scrub_word in words_to_scrub]) or common.is_tabular(w)
            for w in feat_names]
        return X_features[:, keep_mask], feat_names[keep_mask]
    
    def query_for_new_cand(self, iter_llm_prompt, top_feat_names, times_to_retry=4, max_new_tokens=5000):
        llm_response = self.llm_iter.get_output(iter_llm_prompt, max_new_tokens=max_new_tokens)
        while times_to_retry > 0:
            # breakpoint()
            print('times_to_retry', times_to_retry)
            try:
                # breakpoint()
                candidate_concept_dicts = self.get_candidate_concepts(llm_response, keep_num_candidates=self.keep_num_candidates, default_prior=self.default_prior)
                candidate_concept_dicts += [{
                    "concept": feat_name,
                    "prior": self.default_prior
                } for feat_name in top_feat_names if common.is_tabular(feat_name)]
                return candidate_concept_dicts
            except Exception:
                logging.info("llm call failed. Retrying query. here is the response %s", llm_response)
                # print('\n\nRESPONSE:', llm_response, '\n\n')
                breakpoint()
                llm_response = self.llm_iter.get_output(iter_llm_prompt, max_new_tokens=max_new_tokens)
                # breakpoint()
                times_to_retry -= 1

    def extract_concept_prior(self, llm_output, concept_dicts, min_prior=1e-3):
        logging.info("concept prior elicitation =================")
        llm_output = convert_to_json(llm_output)
        logging.info("concept prior dict %s", llm_output)
        backwards_prior_prob = max(float(llm_output["0"]), min_prior)
        assert len(llm_output) > 1
        keep_concept_dicts = []
        for k, v in llm_output.items():
            if int(k) == 0:
                continue
            concept_dicts[int(k) - 1]['prior'] = max(float(v), min_prior)
            keep_concept_dicts.append(concept_dicts[int(k) - 1])
        return keep_concept_dicts, backwards_prior_prob
        
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
        return prompt_template

    def get_overlapping_concepts(self, concepts: list, top_feat_names) -> list:
        """Return concepts that contain some metnion of the top feature names
        """
        candidate_concepts = []
        for feat_name in top_feat_names:
            for c in concepts:
                if (feat_name in c) and (c not in candidate_concepts):
                    candidate_concepts.append(c)
                    break
        return list(candidate_concepts)
