"""
Bootstraps gamma for the concept learner model
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset
from src.concept_learner_model import ConceptKernelLearner

class Bootstrap():
    def __init__(self,
        dataset: TensorDataset,
        make_and_train_model, # function
        out_params_file: str,
        seed: int = 0,
        logging = None
    ):
        self._set_seeds(seed)

        self.logging = logging
        self.dataset = dataset
        self.make_and_train_model = make_and_train_model
        self.out_params_file = out_params_file

    def _set_seeds(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def run(self,
        n_bootstraps: int = 1000
    ):
        seeds = np.random.randint(0, 2**32 - 1, size=n_bootstraps)

        state_dicts = []
        for n_boostrap, seed in enumerate(seeds):
            if self.logging:
                self.logging.info("--------Bootstrap Iteration: %s----------", n_boostrap)

            self._set_seeds(seed)

            rand = np.random.RandomState()
            indices = rand.randint(0, len(self.dataset), len(self.dataset))

            resampled_dataset = Subset(self.dataset, indices)

            assert len(resampled_dataset) == len(self.dataset)

            # fit concept model
            cl_model = self.make_and_train_model(resampled_dataset)

            # save concept model params
            state_dicts.append(cl_model.state_dict())
            torch.cuda.empty_cache()

        torch.save(state_dicts, self.out_params_file)
