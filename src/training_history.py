import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TrainingHistory:
    def __init__(self, force_keep_cols: np.ndarray = None):
        self.force_keep_cols = force_keep_cols
        self._concepts = []
        self._models = []
        self._aucs = []
        self._coefs = []
        self._intercepts = []
        self._log_liks = []
    
    @property
    def num_iters(self):
        return len(self._concepts)

    def get_last_concepts(self) -> list[dict]:
        return self._concepts[-1]

    def get_last_model(self):
        return self._models[-1]

    def get_last_auc(self) -> float:
        return self._aucs[-1]

    def get_last_log_liks(self) -> list:
        if len(self._log_liks):
            return self._log_liks[-1]
        else:
            return None

    def load(self, file_name: str):
        if os.path.exists(file_name):
            with open(file_name, 'rb') as file:
                self = pickle.load(file)
        else:
            raise FileNotFoundError("FILE?", file_name)
                
        return self

    def add_auc(self, auc: float):
        self._aucs.append(auc)

    def add_log_liks(self, log_liks: list):
        self._log_liks.append(log_liks)

    def add_coef(self, coef: float):
        self._coefs.append(coef)

    def add_model(self, model):
        self._models.append(model)

    def add_intercept(self, intercept: float):
        self._intercepts.append(intercept)

    def get_model(self, index:int):
        return self._models[index]
    
    def add_concepts(self, concepts: list[dict]):
        self._concepts.append(concepts)

    def plot_aucs(self, plot_file: str):
        plt.clf()
        plt.plot(self._aucs, linestyle='--')
        plt.title("All extracted concepts AUC")

        plt.tight_layout()
        plt.savefig(plot_file)

    def save(self, file_name: str):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)



