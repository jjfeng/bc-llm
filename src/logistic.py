import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

class MultiLogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)  # raw logits

class LogisticRegressionTorch(BaseEstimator, ClassifierMixin):
    def __init__(self, lambd: float=0.001, num_epochs: int = 5000, seed:int=None):
        self.lambd = lambd
        self.num_epochs = num_epochs
        if seed is not None:
            torch.manual_seed(seed)

    def fit(self, X, y):
        num_cls = np.unique(y).size
        num_obs = X.shape[0]
        num_p = X.shape[1]
        self.classes_ = np.unique(y)
        self.class_mapping_ = {
            j: i for i,j in enumerate(self.classes_)
        }
        y_remapped = np.array([self.class_mapping_[y_elem] for y_elem in y])
        
        self.model = MultiLogisticRegression(num_p, num_cls)
        self.model.train()
        criterion = nn.CrossEntropyLoss()  # combines softmax and negative log-likelihood
        optimizer = optim.Adam(self.model.parameters())
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_remapped, dtype=torch.long)
        for epoch in range(self.num_epochs):
            # Forward pass: compute predicted logits by passing X_tensor to the model.
            outputs = self.model(X_tensor)
            group_lasso = torch.norm(self.model.linear.weight, p=2, dim=0)
            loss = criterion(outputs, y_tensor) + self.lambd * torch.sum(group_lasso)
            
            # Backward pass: compute gradient of the loss with respect to model parameters.
            optimizer.zero_grad()
            loss.backward()
            
            # Update the model parameters.
            optimizer.step()
            
            if (epoch + 1) % 1000 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
        
        self.coef_ = self.model.linear.weight.detach().numpy()
        self.intercept_ = self.model.linear.bias.detach().numpy()

    
    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.detach().numpy()
    
    def predict(self, X):
        probs = self.predict_proba(X)
        pred_class_raw = np.argmax(probs, axis=1)
        return np.array([self.classes_[i] for i in pred_class_raw])
    
    def get_params(self, deep=False):
        return {
            "lambd": self.lambd,
            "num_epochs": self.num_epochs,
        }
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))