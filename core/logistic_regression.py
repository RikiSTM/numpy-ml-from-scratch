import numpy as np
from core.linear_regression import BaseLinearRegression

class LogisticRegression(BaseLinearRegression):
    def _sigmoid(self, z):
        """
        Sigmoid activation function with numerical stability.
        Uses np.clip to prevent exponential overflow during np.exp calculation.
        """
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def forward(self, X):
        """
        Overrides the parent forward method to implement the logistic mapping.
        1. Calculates the linear transformation Z = Xw + b via the parent class.
        2. Applies the Sigmoid activation to squash values between 0 and 1.
        """
        # Calculate linear part (Xw + b) from parent
        z = super().forward(X) 
        return self._sigmoid(z)
        
    def predict_proba(self, X):
        """
        Returns the raw probability scores for each sample.
        Functionally identical to forward(), but exposed as a standard public API.
        """
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        """
        Converts probability scores into discrete binary labels (0 or 1).
        Labels are determined based on a decision threshold (default is 0.5).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)