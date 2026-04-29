import numpy as np

class BaseLinearRegression:
    def __init__(self,learning_rate=0.01,n_iterations=1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None
        
    def _get_penalty_derivative(self):
        """
        Base method for regularization penalty.
        To be overridden by child classes (L1/L2).
        Returns 0 for standard Linear Regression.
        """
        return 0

    def fit(self,X, y):
        """
        Train the model using Gradient Descent.
        
        X shape: (n_samples, n_features) -> e.g., (100 houses, 2 features)
        y shape: (n_samples, n_targets)  -> e.g., (100 house prices, 1 target)
        """
        # 1. Get Data Dimensions
        # n_samples  -> Used as a divisor for mean gradient calculation
        # n_features -> Used to determine the number of rows in the Weights matrix
        n_samples, n_features = X.shape
        
        # 2. Get Target Dimensions to prevent hard-coding
        # Supports Multi-Output Regression (e.g., predicting 2 different values at once)
        n_targets = y.shape[1]
        
        # 3. Initialize Weights and Bias based on feature and target count
        # Initialized with zeros as the starting point for Gradient Descent
        self.weights = np.zeros((n_features, n_targets))
        self.bias = np.zeros((1, n_targets))
        
         # --- PHASE 2: TRAINING LOOP (GRADIENT DESCENT) ---
        for _ in range(self.n_iters):
            # Step 1: Forward Pass (Predict using current weights)
            y_pred = self.forward(X)
            
            # Step 2: Calculate Error (Residuals)
            error = y_pred - y
            
            # Step 3: Calculate Gradients (Using Partial Derivatives)
            # Transpose (X.T) is required to align features with errors
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error, axis=0)
            
            # Injecting the penalty cleanly (Polymorphism)
            # This is where L1 or L2 will modify the gradient
            dw += self._get_penalty_derivative()
            
            # Adjust weights in the opposite direction of the gradient
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
    def forward(self, X):
        """
        Input X: (n_samples, n_features)
        Output: (n_samples, 1)
        """
        # State Validation: Ensure model is trained before inference
        if self.weights is None or self.bias is None:
            raise RuntimeError(
                "Model is not trained yet"
                )
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        """
        Alias for forward pass.
        Exposed as a public API for consistency with ML library standards.
        """
        return self.forward(X)
    


class RidgeRegression(BaseLinearRegression):
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_param=0.1):
        # Call the parent's initialization
        super().__init__(learning_rate, n_iterations)
        self.lambda_param = lambda_param
        
    def _get_penalty_derivative(self):
        """
        Calculate L2 (Ridge) penalty derivative.
        Formula: lambda * weights
        """
        return self.lambda_param * self.weights
    

class LassoRegression(BaseLinearRegression):
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_param=0.1):
        super().__init__(learning_rate, n_iterations)
        self.lambda_param = lambda_param

    def _get_penalty_derivative(self):
        """
        Calculate L1 (Lasso) penalty derivative.
        Formula: lambda * sign(weights)
        """
        return self.lambda_param * np.sign(self.weights)
