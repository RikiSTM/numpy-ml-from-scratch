import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature       # Index of the feature used for splitting
        self.threshold = threshold   # Threshold value for the split
        self.left = left             # Left child node
        self.right = right           # Right child node
        self.value = value           # If it's a leaf node, store the predicted class label

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def _entropy(self, y):
        """
        Calculate Information Gain (Entropy).
        Formula: - sum(p * log2(p))
        """
        # Calculate the proportion (p) of each class in the target y
        hist = np.bincount(y)
        ps = hist / len(y)
        
        # Avoid log(0) by filtering p > 0
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
        
    def _gini(self, y):
        """
        Calculate Gini Impurity.
        Formula: 1 - sum(p^2)
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        
        return 1 - np.sum(ps ** 2)
    def fit(self, X, y):
        """
        Entry point to train the Decision Tree.
        Starts the recursive tree-building process from the root node.
        """
        # Determine the number of features dynamically
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """
        Recursively builds the decision tree by finding the optimal splits.
        """
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # --- QA Check: Stopping Criteria (Base Cases) ---
        # Stop if: max depth is reached, node is pure (1 label), or too few samples to split
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # --- Phase 1: Find the Best Split ---
        feat_idxs = np.random.choice(n_feats, self.n_features_, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # --- Phase 2: Create Child Nodes Recursively ---
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(feature=best_feature, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, feat_idxs):
        """
        Iterates over all features and unique values to find the best split 
        based on the highest Information Gain.
        """
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        """
        Calculates the Information Gain of a specific split.
        Information Gain = Entropy(Parent) - Weighted_Entropy(Children)
        """
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)

        # If the split is invalid (doesn't divide data), gain is 0
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Calculate the weighted average entropy of the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        """
        Splits the data indices into left and right based on the threshold.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        """
        Helper method to find the majority class in a leaf node.
        """
        counter = np.bincount(y)
        return counter.argmax()

    def predict(self, X):
        """
        Predicts the class labels for a batch of input data.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Recursively traverses the tree for a single sample to find its predicted class.
        """
        # If we reach a leaf node, return the final prediction
        if node.is_leaf_node():
            return node.value

        # Navigate left or right based on the threshold
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)