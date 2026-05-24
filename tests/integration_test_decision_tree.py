import numpy as np
from core.decision_tree import DecisionTree

def run_decision_tree_test():
    # 1. Prepare Data
    # X: 2 features (e.g., test score 1, test score 2)
    # y: Labels (0: Fail, 1: Pass)
    # Data is clearly separable (low scores = 0, high scores = 1)
    X = np.array([
        [1.0, 1.5], 
        [2.0, 1.0], 
        [1.5, 2.5], 
        [8.0, 8.5], 
        [9.0, 9.0], 
        [8.5, 7.5]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    # 2. Initialize Model
    # max_depth=3 is more than enough for this simple data
    model = DecisionTree(max_depth=3)

    print("--- Decision Tree Integration Test ---")
    print(f"Training on {len(X)} samples...")

    # 3. Training
    model.fit(X, y)

    # 4. Inference
    predictions = model.predict(X)

    # 5. QA Verification
    accuracy = np.mean(predictions == y)
    
    print(f"Ground Truth: {y}")
    print(f"Predictions : {predictions}")
    print(f"Integration Test Accuracy: {accuracy * 100}%")

    assert accuracy == 1.0, "Model failed to perfectly split simple separable data"
    print("\n✅ Result: [PASSED]")

if __name__ == "__main__":
    run_decision_tree_test()