import numpy as np
from core.logistic_regression import LogisticRegression


def test_logistic_classification():
    # 1. Prepare linearly separable binary data
    # X: Features (e.g., hours studied), y: Target (0: Fail, 1: Pass)
    X = np.array([[1],[2],[3],[7],[8],[9]], dtype=float)
    y = np.array([[0],[0],[0],[1],[1],[1]], dtype=float)

    model = LogisticRegression(learning_rate=0.5,n_iterations=800)


    # 3. Training phase
    model.fit(X,y)

    # 4. Inference
    predictions = model.predict(X)
    probabilities = model.predict_proba(predictions)
    
   # 5. Validation (QA Checklist)
    accuracy = np.mean(predictions == y)
    print(f"\nFinal Probabilities:\n{probabilities.flatten()}")
    print(f"Final Predictions: {predictions.flatten()}")
    print(f"Accuracy: {accuracy * 100}%")
    
    if accuracy == 1.0:
        print("\n✅ Result: PASSED (Model successfully separated classes)")
    else:
        print("\n❌ Result: FAILED (Check learning rate or iterations)")
    
    

if __name__ == "__main__":
    test_logistic_classification()