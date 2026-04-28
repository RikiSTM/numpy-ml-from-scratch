import numpy as np
from core.linear_regression import LinearRegression

def test_training_process():
    # 1. PREPARE DATA (Synthetic Data)
    # make learning connection: y = 2x + 0
    X = np.array([[1],[2],[3],[4],[5]], dtype=float)
    y = np.array([[2], [4], [6], [8], [10]], dtype=float)
    
    # 2. INITIALIZE MODEL
    model = LinearRegression(learning_rate=0.01,n_iterations=1000)
    print("--- INTEGRATION TEST START ---")
    
    # 3. BEFORE TRAINING (Negative Test)
    try:
        model.forward(X)
    except RuntimeError as e:
        print(f"✅ Smoke Test Passed: System rejected prediction before training. Message: {e}")
    
    # 4. TRAINING EXECUTION
    print("\nTTraining in progress...")
    model.fit(X, y)
    
    # 5. AFTER TRAINING (Validation)
    final_pred = model.predict(X)
    print(f"\nPrediction after learning (target [2, 4, 6, 8, 10]):")
    print(final_pred.flatten())
    
    
    # 6. ASSERTION
    weight_result = model.weights[0][0]
    print(f"\nFinal learned weight: {weight_result:.4f}")
    
    if np.allclose(final_pred, y, atol=0.1):
        print("\n✅ RESULT: TEST PASSED! Successfully converged.")
    else:
        print("\n❌ RESULT: TEST FAILED! Model failed to learn.")
        
if __name__ == "__main__":
    test_training_process()