import numpy as np
from core.linear_regression import RidgeRegression,LassoRegression


def test_regularization_behavior():
    """
    Integration test to verify feature selection capability of L1 vs L2.
    Feature 0: House Size (Strong signal)
    Feature 1: Total Rooms (Weak signal)
    Feature 2: Stray Cats (Noise/Irrelevant)
    """
    print("--- Starting Regularization Integration Test ---\n")
    
    # 1. Setup Synthetic Data
    # 4 Houses, 3 Features
    X = np.array([[100,2,5],[120,3,1],[90,1,8],[150,4,2]],dtype=float)
    
    # Target: Price strictly depends on House Size (Size * 10). 
    # Rooms and Stray Cats have 0 real impact.
    y = np.array([1000,1200,900,1500],dtype=float)
    
    
    # Note: We use a larger lambda and specific learning rate to trigger 
    # the feature elimination early for testing purposes.
    lambda_val = 50.0
    lr_val = 0.0001
    iters = 5000
    
    # 2. Train L2 Model (Ridge)
    print("Training Ridge (L2) Model...")
    model_l2 = RidgeRegression(learning_rate=lr_val, n_iterations=iters, lambda_param=lambda_val)
    model_l2.fit(X,y)
    
    # 3. Train L1 Model (Lasso)
    print("Training Ridge (L1) Model...")
    model_l1 = LassoRegression(learning_rate=lr_val, n_iterations=iters, lambda_param=lambda_val)
    model_l1.fit(X,y)
    
    # 4. Assertions & Output
    features = ["House Size", "Rooms", "Stray Cats"]
    
    print("\n[ L2 / RIDGE WEIGHTS ] -> Expecting shrinkage, but NO strict zeros")
    for name,weight in zip (features, model_l2.weights.flatten()):
        print(f"- {name:<12}: {weight:.6f}")
        
        
        
    print("\n[ L1 / LASSO WEIGHTS ] -> Expecting noise to be eliminated (exactly 0.0)")
    for name, weight in zip(features, model_l1.weights.flatten()):
        # Rounding slightly to handle floating point precision limits in Python
        w_rounded = round(weight, 6) 
        if w_rounded == 0.0:
             print(f"- {name:<12}: {w_rounded:.6f} [ELIMINATED]")
        else:
             print(f"- {name:<12}: {w_rounded:.6f}")
             
             
    # Explicit QA Assertions
    l1_cat_stray_weight = round(model_l1.weights[2][0], 6)
    l2_cat_stray_weight = round(model_l2.weights[2][0], 6)
    
    tolerance = 0.01
    
    assert abs(l1_cat_stray_weight) < tolerance, f"BUG: L1 weight {l1_cat_stray_weight} is not close to 0!"
    assert abs(l2_cat_stray_weight) > tolerance, f"BUG: L2 weight {l2_cat_stray_weight} should not shrink this aggressively!"
    
    
    print("\n✅ Integration Test Passed! Architecture is solid.")
    
if __name__ == "__main__":
    test_regularization_behavior()