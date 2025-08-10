import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import minimize

def optimize_ensemble(results_csv):
    """Optimize ensemble weights and threshold"""
    df = pd.read_csv(results_csv)
    
    def objective(params):
        weights = params[:3]  # rules, nlp, vision
        threshold = params[3]
        
        # Recalculate ensemble scores
        ensemble_scores = (
            df['rules_score'] * weights[0] +
            df['nlp_score'] * weights[1] +
            df['vision_score'] * weights[2]
        )
        
        predictions = (ensemble_scores > threshold).astype(int)
        f1 = f1_score(df['true_label'], predictions, zero_division=0)
        return -f1  # Minimize negative F1
    
    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 1}]
    bounds = [(0.1, 0.8), (0.1, 0.8), (0.1, 0.8), (0.3, 0.7)]
    
    result = minimize(objective, x0=[0.4, 0.3, 0.3, 0.5], 
                     bounds=bounds, constraints=constraints)
    
    return {
        'optimal_weights': {'rules': result.x[0], 'nlp': result.x[1], 'vision': result.x[2]},
        'optimal_threshold': result.x[3],
        'expected_f1': -result.fun
    }

if __name__ == "__main__":
    results = optimize_ensemble("../results/detailed_results.csv")
    print("Optimization Results:")
    print(f"Optimal weights: {results['optimal_weights']}")
    print(f"Optimal threshold: {results['optimal_threshold']:.3f}")
    print(f"Expected F1 score: {results['expected_f1']:.3f}")
