import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load your results
df = pd.read_csv('../results/detailed_results.csv')

print("Current Scores Analysis:")
print("=" * 40)
for idx, row in df.iterrows():
    print(f"File: {row['file_path'].split('/')[-1]}")
    print(f"  True Label: {row['true_label']}")
    print(f"  Rules Score: {row['rules_score']:.3f}")
    print(f"  Ensemble Score: {row['ensemble_score']:.3f}")
    print(f"  Current Prediction: {row['predicted_label']}")
    print()

# Test different thresholds
print("Testing Different Thresholds:")
print("=" * 40)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for threshold in thresholds:
    predictions = (df['ensemble_score'] > threshold).astype(int)
    accuracy = accuracy_score(df['true_label'], predictions)
    
    # Calculate precision/recall if possible
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            df['true_label'], predictions, average='binary', zero_division=0
        )
        print(f"Threshold {threshold:.1f}: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    except:
        print(f"Threshold {threshold:.1f}: Accuracy={accuracy:.3f}")

# Find optimal threshold
best_threshold = 0.3  # Start with a lower threshold
best_predictions = (df['ensemble_score'] > best_threshold).astype(int)
best_accuracy = accuracy_score(df['true_label'], best_predictions)

print(f"\nRecommended threshold: {best_threshold}")
print(f"Expected accuracy improvement: {best_accuracy:.3f}")
