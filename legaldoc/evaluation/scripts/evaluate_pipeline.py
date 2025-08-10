import os, json, time, argparse, csv
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from orchestrator.pipeline import DocumentValidationPipeline
import yaml

def load_config():
    """Load the project configuration from config.yaml."""
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_system(manifest_csv_path: str):
    """
    Main evaluation function that runs the pipeline on a dataset
    and calculates performance metrics.
    """
    print("Initializing pipeline...")
    config = load_config()
    pipeline = DocumentValidationPipeline(config)
    
    df = pd.read_csv(manifest_csv_path)
    results = []
    
    print(f"Evaluating {len(df)} documents...")
    
    for idx, row in df.iterrows():
        file_path = os.path.join(Path(manifest_csv_path).parent, row['file_path'])
        true_label = 1 if row['label'].lower() == 'legal' else 0
        
        print(f"Processing {idx+1}/{len(df)}: {Path(file_path).name}")
        
        t0 = time.time()
        try:
            result = pipeline.process_document(file_path, options={'return_detailed': True})
            processing_time = time.time() - t0
            
            # --- CORRECTED LOGIC TO HANDLE TUPLE/DICT MISMATCH ---
            if isinstance(result, dict) and result.get('success'):
                decision = result.get('decision', {})
                predicted_label = 1 if decision.get('is_legal', False) else 0
                confidence = decision.get('confidence', 0.0)
                ensemble_score = decision.get('ensemble_score', 0.0)
                rules_score = decision.get('component_scores', {}).get('rules', {}).get('score', 0.0)
                nlp_score = decision.get('component_scores', {}).get('nlp', {}).get('score', 0.0)
                vision_score = decision.get('component_scores', {}).get('vision', {}).get('score', 0.0)
                error_message = None
            else:
                # Handle pipeline failure or non-dict return gracefully
                predicted_label = 0
                confidence = 0.0
                ensemble_score = 0.0
                rules_score = 0.0
                nlp_score = 0.0
                vision_score = 0.0
                if isinstance(result, dict):
                    error_message = result.get('error', 'Unknown error')
                else:
                    error_message = "Pipeline returned a non-dictionary object (possible tuple)"
            # --- END OF CORRECTED LOGIC ---
                
            results.append({
                'file_path': row['file_path'],
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'ensemble_score': ensemble_score,
                'rules_score': rules_score,
                'nlp_score': nlp_score,
                'vision_score': vision_score,
                'processing_time': processing_time,
                'category': row.get('category', 'unknown'),
                'error': error_message
            })

        except Exception as e:
            print(f"Error processing document {Path(file_path).name}: {e}")
            results.append({
                'file_path': row['file_path'],
                'true_label': true_label,
                'predicted_label': 0,
                'confidence': 0.0,
                'ensemble_score': 0.0,
                'rules_score': 0.0,
                'nlp_score': 0.0,
                'vision_score': 0.0,
                'processing_time': time.time() - t0,
                'category': row.get('category', 'unknown'),
                'error': str(e)
            })
    
    return pd.DataFrame(results)

def calculate_metrics(results_df: pd.DataFrame) -> Dict:
    """Calculate and return a dictionary of performance metrics."""
    if len(results_df) == 0:
        return {'overall': {'total_samples': 0}}
        
    y_true = results_df['true_label'].tolist()
    y_pred = results_df['predicted_label'].tolist()
    confidences = results_df['confidence'].tolist()
    
    accuracy = accuracy_score(y_true, y_pred)
    
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    except ValueError:
        precision, recall, f1 = 0.0, 0.0, 0.0
    
    cm = confusion_matrix(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, confidences)
    except ValueError:
        auc = 0.5
    
    return {
        'overall': {
            'total_samples': len(results_df),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc,
            'confusion_matrix': cm.tolist()
        }
    }

def print_summary(metrics: Dict):
    """Print the final evaluation summary."""
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    overall = metrics['overall']
    print(f"Total Samples: {overall.get('total_samples', 0)}")
    print(f"Accuracy: {overall.get('accuracy', 0):.3f}")
    print(f"Precision: {overall.get('precision', 0):.3f}")
    print(f"Recall: {overall.get('recall', 0):.3f}")
    print(f"F1 Score: {overall.get('f1_score', 0):.3f}")
    print(f"ROC AUC: {overall.get('roc_auc', 0):.3f}")
    print("-" * 50)
    cm = overall.get('confusion_matrix', [])
    if len(cm) == 2:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        print(f"False Positives (Predicted Legal, Actually Not Legal): {fp}")
        print(f"False Negatives (Predicted Not Legal, Actually Legal): {fn}")
    else:
        print("False Positives: N/A")
        print("False Negatives: N/A")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the legal document validation pipeline.')
    parser.add_argument('--manifest', type=str, default='../dataset_manifest.csv', help='Path to the dataset manifest CSV file.')
    args = parser.parse_args()
    
    manifest_path = Path(__file__).parent / args.manifest
    
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at {manifest_path}")
        exit(1)

    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    results_df = evaluate_system(str(manifest_path))
    
    results_df.to_csv(results_dir / 'detailed_results.csv', index=False)
    
    metrics = calculate_metrics(results_df)
    print_summary(metrics)
    
    with open(results_dir / 'evaluation_report.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to: {results_dir}")
    print("- detailed_results.csv: Per-document results")
    print("- evaluation_report.json: Complete metrics and analysis")

