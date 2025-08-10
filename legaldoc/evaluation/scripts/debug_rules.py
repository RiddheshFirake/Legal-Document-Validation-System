import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orchestrator.pipeline import DocumentValidationPipeline
import yaml

def debug_rules_for_document(file_path):
    """Debug what rules are triggering for a specific document"""
    
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    pipeline = DocumentValidationPipeline(config)
    
    print(f"Debugging: {file_path}")
    print("=" * 50)
    
    # Process the document
    result = pipeline.process_document(file_path, {'return_detailed': True})
    
    if result.get('success'):
        decision = result.get('decision', {})
        component_scores = decision.get('component_scores', {})
        
        print(f"Overall Decision: {decision.get('is_legal', False)}")
        print(f"Confidence: {decision.get('confidence', 0):.3f}")
        print(f"Ensemble Score: {decision.get('ensemble_score', 0):.3f}")
        print()
        
        # Detailed rules analysis
        rules_data = component_scores.get('rules', {})
        if 'validator_results' in rules_data:
            print("Rules Breakdown:")
            for validator, data in rules_data['validator_results'].items():
                print(f"  {validator}: score={data['score']:.3f}, confidence={data['confidence']:.3f}")
        
        print(f"Rules Overall: {rules_data.get('score', 0):.3f}")
        print()
        
        # Show validation details if available
        validation = result.get('validation', {})
        if validation:
            print("Validation Details:")
            for key, value in validation.items():
                if isinstance(value, dict) and 'valid' in value:
                    print(f"  {key}: {value.get('valid', False)}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    # Debug your legal documents
    legal_files = [
        "../dataset/legal/sample_employment_contract.txt",
        "../dataset/edge_cases/fake_legal_doc.txt"
    ]
    
    for file_path in legal_files:
        if os.path.exists(file_path):
            debug_rules_for_document(file_path)
            print("\n" + "="*50 + "\n")
