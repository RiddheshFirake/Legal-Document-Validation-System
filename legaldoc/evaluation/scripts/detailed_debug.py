import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import traceback
from pathlib import Path

def debug_step_by_step():
    """Debug each step of document processing individually"""
    
    print("🔍 STEP-BY-STEP DEBUGGING")
    print("=" * 50)
    
    # Test text extraction first
    try:
        from ingest.text_extractor import extract_text
        test_file = "../dataset/legal/sample_employment_contract.txt"
        
        print("Step 1: Testing text extraction...")
        text = extract_text(test_file)
        print(f"✅ Text extracted: {len(text) if text else 0} characters")
        if text:
            print(f"Preview: {text[:100]}...")
        
    except Exception as e:
        print(f"❌ Text extraction failed: {e}")
        traceback.print_exc()
        return
    
    # Test rules validation
    try:
        print("\nStep 2: Testing rules validation...")
        from validators.rules import RulesValidator
        
        # Load config
        import yaml
        config_path = "../../configs/config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        rules_validator = RulesValidator(config.get('rules_validator', {}))
        rules_result = rules_validator.validate(text)
        
        print(f"✅ Rules validation completed")
        print(f"Result type: {type(rules_result)}")
        print(f"Result content: {rules_result}")
        
        # Check if result is tuple or dict
        if isinstance(rules_result, tuple):
            print("⚠️  FOUND THE ISSUE: Rules validator returning tuple!")
            print(f"Tuple contents: {rules_result}")
        elif isinstance(rules_result, dict):
            print("✅ Rules validator returning dict correctly")
            print(f"Dict keys: {rules_result.keys()}")
        
    except Exception as e:
        print(f"❌ Rules validation failed: {e}")
        traceback.print_exc()
        return
    
    # Test decision fusion
    try:
        print("\nStep 3: Testing decision fusion...")
        from orchestrator.decision import DecisionFusion

        # Load config
        import yaml
        config_path = "../../configs/config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        decision_fusion = DecisionFusion(config.get('ensemble', {}))

        # Correctly format mock results to match the expected structure
        combined_results_mock = {
            'validation': {
                'rules': rules_result
            },
            'analysis': {
                'nlp': {'error': 'NLP classifier not loaded - inference will be limited'},
                'vision': {'error': 'Vision model not loaded - visual analysis will be limited'}
            }
        }
        
        print(f"Mock results being sent to DecisionFusion: {combined_results_mock}")
        
        decision = decision_fusion.make_decision(combined_results_mock)
        print(f"✅ Decision fusion completed")
        print(f"Decision: {decision}")
        
    except Exception as e:
        print(f"❌ Decision fusion failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_step_by_step()