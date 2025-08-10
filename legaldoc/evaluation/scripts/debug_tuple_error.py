import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orchestrator.pipeline import DocumentValidationPipeline
import yaml
import traceback

def debug_single_document():
    """Debug processing of a single document to find the tuple error"""
    
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    pipeline = DocumentValidationPipeline(config)
    
    # Test with a simple document
    test_file = "../dataset/legal/sample_employment_contract.txt"
    
    print("🔍 DEBUGGING TUPLE ERROR")
    print("=" * 50)
    print(f"Testing file: {test_file}")
    
    try:
        # Process step by step
        result = pipeline.process_document(test_file, {})
        print("✅ Processing completed successfully")
        print(f"Result type: {type(result)}")
        print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print(f"Error type: {type(e)}")
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_single_document()
