import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ingest.text_extractor import extract_text, detect_file_type

def test_extraction():
    """Test text extraction on your sample files"""
    
    test_files = [
        "../dataset/legal/sample_employment_contract.txt",
        "../dataset/non_legal/sample_recipe.txt",
        "../dataset/legal/CUAD_v1_README.txt"
    ]
    
    print("🧪 TESTING TEXT EXTRACTION")
    print("=" * 50)
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n📄 Testing: {os.path.basename(file_path)}")
            
            # Test file type detection
            file_type = detect_file_type(file_path)
            print(f"   File type: {file_type}")
            
            # Test text extraction
            text = extract_text(file_path)
            
            if text:
                print(f"   ✅ Text extracted: {len(text)} characters")
                print(f"   Preview: {text[:100]}...")
            else:
                print(f"   ❌ Failed to extract text")
        else:
            print(f"❌ File not found: {file_path}")

if __name__ == "__main__":
    test_extraction()
