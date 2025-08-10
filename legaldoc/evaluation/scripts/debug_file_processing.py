import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path
from ingest.file_detector import detect_file_type
from ingest.text_extractor import extract_text

def test_file_processing():
    """Test file processing for your sample files"""
    
    test_files = [
        "../dataset/legal/sample_employment_contract.txt",
        "../dataset/non_legal/sample_recipe.txt", 
        "../dataset/legal/CUAD_v1_README.txt"
    ]
    
    print("🔍 FILE PROCESSING DEBUG")
    print("=" * 50)
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n📄 Testing: {os.path.basename(file_path)}")
            
            # Test file type detection
            file_type = detect_file_type(file_path)
            print(f"   Detected type: {file_type}")
            
            # Test file size
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"   File size: {file_size:.2f} MB")
            
            # Test text extraction
            try:
                text = extract_text(file_path)
                text_length = len(text) if text else 0
                print(f"   Text extracted: {text_length} characters")
                if text_length > 0:
                    print(f"   First 100 chars: {text[:100]}...")
                else:
                    print(f"   ❌ No text extracted!")
            except Exception as e:
                print(f"   ❌ Text extraction error: {e}")
        else:
            print(f"❌ File not found: {file_path}")

if __name__ == "__main__":
    test_file_processing()
