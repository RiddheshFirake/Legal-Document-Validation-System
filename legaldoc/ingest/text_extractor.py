import os
from pathlib import Path
import pandas as pd
import fitz  # PyMuPDF imports as 'fitz'

# Define supported formats
SUPPORTED_FORMATS = {
    'pdf': ['.pdf'],
    'docx': ['.docx', '.doc'],
    'text': ['.txt', '.rtf'],
    'image': ['.jpg', '.jpeg', '.png'],
    'csv': ['.csv']
}

def detect_file_type(file_path):
    """Detect file type based on extension"""
    ext = Path(file_path).suffix.lower()
    
    type_mapping = {
        '.pdf': 'pdf',
        '.docx': 'docx', 
        '.doc': 'docx',
        '.txt': 'text',
        '.rtf': 'text',
        '.csv': 'csv',
        '.jpg': 'image',
        '.jpeg': 'image', 
        '.png': 'image'
    }
    
    return type_mapping.get(ext, 'unknown')

def extract_text_from_txt(file_path):
    """Extract text from .txt files"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read().strip()
                    if content:
                        return content
            except UnicodeDecodeError:
                continue
                
        # If all encodings fail
        with open(file_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore').strip()
            return content
            
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        return None

def extract_text_from_pdf(file_path):
    """Extract text from PDF files using PyMuPDF"""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return None

def extract_text_from_csv(file_path):
    """Extract text from CSV files"""
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 10:
            return f"CSV file too large ({file_size_mb:.1f}MB). Contains tabular data."
            
        df = pd.read_csv(file_path, nrows=100)
        text_content = "Columns: " + ", ".join(df.columns.tolist()) + "\n\n"
        text_content += "Sample data:\n" + df.head(10).to_string(index=False)
        return text_content
        
    except Exception as e:
        print(f"Error extracting text from CSV {file_path}: {e}")
        return None

def extract_text(file_path):
    """Main function to extract text from various file formats"""
    if not os.path.exists(file_path):
        return None
        
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 50:
        return f"File too large ({file_size_mb:.1f}MB)"
    
    file_type = detect_file_type(file_path)
    
    try:
        if file_type == 'text':
            return extract_text_from_txt(file_path)
        elif file_type == 'csv':
            return extract_text_from_csv(file_path)
        elif file_type == 'pdf':
            return extract_text_from_pdf(file_path)  # Now actually extracts PDF text!
        elif file_type == 'docx':
            return f"Word document detected: {Path(file_path).name}. Install python-docx for DOCX processing."
        elif file_type == 'image':
            return f"Image file detected: {Path(file_path).name}. OCR not implemented."
        else:
            return f"Unsupported file type: {file_type}"
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def is_supported_file(file_path):
    """Check if file type is supported"""
    return detect_file_type(file_path) != 'unknown'

def get_supported_extensions():
    """Get list of all supported file extensions"""
    extensions = []
    for ext_list in SUPPORTED_FORMATS.values():
        extensions.extend(ext_list)
    return extensions

# Backward compatibility aliases
def extract_text_from_file(file_path):
    """Alias for extract_text function"""
    return extract_text(file_path)

def get_file_type(file_path):
    """Alias for detect_file_type function"""
    return detect_file_type(file_path)
