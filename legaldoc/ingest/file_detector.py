from pathlib import Path

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
    
    for file_type, extensions in SUPPORTED_FORMATS.items():
        if ext in extensions:
            return file_type
    
    return 'unknown'

def is_supported_file(file_path):
    """Check if file type is supported"""
    return detect_file_type(file_path) != 'unknown'

def get_supported_extensions():
    """Get list of all supported file extensions"""
    extensions = []
    for ext_list in SUPPORTED_FORMATS.values():
        extensions.extend(ext_list)
    return extensions

# ADD THE MISSING CLASS HERE
class FileDetector:
    """File detector class that your pipeline expects"""
    
    def __init__(self):
        self.supported_formats = SUPPORTED_FORMATS
    
    def detect_file_type(self, file_path):
        """Detect file type based on extension"""
        return detect_file_type(file_path)
    
    def is_supported(self, file_path):
        """Check if file is supported"""
        return is_supported_file(file_path)
    
    def get_supported_formats(self):
        """Get supported formats"""
        return self.supported_formats
    
    def validate_file(self, file_path):
        """Validate if file exists and is supported"""
        if not Path(file_path).exists():
            return False, "File does not exist"
        
        if not self.is_supported(file_path):
            return False, f"Unsupported file type: {detect_file_type(file_path)}"
        
        # Check file size (max 50MB)
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        if file_size_mb > 50:
            return False, f"File too large: {file_size_mb:.1f}MB (max 50MB)"
        
        return True, "File is valid"
