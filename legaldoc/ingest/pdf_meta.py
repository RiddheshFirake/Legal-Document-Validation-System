import PyPDF2
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import hashlib
import os

class PDFMetadataExtractor:
    def __init__(self):
        pass
    
    def extract_basic_metadata(self, pdf_path: str) -> Dict:
        """Extract basic PDF metadata"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata
                
                return {
                    'num_pages': len(pdf_reader.pages),
                    'title': metadata.get('/Title', '') if metadata else '',
                    'author': metadata.get('/Author', '') if metadata else '',
                    'subject': metadata.get('/Subject', '') if metadata else '',
                    'creator': metadata.get('/Creator', '') if metadata else '',
                    'producer': metadata.get('/Producer', '') if metadata else '',
                    'creation_date': metadata.get('/CreationDate', '') if metadata else '',
                    'modification_date': metadata.get('/ModDate', '') if metadata else '',
                    'encrypted': pdf_reader.is_encrypted
                }
        except Exception as e:
            return {'error': str(e)}
    
    def extract_advanced_metadata(self, pdf_path: str) -> Dict:
        """Extract advanced PDF metadata using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            # Check for digital signatures
            signatures = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                widgets = page.widgets()
                for widget in widgets:
                    if widget.field_type == fitz.PDF_WIDGET_TYPE_SIGNATURE:
                        signatures.append({
                            'page': page_num,
                            'rect': widget.rect,
                            'field_name': widget.field_name
                        })
            
            # Extract fonts
            fonts = set()
            for page_num in range(len(doc)):
                page = doc[page_num]
                font_list = page.get_fonts()
                for font in font_list:
                    fonts.add(font[3])  # Font name
            
            # Calculate document hash
            doc_hash = hashlib.md5()
            for page_num in range(len(doc)):
                page = doc[page_num]
                doc_hash.update(page.get_text().encode())
            
            doc.close()
            
            return {
                'metadata': metadata,
                'signatures': signatures,
                'fonts': list(fonts),
                'document_hash': doc_hash.hexdigest(),
                'has_signatures': len(signatures) > 0,
                'font_count': len(fonts)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def convert_to_images(self, pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        """Convert PDF pages to images using a robust approach."""
        print(f"Attempting to convert PDF to images: {pdf_path}")
        try:
            from pdf2image import convert_from_path
            
            # On Windows, you need to provide the path to the Poppler bin directory.
            # This is a common point of failure.
            poppler_path = os.environ.get('POPPLER_PATH')
            
            if os.name == 'nt' and poppler_path is None:
                print("Warning: Poppler path not set. `pdf2image` may fail. Trying default path...")
                # Download Poppler from here: https://github.com/oschwartz10612/poppler-windows/releases
                # Replace this with your actual poppler bin directory path.
                poppler_path = r"C:\path\to\your\poppler-windows\Library\bin" # <--- IMPORTANT: Update this path!
                if not os.path.exists(poppler_path):
                    print(f"Warning: Default Poppler path not found at '{poppler_path}'.")
                    poppler_path = None # Let pdf2image handle it if the path is invalid
            
            images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
            print(f"Successfully converted {len(images)} pages to images.")
            return [np.array(img) for img in images]
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            return []  # Return empty list on failure

    def extract_text_by_page(self, pdf_path: str) -> List[str]:
        """Extract text from each page"""
        try:
            doc = fitz.open(pdf_path)
            pages_text = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                pages_text.append(text)
            
            doc.close()
            return pages_text
        except Exception as e:
            print(f"Error extracting text: {e}")
            return []
    
    def analyze_document_structure(self, pdf_path: str) -> Dict:
        """Analyze document structure"""
        try:
            doc = fitz.open(pdf_path)
            
            # Get table of contents
            toc = doc.get_toc()
            
            # Analyze page layouts
            page_layouts = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text blocks
                blocks = page.get_text("dict")
                text_blocks = len(blocks['blocks'])
                
                # Get images
                image_list = page.get_images()
                image_count = len(image_list)
                
                page_layouts.append({
                    'page': page_num,
                    'text_blocks': text_blocks,
                    'images': image_count,
                    'rotation': page.rotation
                })
            
            doc.close()
            
            return {
                'toc': toc,
                'page_layouts': page_layouts,
                'has_toc': len(toc) > 0,
                'total_images': sum(layout['images'] for layout in page_layouts)
            }
        except Exception as e:
            return {'error': str(e)}