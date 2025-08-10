import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image
from typing import List, Dict, Optional
import logging

class OCRProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.easyocr_reader = easyocr.Reader(config.get('languages', ['en']))
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Thresholding
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def extract_text_easyocr(self, image: np.ndarray) -> Dict:
        """Extract text using EasyOCR"""
        try:
            preprocessed = self.preprocess_image(image)
            results = self.easyocr_reader.readtext(preprocessed)
            
            text_blocks = []
            full_text = ""
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence detections
                    text_blocks.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    full_text += text + " "
            
            return {
                'engine': 'easyocr',
                'full_text': full_text.strip(),
                'text_blocks': text_blocks,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def extract_text_tesseract(self, image: np.ndarray) -> Dict:
        """Extract text using Tesseract"""
        try:
            preprocessed = self.preprocess_image(image)
            
            # Get detailed data with bounding boxes
            data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT)
            
            text_blocks = []
            full_text = ""
            
            for i in range(len(data['text'])):
                confidence = int(data['conf'][i])
                text = data['text'][i].strip()
                
                if confidence > 30 and text:  # Filter low confidence and empty text
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    text_blocks.append({
                        'text': text,
                        'confidence': confidence / 100.0,
                        'bbox': [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                    })
                    full_text += text + " "
            
            return {
                'engine': 'tesseract',
                'full_text': full_text.strip(),
                'text_blocks': text_blocks,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def extract_text(self, image: np.ndarray, engine: str = 'easyocr') -> Dict:
        """Extract text using specified engine"""
        if engine == 'easyocr':
            return self.extract_text_easyocr(image)
        elif engine == 'tesseract':
            return self.extract_text_tesseract(image)
        else:
            # Try both engines and return best result
            easyocr_result = self.extract_text_easyocr(image)
            tesseract_result = self.extract_text_tesseract(image)
            
            # Choose result with higher average confidence
            if easyocr_result['success'] and tesseract_result['success']:
                easyocr_conf = np.mean([block['confidence'] for block in easyocr_result['text_blocks']])
                tesseract_conf = np.mean([block['confidence'] for block in tesseract_result['text_blocks']])
                
                return easyocr_result if easyocr_conf > tesseract_conf else tesseract_result
            elif easyocr_result['success']:
                return easyocr_result
            else:
                return tesseract_result
