import hashlib
import fitz  # PyMuPDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509 import load_pem_x509_certificate, load_der_x509_certificate
import base64
from typing import Dict, List, Optional, Tuple
import datetime
import re

class PDFSignatureValidator:
    def __init__(self):
        self.signature_types = {
            'digital': 'Digital signature with certificate',
            'electronic': 'Electronic signature',
            'handwritten': 'Scanned handwritten signature',
            'stamp': 'Official stamp or seal'
        }
    
    def extract_digital_signatures(self, pdf_path: str) -> Dict:
        """Extract digital signatures from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            signatures = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get form widgets (includes signature fields)
                widgets = page.widgets()
                
                for widget in widgets:
                    if widget.field_type == fitz.PDF_WIDGET_TYPE_SIGNATURE:
                        sig_info = {
                            'page': page_num,
                            'field_name': widget.field_name,
                            'rect': list(widget.rect),
                            'type': 'digital'
                        }
                        
                        # Try to get signature details
                        try:
                            sig_contents = page.get_signature(widget.field_name)
                            if sig_contents:
                                sig_info.update(self._parse_signature_contents(sig_contents))
                        except:
                            pass
                        
                        signatures.append(sig_info)
            
            doc.close()
            
            return {
                'found': len(signatures) > 0,
                'count': len(signatures),
                'signatures': signatures
            }
            
        except Exception as e:
            return {
                'found': False,
                'count': 0,
                'signatures': [],
                'error': str(e)
            }
    
    def _parse_signature_contents(self, sig_contents: Dict) -> Dict:
        """Parse signature contents to extract certificate info"""
        parsed_info = {}
        
        try:
            # Extract basic signature information
            if 'contents' in sig_contents:
                parsed_info['has_certificate'] = True
                parsed_info['signature_time'] = sig_contents.get('M', 'Unknown')
                parsed_info['reason'] = sig_contents.get('Reason', 'Not specified')
                parsed_info['location'] = sig_contents.get('Location', 'Not specified')
                parsed_info['contact_info'] = sig_contents.get('ContactInfo', 'Not provided')
                
                # Try to extract certificate details
                cert_data = sig_contents.get('Contents')
                if cert_data:
                    cert_info = self._extract_certificate_info(cert_data)
                    parsed_info.update(cert_info)
            
        except Exception as e:
            parsed_info['parse_error'] = str(e)
        
        return parsed_info
    
    def _extract_certificate_info(self, cert_data: bytes) -> Dict:
        """Extract information from certificate data"""
        cert_info = {}
        
        try:
            # Try to parse as X.509 certificate
            try:
                cert = load_der_x509_certificate(cert_data)
            except:
                cert = load_pem_x509_certificate(cert_data)
            
            # Extract certificate details
            cert_info['subject'] = str(cert.subject)
            cert_info['issuer'] = str(cert.issuer)
            cert_info['serial_number'] = str(cert.serial_number)
            cert_info['not_before'] = cert.not_valid_before.isoformat()
            cert_info['not_after'] = cert.not_valid_after.isoformat()
            
            # Check if certificate is currently valid
            now = datetime.datetime.now()
            cert_info['is_valid'] = cert.not_valid_before <= now <= cert.not_valid_after
            
            # Extract public key info
            public_key = cert.public_key()
            if isinstance(public_key, rsa.RSAPublicKey):
                cert_info['key_algorithm'] = 'RSA'
                cert_info['key_size'] = public_key.key_size
            
        except Exception as e:
            cert_info['extraction_error'] = str(e)
        
        return cert_info
    
    def detect_visual_signatures(self, pdf_path: str) -> Dict:
        """Detect visual signature patterns in PDF"""
        try:
            doc = fitz.open(pdf_path)
            visual_signatures = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get page as image for analysis
                pix = page.get_pixmap()
                img_data = pix.pil_tobytes()
                
                # Analyze image for signature-like patterns
                signature_regions = self._analyze_image_for_signatures(img_data, page_num)
                visual_signatures.extend(signature_regions)
            
            doc.close()
            
            return {
                'found': len(visual_signatures) > 0,
                'count': len(visual_signatures),
                'signatures': visual_signatures
            }
            
        except Exception as e:
            return {
                'found': False,
                'count': 0,
                'signatures': [],
                'error': str(e)
            }
    
    def _analyze_image_for_signatures(self, img_data: bytes, page_num: int) -> List[Dict]:
        """Analyze image data for signature-like patterns"""
        import cv2
        import numpy as np
        from PIL import Image
        import io
        
        signatures = []
        
        try:
            # Convert to OpenCV format
            pil_img = Image.open(io.BytesIO(img_data))
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Look for signature-like regions (typically in lower portion)
            height, width = gray.shape
            
            # Focus on bottom 30% of document for signatures
            roi = gray[int(height * 0.7):, :]
            
            # Edge detection to find handwritten elements
            edges = cv2.Canny(roi, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours that might be signatures
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                
                # Signature-like characteristics
                if 500 < area < 5000 and w > h and w > 50:  # Horizontal, reasonable size
                    signatures.append({
                        'page': page_num,
                        'type': 'visual',
                        'confidence': self._calculate_signature_confidence(contour, roi),
                        'bbox': [x, y + int(height * 0.7), w, h],
                        'area': area
                    })
        
        except Exception as e:
            pass  # Skip image analysis errors
        
        return signatures
    
    def _calculate_signature_confidence(self, contour, roi) -> float:
        """Calculate confidence that a contour represents a signature"""
        # Simple heuristic based on contour properties
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Aspect ratio (signatures tend to be wider than tall)
        aspect_ratio = w / h if h > 0 else 0
        
        # Solidity (filled area vs. convex hull)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Perimeter to area ratio (signatures tend to be more complex)
        perimeter = cv2.arcLength(contour, True)
        complexity = perimeter / area if area > 0 else 0
        
        # Calculate confidence based on heuristics
        confidence = 0.0
        
        # Prefer horizontal shapes
        if 1.5 < aspect_ratio < 6:
            confidence += 0.3
        
        # Prefer moderately complex shapes
        if 0.3 < solidity < 0.8:
            confidence += 0.3
        
        # Prefer moderate complexity
        if 0.02 < complexity < 0.1:
            confidence += 0.4
        
        return min(confidence, 1.0)
    
    def detect_text_signatures(self, text: str) -> Dict:
        """Detect text-based signature patterns"""
        signature_patterns = [
            r'(?:/s/|digitally signed by)\s*([A-Za-z\s]+)',
            r'(?:signature|signed):\s*([A-Za-z\s]+)',
            r'(?:executed|signed)\s+by\s*([A-Za-z\s]+)',
            r'([A-Za-z\s]+)\s*(?:signature|signed|executor)',
            r'(?:witness|notarized)\s+by\s*([A-Za-z\s]+)'
        ]
        
        text_signatures = []
        
        for pattern in signature_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                signatory_name = match.group(1).strip()
                if signatory_name and len(signatory_name) > 2:
                    text_signatures.append({
                        'type': 'text',
                        'signatory': signatory_name,
                        'pattern': pattern,
                        'position': match.start(),
                        'confidence': 0.7  # Text patterns are less reliable
                    })
        
        return {
            'found': len(text_signatures) > 0,
            'count': len(text_signatures),
            'signatures': text_signatures
        }
    
    def validate_signature_integrity(self, pdf_path: str) -> Dict:
        """Validate signature integrity and document tampering"""
        try:
            doc = fitz.open(pdf_path)
            
            # Calculate document hash
            doc_hash = hashlib.sha256()
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                doc_hash.update(text.encode('utf-8'))
            
            document_hash = doc_hash.hexdigest()
            
            # Check for signs of tampering
            tampering_indicators = []
            
            # Check modification date vs creation date
            metadata = doc.metadata
            creation_date = metadata.get('creationDate')
            mod_date = metadata.get('modDate')
            
            if creation_date and mod_date:
                if mod_date != creation_date:
                    tampering_indicators.append("Document modification date differs from creation date")
            
            # Check for incremental updates (might indicate modifications)
            # This is a simplified check - real implementation would be more complex
            if hasattr(doc, 'get_page_fonts'):
                font_variations = set()
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    fonts = page.get_fonts()
                    for font in fonts:
                        font_variations.add(font[3])  # Font name
                
                if len(font_variations) > 5:  # Many different fonts might indicate tampering
                    tampering_indicators.append("Unusual variety of fonts detected")
            
            doc.close()
            
            return {
                'document_hash': document_hash,
                'tampering_indicators': tampering_indicators,
                'integrity_score': 1.0 - (len(tampering_indicators) * 0.2),
                'appears_tampered': len(tampering_indicators) > 2
            }
            
        except Exception as e:
            return {
                'document_hash': None,
                'tampering_indicators': [f"Error checking integrity: {str(e)}"],
                'integrity_score': 0.0,
                'appears_tampered': True
            }
    
    def comprehensive_signature_validation(self, pdf_path: str, text: str) -> Dict:
        """Perform comprehensive signature validation"""
        # Extract digital signatures
        digital_sigs = self.extract_digital_signatures(pdf_path)
        
        # Detect visual signatures
        visual_sigs = self.detect_visual_signatures(pdf_path)
        
        # Detect text signatures
        text_sigs = self.detect_text_signatures(text)
        
        # Validate integrity
        integrity = self.validate_signature_integrity(pdf_path)
        
        # Calculate overall signature score
        total_signatures = digital_sigs['count'] + visual_sigs['count'] + text_sigs['count']
        
        # Weight different signature types
        signature_score = (
            digital_sigs['count'] * 0.5 +  # Digital signatures are most reliable
            visual_sigs['count'] * 0.3 +   # Visual signatures are moderately reliable
            text_sigs['count'] * 0.2       # Text signatures are least reliable
        )
        
        signature_score = min(signature_score, 1.0)
        
        # Overall validation
        has_signatures = total_signatures > 0
        signatures_valid = (
            has_signatures and 
            integrity['integrity_score'] > 0.6 and
            not integrity['appears_tampered']
        )
        
        return {
            'has_signatures': has_signatures,
            'signatures_valid': signatures_valid,
            'total_signature_count': total_signatures,
            'signature_score': signature_score,
            'digital_signatures': digital_sigs,
            'visual_signatures': visual_sigs,
            'text_signatures': text_sigs,
            'integrity_check': integrity,
            'confidence': signature_score * integrity['integrity_score'],
            'validation_summary': {
                'digital_count': digital_sigs['count'],
                'visual_count': visual_sigs['count'],
                'text_count': text_sigs['count'],
                'integrity_score': integrity['integrity_score'],
                'overall_valid': signatures_valid
            }
        }
