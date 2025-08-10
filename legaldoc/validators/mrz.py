import re
import hashlib
from typing import Dict, Optional, List, Tuple
import string

class MRZValidator:
    def __init__(self):
        # MRZ character mapping for check digit calculation
        self.char_values = {}
        for i, char in enumerate(string.ascii_uppercase):
            self.char_values[char] = i + 10
        for i in range(10):
            self.char_values[str(i)] = i
        self.char_values['<'] = 0
        
        # Common MRZ patterns
        self.mrz_patterns = {
            'passport': {
                'line1': r'^P[A-Z0-9<]{1}[A-Z]{3}[A-Z0-9<]{39}$',
                'line2': r'^[A-Z0-9<]{9}[0-9]{1}[A-Z]{3}[0-9]{6}[0-9]{1}[MF<]{1}[0-9]{6}[0-9]{1}[A-Z0-9<]{14}[0-9]{1}$'
            },
            'id_card': {
                'line1': r'^I[A-Z0-9<]{1}[A-Z]{3}[A-Z0-9<]{9}[0-9]{1}[A-Z0-9<]{15}$',
                'line2': r'^[0-9]{6}[0-9]{1}[MF<]{1}[0-9]{6}[0-9]{1}[A-Z]{3}[A-Z0-9<]{11}[0-9]{1}$',
                'line3': r'^[A-Z0-9<]{30}$'
            },
            'visa': {
                'line1': r'^V[A-Z0-9<]{1}[A-Z]{3}[A-Z0-9<]{39}$',
                'line2': r'^[A-Z0-9<]{9}[0-9]{1}[A-Z]{3}[0-9]{6}[0-9]{1}[MF<]{1}[0-9]{6}[0-9]{1}[A-Z0-9<]{8}$'
            }
        }
    
    def calculate_check_digit(self, data: str) -> int:
        """Calculate MRZ check digit"""
        weights = [7, 3, 1] * (len(data) // 3 + 1)
        total = 0
        
        for i, char in enumerate(data.upper()):
            if char in self.char_values:
                total += self.char_values[char] * weights[i]
        
        return total % 10
    
    def validate_check_digits(self, mrz_lines: List[str], document_type: str) -> Dict:
        """Validate check digits in MRZ"""
        if document_type not in self.mrz_patterns:
            return {'valid': False, 'error': 'Unsupported document type'}
        
        check_results = {}
        
        if document_type == 'passport':
            if len(mrz_lines) != 2:
                return {'valid': False, 'error': 'Invalid number of MRZ lines for passport'}
            
            line2 = mrz_lines[1]
            
            # Check document number check digit
            doc_number = line2[0:9]
            doc_check = int(line2[9])
            calc_doc_check = self.calculate_check_digit(doc_number)
            check_results['document_number'] = doc_check == calc_doc_check
            
            # Check birth date check digit
            birth_date = line2[13:19]
            birth_check = int(line2[19])
            calc_birth_check = self.calculate_check_digit(birth_date)
            check_results['birth_date'] = birth_check == calc_birth_check
            
            # Check expiry date check digit
            expiry_date = line2[21:27]
            expiry_check = int(line2[27])
            calc_expiry_check = self.calculate_check_digit(expiry_date)
            check_results['expiry_date'] = expiry_check == calc_expiry_check
            
            # Check personal number check digit
            personal_number = line2[28:42]
            personal_check = int(line2[42])
            calc_personal_check = self.calculate_check_digit(personal_number)
            check_results['personal_number'] = personal_check == calc_personal_check
            
            # Check composite check digit
            composite_data = doc_number + line2[9] + birth_date + line2[19] + expiry_date + line2[27] + personal_number + line2[42]
            composite_check = int(line2[43])
            calc_composite_check = self.calculate_check_digit(composite_data)
            check_results['composite'] = composite_check == calc_composite_check
        
        all_valid = all(check_results.values())
        
        return {
            'valid': all_valid,
            'check_results': check_results,
            'document_type': document_type
        }
    
    def extract_mrz_data(self, mrz_lines: List[str], document_type: str) -> Dict:
        """Extract structured data from MRZ"""
        if document_type not in self.mrz_patterns:
            return {'error': 'Unsupported document type'}
        
        data = {}
        
        if document_type == 'passport':
            if len(mrz_lines) != 2:
                return {'error': 'Invalid number of MRZ lines for passport'}
            
            line1, line2 = mrz_lines
            
            # Extract from line 1
            data['document_type'] = line1[0]
            data['issuing_country'] = line1[2:5].replace('<', '')
            data['surname'] = line1[5:].split('<<')[0].replace('<', ' ').strip()
            
            if '<<' in line1[5:]:
                given_names_part = line1[5:].split('<<')[1] if len(line1[5:].split('<<')) > 1 else ''
                data['given_names'] = given_names_part.replace('<', ' ').strip()
            else:
                data['given_names'] = ''
            
            # Extract from line 2
            data['document_number'] = line2[0:9].replace('<', '')
            data['nationality'] = line2[10:13].replace('<', '')
            data['birth_date'] = self.format_date(line2[13:19])
            data['sex'] = line2[20]
            data['expiry_date'] = self.format_date(line2[21:27])
            data['personal_number'] = line2[28:42].replace('<', '')
        
        return data
    
    def format_date(self, date_str: str) -> str:
        """Format MRZ date (YYMMDD) to readable format"""
        if len(date_str) != 6 or not date_str.isdigit():
            return date_str
        
        year = int(date_str[:2])
        month = date_str[2:4]
        day = date_str[4:6]
        
        # Assume years 00-30 are 2000-2030, 31-99 are 1931-1999
        if year <= 30:
            full_year = 2000 + year
        else:
            full_year = 1900 + year
        
        return f"{full_year}-{month}-{day}"
    
    def detect_mrz_lines(self, text: str) -> List[str]:
        """Detect potential MRZ lines in text"""
        lines = text.strip().split('\n')
        mrz_lines = []
        
        for line in lines:
            # Clean line
            cleaned_line = re.sub(r'[^A-Z0-9<]', '', line.upper())
            
            # Check if line matches MRZ pattern lengths
            if len(cleaned_line) in [30, 36, 44]:  # Common MRZ line lengths
                # Check if line contains typical MRZ characters
                if re.search(r'[A-Z0-9<]{20,}', cleaned_line):
                    mrz_lines.append(cleaned_line)
        
        return mrz_lines
    
    def validate_mrz_structure(self, mrz_lines: List[str]) -> Dict:
        """Validate MRZ structure and determine document type"""
        if not mrz_lines:
            return {'valid': False, 'error': 'No MRZ lines found'}
        
        results = []
        
        for doc_type, patterns in self.mrz_patterns.items():
            if len(mrz_lines) == len(patterns):
                pattern_match = True
                for i, (line_name, pattern) in enumerate(patterns.items()):
                    if not re.match(pattern, mrz_lines[i]):
                        pattern_match = False
                        break
                
                if pattern_match:
                    # Validate check digits
                    check_validation = self.validate_check_digits(mrz_lines, doc_type)
                    
                    # Extract data
                    extracted_data = self.extract_mrz_data(mrz_lines, doc_type)
                    
                    results.append({
                        'document_type': doc_type,
                        'structure_valid': True,
                        'check_digits_valid': check_validation['valid'],
                        'extracted_data': extracted_data,
                        'confidence': 0.9 if check_validation['valid'] else 0.6
                    })
        
                if results:
                    # Return best match (highest confidence)
                    best_result = max(results, key=lambda x: x['confidence'])
                    return {
                        'valid': True,
                        'document_type': best_result['document_type'],
                        'structure_valid': best_result['structure_valid'],
                        'check_digits_valid': best_result['check_digits_valid'],
                        'extracted_data': best_result['extracted_data'],
                        'confidence': best_result['confidence']
                    }
                else:
                    return {'valid': False, 'error': 'No valid MRZ structure detected'}
            
            def comprehensive_mrz_validation(self, text: str) -> Dict:
                """Perform comprehensive MRZ validation"""
                # Detect MRZ lines
                mrz_lines = self.detect_mrz_lines(text)
                
                if not mrz_lines:
                    return {
                        'valid': False,
                        'has_mrz': False,
                        'error': 'No MRZ detected in text'
                    }
                
                # Validate structure
                validation_result = self.validate_mrz_structure(mrz_lines)
                
                return {
                    'valid': validation_result['valid'],
                    'has_mrz': True,
                    'mrz_lines': mrz_lines,
                    'validation_result': validation_result
                }

