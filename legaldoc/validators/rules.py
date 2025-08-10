import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import hashlib
from pathlib import Path

class RulesValidator:
    """
    A comprehensive, rule-based validator for legal documents.
    This class performs multiple checks on a document's text and metadata
    to determine its legality, consistency, and authenticity.
    """
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Define the set of rules, their corresponding checking functions, and weights.
        self.validation_rules = {
            'document_age': {
                'max_age_days': 365 * 10,  # 10 years
                'check_function': self._check_document_age,
                'weight': 0.1
            },
            'signature_presence': {
                'check_function': self._check_signatures,
                'weight': 0.2
            },
            'date_consistency': {
                'check_function': self._check_date_consistency,
                'weight': 0.15
            },
            'legal_formatting': {
                'check_function': self._check_legal_formatting,
                'weight': 0.1
            },
            'completeness': {
                'check_function': self._check_document_completeness,
                'weight': 0.15
            },
            'authenticity_markers': {
                'check_function': self._check_authenticity_markers,
                'weight': 0.1
            },
            'content_consistency': {
                'check_function': self._check_content_consistency,
                'weight': 0.1
            },
            'legal_language': {
                'check_function': self._check_legal_language_quality,
                'weight': 0.1
            }
        }
        
        # Legal keywords for basic legal document detection
        self.legal_keywords = [
            'agreement', 'contract', 'party', 'parties', 'terms', 'conditions',
            'shall', 'hereby', 'whereas', 'therefore', 'consideration',
            'agrees to', 'undertakes', 'covenants', 'represents', 'warrants',
            'binds', 'obligations', 'rights', 'duties',
            'governed by', 'jurisdiction', 'effective date', 'termination',
            'confidential', 'proprietary', 'intellectual property',
            'signature', 'signed', 'executed', 'witness', 'notary',
            'effective as of', 'in witness whereof'
        ]
        
        # Suspicious patterns that might indicate fraud
        self.suspicious_patterns = [
            r'(?:fake|forged|copied|duplicate)\s+(?:document|signature)',
            r'(?:not\s+original|reproduction|photocopy)',
            r'(?:invalid|expired|revoked|cancelled)',
            r'(?:tampered|altered|modified)\s+(?:document|content)',
        ]
        
        # Required elements for different document types
        self.document_requirements = {
            'contract': ['parties', 'obligations', 'consideration', 'signatures'],
            'agreement': ['parties', 'terms', 'signatures', 'date'],
            'license': ['licensor', 'licensee', 'terms', 'duration'],
            'certificate': ['issuing_authority', 'subject', 'date', 'seal_signature'],
            'legal_notice': ['recipient', 'sender', 'legal_basis', 'date']
        }

    def validate(self, text: str, metadata: Dict = None) -> Dict:
        """
        Main validation method that the pipeline expects.
        It runs all document rules, calculates an overall score, and returns
        a final result with confidence.
        
        Args:
            text (str): The extracted text from the document.
            metadata (Dict): Extracted metadata about the document.
            
        Returns:
            Dict: A dictionary containing the final score, confidence, and detailed results.
        """
        if metadata is None:
            metadata = {}
            
        # --- CRITICAL FIX: Add a check for empty text input ---
        if not text or not text.strip():
            return {
                'score': 0.5,
                'confidence': 0.0,
                'valid': False,
                'detailed_results': {'summary': 'No text content found to validate.'}
            }
            
        # Run comprehensive rule validation
        detailed_results = self.validate_document_rules(text, metadata)
        
        # Calculate a basic legal keyword score for a quick, secondary assessment
        legal_keyword_score = self._calculate_legal_keyword_score(text)
        
        # Combine scores from various checks into one overall score for this validator
        # A simple weighted average can be used here.
        rules_score = detailed_results['overall_score']
        combined_score = (rules_score * 0.7 + legal_keyword_score * 0.3)
        
        # Calculate confidence based on the consistency of the results
        confidence = self._calculate_confidence(detailed_results, legal_keyword_score)
        
        return {
            'score': combined_score,
            'confidence': confidence,
            'valid': combined_score > 0.5,
            'detailed_results': detailed_results
        }

    def _calculate_legal_keyword_score(self, text: str) -> float:
        """Calculate a raw score based on the density of legal keywords found in the text."""
        if not text:
            return 0.0
            
        text_lower = text.lower()
        found_keywords = sum(1 for keyword in self.legal_keywords if keyword in text_lower)
        total_keywords = len(self.legal_keywords)
        
        # Normalize score based on document length and keyword density
        text_words = len(text.split())
        if text_words == 0:
            return 0.0

        keyword_density = found_keywords / (text_words / 100) # Keywords per 100 words
        
        # Cap the score to a reasonable maximum
        return min(keyword_density / 5, 1.0)

    def _calculate_confidence(self, detailed_results: Dict, legal_score: float) -> float:
        """Calculate a confidence score based on the consistency of individual rules."""
        rules_score = detailed_results['overall_score']
        passed_ratio = detailed_results['passed_rules'] / detailed_results['total_rules']
        
        # High confidence when the overall score, passed ratio, and legal keyword score are consistent
        score_consistency = 1.0 - abs(rules_score - legal_score)
        
        # Combine different confidence signals
        confidence = (passed_ratio * 0.5 + score_consistency * 0.5)
        
        return min(confidence, 0.95) # Cap confidence at 95%
    
    def _check_document_age(self, text: str, metadata: Dict) -> Dict:
        """Check if document is within acceptable age limits."""
        # Extract dates from document
        date_patterns = [
            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b',
            r'\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        dates_found = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates_found.extend(matches)
        
        # Check metadata dates if available
        creation_date = metadata.get('creation_date')
        
        oldest_date = None
        if creation_date:
            try:
                oldest_date = datetime.strptime(str(creation_date)[:10], '%Y-%m-%d')
            except:
                pass
        
        # Calculate age
        current_date = datetime.now()
        max_age = timedelta(days=self.validation_rules['document_age']['max_age_days'])
        
        age_valid = True
        age_days = 0
        
        if oldest_date:
            age_days = (current_date - oldest_date).days
            age_valid = age_days <= max_age.days
        
        return {
            'valid': age_valid,
            'age_days': age_days,
            'dates_found': len(dates_found),
            'oldest_date': oldest_date.isoformat() if oldest_date else None,
            'details': 'Document age is within acceptable limits' if age_valid else 'Document may be too old'
        }
    
    def _check_signatures(self, text: str, metadata: Dict) -> Dict:
        """Check for presence and validity of signatures."""
        signature_indicators = [
            r'(?:signature|signed|executed)\s*:',
            r'/s/\s*[A-Za-z\s]+',
            r'(?:digitally\s+signed|electronic\s+signature)',
            r'(?:witness|notarized)\s+(?:by|signature)',
            r'(?:seal|stamp)\s+(?:of|affixed)',
            r'(?:date|dated)\s*:\s*\d'
        ]
        
        signature_count = 0
        signature_types = []
        
        for pattern in signature_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                signature_count += len(matches)
                signature_types.append(pattern)
        
        # Check for digital signature indicators in metadata
        has_digital_signature = metadata.get('has_signatures', False)
        
        # Basic validation
        has_signatures = signature_count > 0 or has_digital_signature
        sufficient_signatures = signature_count >= 2  # At least two parties
        
        return {
            'valid': has_signatures,
            'signature_count': signature_count,
            'has_digital_signature': has_digital_signature,
            'sufficient_signatures': sufficient_signatures,
            'signature_types': signature_types,
            'details': f'Found {signature_count} signature indicators' if has_signatures else 'No signatures detected'
        }
    
    def _check_date_consistency(self, text: str, metadata: Dict) -> Dict:
        """Check for date consistency throughout the document."""
        # Extract all dates
        date_pattern = r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b'
        dates = re.findall(date_pattern, text)
        
        # Convert to datetime objects
        parsed_dates = []
        for match in dates:
            try:
                day, month, year = match
                if len(year) == 2:
                    year = '20' + year if int(year) < 50 else '19' + year
                
                date_obj = datetime(int(year), int(month), int(day))
                parsed_dates.append(date_obj)
            except ValueError:
                continue
        
        # Check consistency
        date_issues = []
        
        if parsed_dates:
            # Check for future dates
            current_date = datetime.now()
            future_dates = [d for d in parsed_dates if d > current_date + timedelta(days=30)]
            if future_dates:
                date_issues.append("Contains dates far in the future")
            
            # Check for very old dates (unless it's a historical document)
            very_old_dates = [d for d in parsed_dates if d < datetime(1900, 1, 1)]
            if very_old_dates:
                date_issues.append("Contains suspiciously old dates")
            
            # Check date order for execution dates
            sorted_dates = sorted(parsed_dates)
            if len(sorted_dates) > 1:
                date_range = (sorted_dates[-1] - sorted_dates[0]).days
                if date_range > 365:  # More than a year difference
                    date_issues.append("Large time span between dates")
        
        return {
            'valid': len(date_issues) == 0,
            'dates_found': len(parsed_dates),
            'issues': date_issues,
            'date_range_days': (max(parsed_dates) - min(parsed_dates)).days if len(parsed_dates) > 1 else 0,
            'details': 'Date consistency check passed' if not date_issues else '; '.join(date_issues)
        }
    
    def _check_legal_formatting(self, text: str, metadata: Dict) -> Dict:
        """Check for proper legal document formatting."""
        formatting_checks = {
            'has_title': bool(re.search(r'^(?:agreement|contract|license|certificate)', text.strip(), re.IGNORECASE | re.MULTILINE)),
            'has_sections': bool(re.search(r'(?:section|article|clause)\s+\d+', text, re.IGNORECASE)),
            'has_numbering': bool(re.search(r'\b\d+\.\d+|\(\d+\)|\d+\)', text)),
            'proper_capitalization': self._check_capitalization(text),
            'consistent_spacing': self._check_spacing_consistency(text),
            'paragraph_structure': self._check_paragraph_structure(text)
        }
        
        formatting_score = sum(formatting_checks.values()) / len(formatting_checks)
        
        return {
            'valid': formatting_score > 0.6,
            'formatting_score': formatting_score,
            'checks': formatting_checks,
            'details': f'Formatting score: {formatting_score:.2f}/1.0'
        }
    
    def _check_document_completeness(self, text: str, metadata: Dict) -> Dict:
        """Check if document appears complete and not truncated."""
        completeness_indicators = {
            'has_beginning': any(word in text.lower()[:200] for word in ['this', 'agreement', 'contract', 'whereas']),
            'has_ending': any(word in text.lower()[-200:] for word in ['signature', 'witness', 'executed', 'sealed']),
            'reasonable_length': len(text.split()) > 50,
            'no_truncation_markers': '...' not in text and '[truncated]' not in text.lower(),
            'balanced_structure': self._check_balanced_structure(text)
        }
        
        completeness_score = sum(completeness_indicators.values()) / len(completeness_indicators)
        
        return {
            'valid': completeness_score > 0.7,
            'completeness_score': completeness_score,
            'indicators': completeness_indicators,
            'word_count': len(text.split()),
            'details': f'Completeness score: {completeness_score:.2f}/1.0'
        }
    
    def _check_authenticity_markers(self, text: str, metadata: Dict) -> Dict:
        """Check for authenticity markers and suspicious content."""
        # Look for positive authenticity markers
        authenticity_markers = [
            r'(?:notarized|witnessed|sealed)',
            r'(?:official|certified|authentic)',
            r'(?:original|executed\s+original)',
            r'(?:file\s+number|case\s+number|reference\s+number)'
        ]
        
        positive_markers = sum(1 for pattern in authenticity_markers if re.search(pattern, text, re.IGNORECASE))
        
        # Look for suspicious content
        suspicious_content = sum(1 for pattern in self.suspicious_patterns if re.search(pattern, text, re.IGNORECASE))
        
        # Check metadata for authenticity indicators
        has_digital_signature = metadata.get('has_signatures', False)
        has_creator_info = bool(metadata.get('creator') or metadata.get('author'))
        
        authenticity_score = (positive_markers * 0.3 + 
                              (1 if has_digital_signature else 0) * 0.4 +
                              (1 if has_creator_info else 0) * 0.2 +
                              (0 if suspicious_content > 0 else 1) * 0.1)
        
        return {
            'valid': authenticity_score > 0.5 and suspicious_content == 0,
            'authenticity_score': min(authenticity_score, 1.0),
            'positive_markers': positive_markers,
            'suspicious_content': suspicious_content,
            'has_digital_signature': has_digital_signature,
            'details': f'Authenticity score: {authenticity_score:.2f}/1.0'
        }
    
    def _check_content_consistency(self, text: str, metadata: Dict) -> Dict:
        """Check for internal consistency in document content."""
        # Check for contradictory statements
        contradiction_patterns = [
            (r'(?:shall|will)\s+not', r'(?:shall|will)\s+(?:provide|deliver|pay)'),
            (r'(?:effective|valid)\s+(?:immediately|now)', r'(?:effective|valid)\s+(?:on|from)\s+\d'),
            (r'(?:free|without\s+charge)', r'(?:fee|cost|payment)\s+of')
        ]
        
        contradictions = []
        for neg_pattern, pos_pattern in contradiction_patterns:
            if re.search(neg_pattern, text, re.IGNORECASE) and re.search(pos_pattern, text, re.IGNORECASE):
                contradictions.append(f"Potential contradiction: {neg_pattern} vs {pos_pattern}")
        
        # Check for consistent terminology
        term_variations = self._check_term_consistency(text)
        
        consistency_score = (1.0 - len(contradictions) * 0.3 - len(term_variations) * 0.1)
        consistency_score = max(0, consistency_score)
        
        return {
            'valid': consistency_score > 0.7,
            'consistency_score': consistency_score,
            'contradictions': contradictions,
            'term_variations': term_variations,
            'details': f'Content consistency score: {consistency_score:.2f}/1.0'
        }
    
    def _check_legal_language_quality(self, text: str, metadata: Dict) -> Dict:
        """Check quality and appropriateness of legal language."""
        # Legal language quality indicators
        quality_indicators = {
            'formal_pronouns': len(re.findall(r'\b(?:herein|hereby|hereof|hereunder|thereof)\b', text, re.IGNORECASE)),
            'legal_conjunctions': len(re.findall(r'\b(?:whereas|provided that|notwithstanding|pursuant to)\b', text, re.IGNORECASE)),
            'precise_language': len(re.findall(r'\b(?:shall|must|may not|prohibited|required)\b', text, re.IGNORECASE)),
            'defined_terms': len(re.findall(r'(?:"[^"]+"|\'[^\']+\')\s+(?:means|shall mean)', text, re.IGNORECASE))
        }
        
        # Calculate language quality score
        text_length = len(text.split())
        normalized_score = sum(quality_indicators.values()) / max(text_length / 100, 1)
        quality_score = min(normalized_score, 1.0)
        
        # Check for colloquial language that might indicate non-legal document
        informal_patterns = [r'\b(?:gonna|wanna|ain\'t|yeah|ok|okay)\b']
        informal_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in informal_patterns)
        
        if informal_count > 0:
            quality_score *= 0.5  # Penalize informal language
        
        return {
            'valid': quality_score > 0.3,
            'quality_score': quality_score,
            'indicators': quality_indicators,
            'informal_language_count': informal_count,
            'details': f'Legal language quality score: {quality_score:.2f}/1.0'
        }
    
    # Helper methods
    def _check_capitalization(self, text: str) -> bool:
        """Check for proper capitalization patterns."""
        lines = text.split('\n')
        properly_capitalized_lines = 0
        
        for line in lines:
            line = line.strip()
            if line:
                # Check if line starts with capital or number
                if line[0].isupper() or line[0].isdigit():
                    properly_capitalized_lines += 1
        
        return properly_capitalized_lines / max(len([l for l in lines if l.strip()]), 1) > 0.7
    
    def _check_spacing_consistency(self, text: str) -> bool:
        """Check for consistent spacing."""
        # Check for consistent paragraph spacing
        double_newlines = text.count('\n\n')
        single_newlines = text.count('\n') - double_newlines * 2
        
        # Check for consistent sentence spacing
        double_spaces = text.count('  ')
        
        return double_spaces < len(text) * 0.01  # Less than 1% double spaces
    
    def _check_paragraph_structure(self, text: str) -> bool:
        """Check for proper paragraph structure."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return False
        
        # Check if paragraphs have reasonable length
        reasonable_length_paragraphs = [p for p in paragraphs if 20 < len(p.split()) < 500]
        
        return len(reasonable_length_paragraphs) / len(paragraphs) > 0.5
    
    def _check_balanced_structure(self, text: str) -> bool:
        """Check if document has balanced beginning, middle, and end."""
        total_length = len(text)
        third = total_length // 3
        
        beginning = text[:third]
        middle = text[third:2*third]
        end = text[2*third:]
        
        # Check if each section has reasonable content
        sections_with_content = sum(1 for section in [beginning, middle, end] 
                                    if len(section.split()) > 10)
        
        return sections_with_content >= 2
    
    def _check_term_consistency(self, text: str) -> List[str]:
        """Check for consistent use of legal terms."""
        # Common term variations that should be consistent
        term_groups = [
            ['agreement', 'contract', 'accord'],
            ['party', 'parties'],
            ['shall', 'will', 'must'],
            ['license', 'licence'],
            ['advisor', 'adviser']
        ]
        
        inconsistencies = []
        
        for group in term_groups:
            found_terms = [term for term in group if term.lower() in text.lower()]
            if len(found_terms) > 1:
                inconsistencies.append(f"Mixed usage: {', '.join(found_terms)}")
        
        return inconsistencies
    
    def validate_document_rules(self, text: str, metadata: Dict) -> Dict:
        """Perform comprehensive rule-based validation."""
        validation_results = {}
        total_weight = 0
        weighted_score = 0
        
        # Run all validation checks
        for rule_name, rule_config in self.validation_rules.items():
            try:
                result = rule_config['check_function'](text, metadata)
                validation_results[rule_name] = result
                
                weight = rule_config['weight']
                total_weight += weight
                
                if result['valid']:
                    weighted_score += weight
                    
            except Exception as e:
                validation_results[rule_name] = {
                    'valid': False,
                    'error': str(e),
                    'details': f'Error in {rule_name} validation'
                }
        
        # Calculate overall score
        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine if document passes rule validation
        passes_validation = overall_score > 0.6
        
        # Generate summary
        passed_rules = [name for name, result in validation_results.items() if result.get('valid', False)]
        failed_rules = [name for name, result in validation_results.items() if not result.get('valid', False)]
        
        return {
            'passes_validation': passes_validation,
            'overall_score': overall_score,
            'total_rules': len(self.validation_rules),
            'passed_rules': len(passed_rules),
            'failed_rules': len(failed_rules),
            'rule_results': validation_results,
            'passed_rule_names': passed_rules,
            'failed_rule_names': failed_rules,
            'confidence': min(overall_score * 1.1, 1.0),
            'summary': f"Passed {len(passed_rules)}/{len(self.validation_rules)} validation rules"
        }

# Alias for backward compatibility
DocumentRulesValidator = RulesValidator