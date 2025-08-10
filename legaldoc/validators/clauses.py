import re
from typing import Dict, List, Optional, Set
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class LegalClauseValidator:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        
        # Essential legal clause patterns
        self.essential_clauses = {
            'parties': {
                'patterns': [
                    r'(?:this\s+agreement|contract|document)\s+is\s+(?:made|entered\s+into)',
                    r'(?:party|parties)\s+(?:to\s+this|hereby|herein)',
                    r'between\s+.*?\s+and\s+.*?',
                    r'(?:licensor|licensee|contractor|client)'
                ],
                'required': True,
                'weight': 0.2
            },
            'consideration': {
                'patterns': [
                    r'(?:in\s+consideration\s+of|for\s+valuable\s+consideration)',
                    r'payment\s+of\s+.*?(?:\$|\d+)',
                    r'(?:fee|sum|amount).*?(?:paid|payable)',
                    r'(?:compensation|remuneration)'
                ],
                'required': True,
                'weight': 0.15
            },
            'obligations': {
                'patterns': [
                    r'(?:shall|will|must|agrees?\s+to)\s+(?:provide|deliver|perform)',
                    r'(?:obligation|duty|responsibility)\s+(?:of|to)',
                    r'(?:undertakes?|covenants?)\s+(?:to|that)',
                    r'(?:terms\s+and\s+conditions|provisions)'
                ],
                'required': True,
                'weight': 0.15
            },
            'termination': {
                'patterns': [
                    r'(?:termination|terminate|expires?)\s+(?:of|on|upon)',
                    r'(?:end|conclusion)\s+of\s+(?:this|the)',
                    r'(?:breach|default|violation)\s+of\s+(?:this|any)',
                    r'(?:notice\s+of\s+termination|thirty\s+days?\s+notice)'
                ],
                'required': False,
                'weight': 0.1
            },
            'governing_law': {
                'patterns': [
                    r'(?:governed\s+by|subject\s+to)\s+(?:the\s+laws?\s+of|.*?\s+law)',
                    r'(?:jurisdiction|venue)\s+(?:of|in|shall\s+be)',
                    r'(?:courts?\s+of|dispute\s+resolution)',
                    r'(?:applicable\s+law|governing\s+jurisdiction)'
                ],
                'required': False,
                'weight': 0.1
            },
            'signatures': {
                'patterns': [
                    r'(?:signed|executed|witnessed)\s+(?:by|on|this)',
                    r'(?:signature|sign|executed)\s+(?:of|by)',
                    r'(?:in\s+witness\s+whereof|witness\s+our\s+hands)',
                    r'(?:date|dated)\s+(?:this|as\s+of)'
                ],
                'required': True,
                'weight': 0.1
            },
            'definitions': {
                'patterns': [
                    r'(?:as\s+used\s+herein|for\s+purposes\s+of\s+this)',
                    r'(?:means|shall\s+mean|defined\s+as)',
                    r'(?:definition|definitions)\s+(?:section|clause)',
                    r'(?:hereinafter\s+referred\s+to\s+as|shall\s+be\s+referred\s+to)'
                ],
                'required': False,
                'weight': 0.08
            },
            'warranties': {
                'patterns': [
                    r'(?:warranty|warranties|warrants?)\s+(?:that|of)',
                    r'(?:represents?\s+and\s+warrants?|representation)',
                    r'(?:guarantee|guarantees)\s+(?:that|the)',
                    r'(?:disclaimer\s+of\s+warranties|without\s+warranty)'
                ],
                'required': False,
                'weight': 0.07
            },
            'liability': {
                'patterns': [
                    r'(?:liability|liable|responsible)\s+(?:for|to|of)',
                    r'(?:limitation\s+of\s+liability|damages)',
                    r'(?:indemnif|hold\s+harmless)',
                    r'(?:consequential|incidental|punitive)\s+damages'
                ],
                'required': False,
                'weight': 0.05
            }
        }
        
        # Legal document indicators
        self.legal_indicators = {
            'formal_language': [
                'whereas', 'hereby', 'herein', 'hereof', 'hereunder', 'heretofore',
                'aforementioned', 'aforesaid', 'pursuant', 'notwithstanding'
            ],
            'legal_terms': [
                'agreement', 'contract', 'covenant', 'clause', 'provision', 'article',
                'section', 'subsection', 'paragraph', 'subparagraph'
            ],
            'modal_verbs': [
                'shall', 'will', 'must', 'may', 'should', 'ought'
            ]
        }
    
    def extract_clauses(self, text: str) -> Dict:
        """Extract and analyze legal clauses from text"""
        sentences = sent_tokenize(text.lower())
        clause_results = {}
        
        for clause_type, clause_info in self.essential_clauses.items():
            matches = []
            
            for pattern in clause_info['patterns']:
                for sentence in sentences:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        matches.append({
                            'sentence': sentence.strip(),
                            'pattern': pattern,
                            'position': sentences.index(sentence)
                        })
            
            clause_results[clause_type] = {
                'found': len(matches) > 0,
                'count': len(matches),
                'matches': matches,
                'required': clause_info['required'],
                'weight': clause_info['weight']
            }
        
        return clause_results
    
    def calculate_clause_score(self, clause_results: Dict) -> Dict:
        """Calculate overall clause compliance score"""
        total_weight = 0
        achieved_weight = 0
        required_missing = []
        
        for clause_type, result in clause_results.items():
            weight = result['weight']
            total_weight += weight
            
            if result['found']:
                achieved_weight += weight
            elif result['required']:
                required_missing.append(clause_type)
        
        compliance_score = achieved_weight / total_weight if total_weight > 0 else 0
        
        return {
            'compliance_score': compliance_score,
            'total_clauses_found': sum(1 for r in clause_results.values() if r['found']),
            'required_clauses_found': sum(1 for r in clause_results.values() 
                                        if r['found'] and r['required']),
            'total_required_clauses': sum(1 for r in clause_results.values() if r['required']),
            'missing_required_clauses': required_missing,
            'is_compliant': len(required_missing) == 0 and compliance_score > 0.6
        }
    
    def analyze_legal_language(self, text: str) -> Dict:
        """Analyze the presence of legal language patterns"""
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        
        # Remove stopwords for analysis
        words_filtered = [w for w in words if w not in self.stop_words and w.isalpha()]
        total_words = len(words_filtered)
        
        language_analysis = {}
        
        for category, term_list in self.legal_indicators.items():
            found_terms = [term for term in term_list if term in text_lower]
            term_count = sum(text_lower.count(term) for term in found_terms)
            
            language_analysis[category] = {
                'terms_found': found_terms,
                'count': term_count,
                'density': term_count / total_words if total_words > 0 else 0
            }
        
        # Overall legal language density
        total_legal_terms = sum(analysis['count'] for analysis in language_analysis.values())
        overall_density = total_legal_terms / total_words if total_words > 0 else 0
        
        return {
            'category_analysis': language_analysis,
            'overall_legal_density': overall_density,
            'total_legal_terms': total_legal_terms,
            'total_words': total_words,
            'is_formal_legal_language': overall_density > 0.02  # 2% threshold
        }
    
    def validate_document_structure(self, text: str) -> Dict:
        """Validate document structure and formatting"""
        lines = text.split('\n')
        sentences = sent_tokenize(text)
        
        # Check for common document sections
        sections = {
            'title_section': any(re.search(r'^(?:agreement|contract|memorandum)', line.strip().lower()) 
                               for line in lines[:5]),
            'date_section': bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)),
            'signature_section': any('signature' in line.lower() or 'signed' in line.lower() 
                                   for line in lines[-10:]),
            'numbered_sections': bool(re.search(r'\b(?:section|article|clause)\s+\d+', text.lower())),
            'proper_formatting': len([l for l in lines if l.strip()]) > len(lines) * 0.7
        }
        
        structure_score = sum(sections.values()) / len(sections)
        
        return {
            'sections_found': sections,
            'structure_score': structure_score,
            'total_lines': len(lines),
            'total_sentences': len(sentences),
            'avg_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
            'is_well_structured': structure_score > 0.6
        }
    
    def comprehensive_clause_validation(self, text: str) -> Dict:
        """Perform comprehensive legal clause validation"""
        # Extract clauses
        clause_results = self.extract_clauses(text)
        
        # Calculate compliance score
        compliance = self.calculate_clause_score(clause_results)
        
        # Analyze legal language
        language_analysis = self.analyze_legal_language(text)
        
        # Validate structure
        structure_validation = self.validate_document_structure(text)
        
        # Overall assessment
        overall_score = (
            compliance['compliance_score'] * 0.5 +
            min(language_analysis['overall_legal_density'] * 50, 1.0) * 0.3 +
            structure_validation['structure_score'] * 0.2
        )
        
        is_valid_legal_document = (
            compliance['is_compliant'] and
            language_analysis['is_formal_legal_language'] and
            structure_validation['is_well_structured'] and
            overall_score > 0.7
        )
        
        return {
            'clause_analysis': clause_results,
            'compliance': compliance,
            'language_analysis': language_analysis,
            'structure_validation': structure_validation,
            'overall_score': overall_score,
            'is_valid_legal_document': is_valid_legal_document,
            'confidence': min(overall_score * 1.2, 1.0),
            'recommendations': self._generate_recommendations(compliance, language_analysis, structure_validation)
        }
    
    def _generate_recommendations(self, compliance: Dict, language: Dict, structure: Dict) -> List[str]:
        """Generate recommendations for improving document legality"""
        recommendations = []
        
        if compliance['missing_required_clauses']:
            recommendations.append(f"Add missing required clauses: {', '.join(compliance['missing_required_clauses'])}")
        
        if not language['is_formal_legal_language']:
            recommendations.append("Use more formal legal language and terminology")
        
        if not structure['is_well_structured']:
            recommendations.append("Improve document structure with clear sections and numbering")
        
        if compliance['compliance_score'] < 0.8:
            recommendations.append("Include more comprehensive legal provisions")
        
        return recommendations
