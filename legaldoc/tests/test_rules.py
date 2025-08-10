import pytest
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validators.rules import DocumentRulesValidator

class TestDocumentRulesValidator:
    def setup_method(self):
        """Setup test fixtures"""
        self.validator = DocumentRulesValidator()
        
        # Sample legal document text
        self.sample_legal_document = """
        AGREEMENT
        
        This agreement is made between Party A and Party B on January 1, 2024.
        
        WHEREAS Party A agrees to provide services, and
        WHEREAS Party B agrees to pay compensation,
        
        NOW THEREFORE, the parties hereby agree as follows:
        
        1. Party A shall provide consulting services as described herein.
        2. Party B shall pay $10,000 upon execution of this agreement.
        3. This agreement shall terminate on December 31, 2024.
        
        IN WITNESS WHEREOF, the parties have executed this agreement.
        
        /s/ John Doe                    Date: January 1, 2024
        Party A
        
        /s/ Jane Smith                  Date: January 1, 2024
        Party B
        """
        
        # Sample non-legal document
        self.sample_non_legal_text = """
        This is just a regular document with some text.
        It doesn't contain legal language or structure.
        There are no signatures or legal clauses here.
        """
        
        # Sample metadata
        self.sample_metadata = {
            'creation_date': '2024-01-01',
            'modification_date': '2024-01-01',
            'author': 'Legal Department',
            'has_signatures': True
        }
    
    def test_document_age_validation(self):
        """Test document age validation"""
        result = self.validator._check_document_age(self.sample_legal_document, self.sample_metadata)
        
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'age_days' in result
        assert 'dates_found' in result
        assert 'details' in result
    
    def test_signature_detection(self):
        """Test signature detection"""
        result = self.validator._check_signatures(self.sample_legal_document, self.sample_metadata)
        
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'signature_count' in result
        assert 'has_digital_signature' in result
        assert result['valid'] is True  # Should find signatures in sample text
        assert result['signature_count'] > 0
    
    def test_date_consistency_check(self):
        """Test date consistency validation"""
        result = self.validator._check_date_consistency(self.sample_legal_document, self.sample_metadata)
        
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'dates_found' in result
        assert 'issues' in result
        assert 'date_range_days' in result
    
    def test_legal_formatting_check(self):
        """Test legal document formatting validation"""
        result = self.validator._check_legal_formatting(self.sample_legal_document, self.sample_metadata)
        
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'formatting_score' in result
        assert 'checks' in result
        assert result['formatting_score'] > 0  # Should have some formatting elements
    
    def test_document_completeness(self):
        """Test document completeness check"""
        result = self.validator._check_document_completeness(self.sample_legal_document, self.sample_metadata)
        
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'completeness_score' in result
        assert 'indicators' in result
        assert 'word_count' in result
        assert result['word_count'] > 50  # Sample document should have enough words
    
    def test_authenticity_markers(self):
        """Test authenticity markers detection"""
        result = self.validator._check_authenticity_markers(self.sample_legal_document, self.sample_metadata)
        
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'authenticity_score' in result
        assert 'positive_markers' in result
        assert 'suspicious_content' in result
    
    def test_content_consistency(self):
        """Test content consistency validation"""
        result = self.validator._check_content_consistency(self.sample_legal_document, self.sample_metadata)
        
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'consistency_score' in result
        assert 'contradictions' in result
        assert 'term_variations' in result
    
    def test_legal_language_quality(self):
        """Test legal language quality assessment"""
        result = self.validator._check_legal_language_quality(self.sample_legal_document, self.sample_metadata)
        
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'quality_score' in result
        assert 'indicators' in result
        assert 'informal_language_count' in result
    
    def test_comprehensive_validation(self):
        """Test comprehensive document rules validation"""
        result = self.validator.validate_document_rules(self.sample_legal_document, self.sample_metadata)
        
        assert isinstance(result, dict)
        assert 'passes_validation' in result
        assert 'overall_score' in result
        assert 'total_rules' in result
        assert 'passed_rules' in result
        assert 'failed_rules' in result
        assert 'rule_results' in result
        assert 'confidence' in result
        assert 'summary' in result
        
        # Legal document should pass most validations
        assert result['overall_score'] > 0.3
    
    def test_non_legal_document(self):
        """Test validation with non-legal document"""
        result = self.validator.validate_document_rules(self.sample_non_legal_text, {})
        
        assert isinstance(result, dict)
        assert result['overall_score'] < 0.6  # Should score lower
        assert result['passes_validation'] is False
    
    def test_helper_methods(self):
        """Test helper methods"""
        # Test capitalization check
        properly_capitalized = "This is properly capitalized text.\nEach line starts with capital."
        assert self.validator._check_capitalization(properly_capitalized) is True
        
        poorly_capitalized = "this is not properly capitalized.\nlowercase start."
        assert self.validator._check_capitalization(poorly_capitalized) is False
        
        # Test spacing consistency
        good_spacing = "This text has consistent spacing. No double spaces here."
        assert self.validator._check_spacing_consistency(good_spacing) is True
        
        bad_spacing = "This  text  has  many  double  spaces."
        assert self.validator._check_spacing_consistency(bad_spacing) is False
        
        # Test balanced structure
        balanced_text = "Beginning content " + "middle content " * 20 + "ending content"
        assert self.validator._check_balanced_structure(balanced_text) is True
        
        unbalanced_text = "Only beginning content"
        assert self.validator._check_balanced_structure(unbalanced_text) is False
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty document
        result = self.validator.validate_document_rules("", {})
        assert result['overall_score'] == 0.0
        
        # Very short document
        short_text = "Short"
        result = self.validator.validate_document_rules(short_text, {})
        assert result['passes_validation'] is False

if __name__ == "__main__":
    pytest.main([__file__])
