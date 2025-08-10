import pytest
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validators.mrz import MRZValidator

class TestMRZValidator:
    def setup_method(self):
        """Setup test fixtures"""
        self.validator = MRZValidator()
        
        # Sample MRZ data for testing
        self.sample_passport_mrz = [
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
            "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
        ]
        
        self.sample_invalid_mrz = [
            "INVALID_LINE_1",
            "INVALID_LINE_2"
        ]
    
    def test_check_digit_calculation(self):
        """Test check digit calculation"""
        # Test cases from official MRZ specification
        test_cases = [
            ("AB2134", 5),
            ("D23145890", 1),
            ("123456789", 4)
        ]
        
        for data, expected in test_cases:
            calculated = self.validator.calculate_check_digit(data)
            assert calculated == expected, f"Check digit for {data} should be {expected}, got {calculated}"
    
    def test_validate_passport_check_digits(self):
        """Test passport MRZ check digit validation"""
        result = self.validator.validate_check_digits(self.sample_passport_mrz, 'passport')
        
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'check_results' in result
        assert 'document_type' in result
        assert result['document_type'] == 'passport'
    
    def test_extract_passport_data(self):
        """Test passport data extraction"""
        data = self.validator.extract_mrz_data(self.sample_passport_mrz, 'passport')
        
        assert isinstance(data, dict)
        assert 'document_type' in data
        assert 'issuing_country' in data
        assert 'surname' in data
        assert 'given_names' in data
        assert 'document_number' in data
        assert 'nationality' in data
        assert 'birth_date' in data
        assert 'sex' in data
        assert 'expiry_date' in data
        
        # Check specific values
        assert data['document_type'] == 'P'
        assert data['issuing_country'] == 'UTO'
        assert data['surname'] == 'ERIKSSON'
        assert data['given_names'] == 'ANNA MARIA'
    
    def test_date_formatting(self):
        """Test date formatting functionality"""
        test_cases = [
            ("741212", "1974-12-12"),  # Old date
            ("201231", "2020-12-31"),  # Recent date
            ("invalid", "invalid")     # Invalid date
        ]
        
        for input_date, expected in test_cases:
            formatted = self.validator.format_date(input_date)
            assert formatted == expected, f"Date {input_date} should format to {expected}, got {formatted}"
    
    def test_detect_mrz_lines(self):
        """Test MRZ line detection in text"""
        # Text containing MRZ
        text_with_mrz = """
        Some document content
        P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<
        L898902C36UTO7408122F1204159ZE184226B<<<<<10
        More document content
        """
        
        lines = self.validator.detect_mrz_lines(text_with_mrz)
        assert len(lines) == 2
        assert lines[0] == "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<"
        assert lines[1] == "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
        
        # Text without MRZ
        text_without_mrz = "This is regular document text without MRZ codes."
        lines = self.validator.detect_mrz_lines(text_without_mrz)
        assert len(lines) == 0
    
    def test_validate_mrz_structure(self):
        """Test MRZ structure validation"""
        # Valid passport MRZ
        result = self.validator.validate_mrz_structure(self.sample_passport_mrz)
        assert isinstance(result, dict)
        assert 'valid' in result
        
        # Invalid MRZ
        result = self.validator.validate_mrz_structure(self.sample_invalid_mrz)
        assert result['valid'] is False
        assert 'error' in result
    
    def test_comprehensive_mrz_validation(self):
        """Test comprehensive MRZ validation"""
        # Text with valid MRZ
        text_with_mrz = "\n".join(self.sample_passport_mrz)
        result = self.validator.comprehensive_mrz_validation(text_with_mrz)
        
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'has_mrz' in result
        assert result['has_mrz'] is True
        
        # Text without MRZ
        text_without_mrz = "Regular document text"
        result = self.validator.comprehensive_mrz_validation(text_without_mrz)
        assert result['has_mrz'] is False
        assert result['valid'] is False
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty input
        result = self.validator.comprehensive_mrz_validation("")
        assert result['valid'] is False
        assert result['has_mrz'] is False
        
        # Invalid document type
        result = self.validator.validate_check_digits(self.sample_passport_mrz, 'unknown_type')
        assert result['valid'] is False
        assert 'error' in result
        
        # Malformed MRZ lines
        malformed_mrz = ["TOO_SHORT", "ALSO_TOO_SHORT"]
        result = self.validator.validate_mrz_structure(malformed_mrz)
        assert result['valid'] is False

if __name__ == "__main__":
    pytest.main([__file__])

