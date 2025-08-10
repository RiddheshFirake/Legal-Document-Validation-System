import pytest
import os
import sys
import tempfile
import yaml
from unittest.mock import Mock, patch
import asyncio

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.pipeline import DocumentValidationPipeline

class TestDocumentValidationPipeline:
    def setup_method(self):
        """Setup test fixtures"""
        # Mock configuration
        self.config = {
            'model': {
                'nlp': {'model_name': 'test_model'},
                'vision': {'model_name': 'test_vision'},
                'ensemble': {'weights': {'nlp': 0.4, 'vision': 0.3, 'rules': 0.3}}
            },
            'paths': {
                'models_dir': 'test_models/',
                'data_dir': 'test_data/'
            },
            'processing': {
                'ocr': {'languages': ['en']}
            }
        }
        
        # Create sample test files
        self.temp_dir = tempfile.mkdtemp()
        self.sample_pdf_path = os.path.join(self.temp_dir, 'test.pdf')
        self.sample_image_path = os.path.join(self.temp_dir, 'test.jpg')
        
        # Create minimal test files (these won't be real PDFs/images)
        with open(self.sample_pdf_path, 'w') as f:
            f.write('Sample PDF content')
        
        with open(self.sample_image_path, 'w') as f:
            f.write('Sample image content')
    
    def teardown_method(self):
        """Cleanup after tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """Test pipeline component initialization"""
        with patch('orchestrator.pipeline.FileDetector'), \
             patch('orchestrator.pipeline.OCRProcessor'), \
             patch('orchestrator.pipeline.PDFMetadataExtractor'), \
             patch('orchestrator.pipeline.TextPreprocessor'), \
             patch('orchestrator.pipeline.TextVectorizer'), \
             patch('orchestrator.pipeline.DocumentInference'), \
             patch('orchestrator.pipeline.DocumentVisionInference'), \
             patch('orchestrator.pipeline.MRZValidator'), \
             patch('orchestrator.pipeline.LegalClauseValidator'), \
             patch('orchestrator.pipeline.DocumentRulesValidator'), \
             patch('orchestrator.pipeline.PDFSignatureValidator'):
            
            pipeline = DocumentValidationPipeline(self.config)
            
            assert pipeline.config == self.config
            assert pipeline.file_detector is not None
            assert pipeline.ocr_processor is not None
            assert pipeline.pdf_extractor is not None
    
    @patch('orchestrator.pipeline.DocumentValidationPipeline._initialize_components')
    def test_get_pipeline_status(self, mock_init):
        """Test pipeline status retrieval"""
        pipeline = DocumentValidationPipeline(self.config)
        
        # Mock some components
        pipeline.file_detector = Mock()
        pipeline.ocr_processor = Mock()
        pipeline.nlp_classifier = None  # Simulate missing component
        
        status = pipeline.get_pipeline_status()
        
        assert isinstance(status, dict)
        assert 'file_detector' in status
        assert 'ocr_processor' in status
        assert 'nlp_classifier' in status
        assert 'timestamp' in status
        
        assert status['file_detector'] is True
        assert status['ocr_processor'] is True
        assert status['nlp_classifier'] is False
    
    @patch('orchestrator.pipeline.DocumentValidationPipeline._initialize_components')
    def test_extract_pdf_content_mock(self, mock_init):
        """Test PDF content extraction with mocked components"""
        pipeline = DocumentValidationPipeline(self.config)
        
        # Mock PDF extractor
        mock_pdf_extractor = Mock()
        mock_pdf_extractor.extract_text_by_page.return_value = ['Page 1 text', 'Page 2 text']
        mock_pdf_extractor.extract_basic_metadata.return_value = {'pages': 2}
        mock_pdf_extractor.extract_advanced_metadata.return_value = {'fonts': ['Arial']}
        mock_pdf_extractor.convert_to_images.return_value = [Mock(), Mock()]
        
        pipeline.pdf_extractor = mock_pdf_extractor
        
        # Mock file info
        file_info = {'mime_type': 'application/pdf'}
        
        result = pipeline._extract_pdf_content(self.sample_pdf_path)
        
        assert result['success'] is True
        assert 'text' in result
        assert 'pages_text' in result
        assert 'images' in result
        assert 'metadata' in result
    
    @patch('orchestrator.pipeline.DocumentValidationPipeline._initialize_components')
    def test_process_text_mock(self, mock_init):
        """Test text processing with mocked components"""
        pipeline = DocumentValidationPipeline(self.config)
        
        # Mock text preprocessor
        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_for_classification.return_value = {
            'processed_data': {
                'cleaned_text': 'Sample text',
                'features': {'word_count': 10}
            }
        }
        
        pipeline.text_preprocessor = mock_preprocessor
        
        result = pipeline._process_text('Sample input text')
        
        assert result['success'] is True
        assert 'processed_data' in result
    
    @patch('orchestrator.pipeline.DocumentValidationPipeline._initialize_components')
    def test_analyze_document_mock(self, mock_init):
        """Test document analysis with mocked components"""
        pipeline = DocumentValidationPipeline(self.config)
        
        # Mock NLP classifier
        mock_nlp_classifier = Mock()
        mock_nlp_classifier.predict_single.return_value = {
            'is_legal': True,
            'confidence': 0.85
        }
        
        # Mock vision model
        mock_vision_model = Mock()
        mock_vision_model.comprehensive_analysis.return_value = {
            'prediction': {'is_legal': True, 'confidence': 0.80}
        }
        
        pipeline.nlp_classifier = mock_nlp_classifier
        pipeline.vision_model = mock_vision_model
        
        result = pipeline._analyze_document(
            file_path='test.pdf',
            text='Sample text',
            processed_data={'processed_data': {}},
            metadata={},
            images=[Mock()]
        )
        
        assert 'nlp' in result
        assert 'vision' in result
        assert result['nlp']['is_legal'] is True
    
    @patch('orchestrator.pipeline.DocumentValidationPipeline._initialize_components')
    def test_validate_document_mock(self, mock_init):
        """Test document validation with mocked validators"""
        pipeline = DocumentValidationPipeline(self.config)
        
        # Mock validators
        mock_mrz_validator = Mock()
        mock_mrz_validator.comprehensive_mrz_validation.return_value = {
            'valid': False,
            'has_mrz': False
        }
        
        mock_clause_validator = Mock()
        mock_clause_validator.comprehensive_clause_validation.return_value = {
            'is_valid_legal_document': True,
            'overall_score': 0.75
        }
        
        mock_rules_validator = Mock()
        mock_rules_validator.validate_document_rules.return_value = {
            'passes_validation': True,
            'overall_score': 0.80
        }
        
        mock_signature_validator = Mock()
        mock_signature_validator.comprehensive_signature_validation.return_value = {
            'signatures_valid': True,
            'confidence': 0.85
        }
        
        pipeline.mrz_validator = mock_mrz_validator
        pipeline.clause_validator = mock_clause_validator
        pipeline.rules_validator = mock_rules_validator
        pipeline.signature_validator = mock_signature_validator
        
        result = pipeline._validate_document('Sample text', {}, 'test.pdf')
        
        assert 'mrz' in result
        assert 'clauses' in result
        assert 'rules' in result
        assert 'signatures' in result
    
    @patch('orchestrator.pipeline.DocumentValidationPipeline._initialize_components')
    def test_error_handling(self, mock_init):
        """Test error handling in pipeline"""
        pipeline = DocumentValidationPipeline(self.config)
        
        # Mock file detector to return invalid file
        mock_file_detector = Mock()
        mock_file_detector.validate_file.return_value = {
            'valid': False,
            'error': 'Invalid file type'
        }
        
        pipeline.file_detector = mock_file_detector
        
        result = pipeline.process_document('invalid_file.txt')
        
        assert result['success'] is False
        assert 'error' in result
        assert result['stage'] == 'file_detection'
    
    @patch('orchestrator.pipeline.DocumentValidationPipeline._initialize_components')
    def test_batch_processing_mock(self, mock_init):
        """Test batch processing functionality"""
        pipeline = DocumentValidationPipeline(self.config)
        
        # Mock process_document to return success
        def mock_process_document(file_path, options=None):
            return {
                'success': True,
                'decision': {'is_legal': True, 'confidence': 0.8},
                'processing_time': 1.0
            }
        
        pipeline.process_document = mock_process_document
        
        # Test batch processing
        file_paths = [self.sample_pdf_path, self.sample_image_path]
        
        async def run_batch_test():
            results = await pipeline.process_batch(file_paths)
            return results
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(run_batch_test())
            
            assert len(results) == 2
            assert all(result['success'] for result in results)
        finally:
            loop.close()

if __name__ == "__main__":
    pytest.main([__file__])
