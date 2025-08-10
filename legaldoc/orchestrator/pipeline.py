import asyncio
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
import time
import numpy as np

class DocumentValidationPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.file_detector = None
        self.ocr_processor = None
        self.pdf_extractor = None
        self.text_preprocessor = None
        self.vectorizer = None
        self.nlp_classifier = None
        self.vision_model = None
        self.mrz_validator = None
        self.clause_validator = None
        self.rules_validator = None
        self.signature_validator = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Import and initialize components
            from ingest.file_detector import FileDetector
            from ingest.ocr import OCRProcessor
            from ingest.pdf_meta import PDFMetadataExtractor
            from nlp.preprocess import TextPreprocessor
            from nlp.vectorize import TextVectorizer
            from nlp.clf_infer import DocumentInference
            from vision.infer import DocumentVisionInference
            from validators.mrz import MRZValidator
            from validators.clauses import LegalClauseValidator
            from validators.rules import RulesValidator  # Corrected to import RulesValidator
            from validators.pdf_sign import PDFSignatureValidator

            self.file_detector = FileDetector()
            self.ocr_processor = OCRProcessor(self.config.get('processing', {}).get('ocr', {}))
            self.pdf_extractor = PDFMetadataExtractor()
            self.text_preprocessor = TextPreprocessor()
            self.vectorizer = TextVectorizer(self.config.get('model', {}).get('nlp', {}))

            # Load trained models if available
            model_paths = self.config.get('paths', {})
            models_dir = model_paths.get('models_dir', 'models/')

            try:
                self.nlp_classifier = DocumentInference(
                    f"{models_dir}/nlp_model.pkl",
                    f"{models_dir}/vectorizer.pkl"
                )
            except Exception as e:
                self.logger.warning(f"NLP classifier not loaded - inference will be limited. Error: {e}")
                self.nlp_classifier = None

            try:
                self.vision_model = DocumentVisionInference(
                    f"{models_dir}/vision_model.pth",
                    self.config.get('model', {}).get('vision', {})
                )
            except Exception as e:
                self.logger.warning(f"Vision model not loaded - visual analysis will be limited. Error: {e}")
                self.vision_model = None

            # Initialize validators
            self.mrz_validator = MRZValidator()
            self.clause_validator = LegalClauseValidator()
            self.rules_validator = RulesValidator(self.config.get('rules', {})) # Corrected to use RulesValidator
            self.signature_validator = PDFSignatureValidator()

            self.logger.info("Pipeline components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing pipeline components: {e}")
            raise

    def process_document(self, file_path: str, options: Dict = None) -> Dict:
        """Process a single document through the complete pipeline."""
        options = options or {}
        start_time = time.time()

        try:
            # Stage 1: File Detection and Validation
            self.logger.info(f"Processing document: {file_path}")
            
            # --- CORRECTED CODE: Unpack the tuple return from validate_file
            is_valid, error_message = self.file_detector.validate_file(file_path)

            if not is_valid:
                return {
                    'success': False,
                    'error': error_message,
                    'stage': 'file_detection'
                }
            # --- END OF CORRECTED CODE ---

            # Stage 2: Content Extraction
            extraction_result = self._extract_content(file_path, is_valid)
            if not extraction_result['success']:
                return {
                    'success': False,
                    'error': extraction_result['error'],
                    'stage': 'content_extraction'
                }
            
            # Stage 3: Text Processing
            processing_result = self._process_text(extraction_result['text'])

            # Stage 4: Multi-modal Analysis
            analysis_result = self._analyze_document(
                file_path=file_path,
                text=extraction_result['text'],
                processed_data=processing_result,
                metadata=extraction_result['metadata'],
                images=extraction_result.get('images', [])
            )

            # Stage 5: Rule-based Validation
            validation_result = self._validate_document(
                text=extraction_result['text'],
                metadata=extraction_result['metadata'],
                file_path=file_path
            )

            # Stage 6: Decision Fusion
            final_decision = self._make_final_decision(
                analysis_result, validation_result, options
            )

            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'file_info': {'is_valid': is_valid},
                'extraction': extraction_result,
                'processing': processing_result,
                'analysis': analysis_result,
                'validation': validation_result,
                'decision': final_decision,
                'processing_time': processing_time,
                'timestamp': time.time()
            }

        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'stage': 'unknown',
                'processing_time': time.time() - start_time
            }

    def _extract_content(self, file_path: str, file_info: Dict) -> Dict:
        """Extract content from the document."""
        try:
            # Get the MIME type, which is no longer in file_info, so we call it directly
            from ingest.file_detector import detect_file_type
            mime_type = detect_file_type(file_path)

            if mime_type == 'pdf':
                return self._extract_pdf_content(file_path)
            elif mime_type.startswith('image'):
                return self._extract_image_content(file_path)
            elif mime_type == 'text':
                from ingest.text_extractor import extract_text
                text_content = extract_text(file_path)
                return {
                    'success': True,
                    'text': text_content,
                    'pages_text': [text_content],
                    'metadata': {'file_type': 'text'}
                }
            elif mime_type == 'csv':
                from ingest.text_extractor import extract_text
                text_content = extract_text(file_path)
                return {
                    'success': True,
                    'text': text_content,
                    'pages_text': [text_content],
                    'metadata': {'file_type': 'csv'}
                }
            else:
                return {
                    'success': False,
                    'error': f'Unsupported file type: {mime_type}'
                }
        except Exception as e:
            return {
                'success': False,
                'error': f'Content extraction failed: {str(e)}'
            }
            
    def _extract_pdf_content(self, file_path: str) -> Dict:
        """Extract content from PDF file."""
        try:
            pages_text = self.pdf_extractor.extract_text_by_page(file_path)
            full_text = '\n'.join(pages_text)
            basic_metadata = self.pdf_extractor.extract_basic_metadata(file_path)
            advanced_metadata = self.pdf_extractor.extract_advanced_metadata(file_path)
            images = self.pdf_extractor.convert_to_images(file_path, dpi=300)
            
            if len(full_text.strip()) < 100 and images:
                ocr_text = ""
                for image in images:
                    ocr_result = self.ocr_processor.extract_text(image)
                    if ocr_result['success']:
                        ocr_text += ocr_result['full_text'] + "\n"
                if ocr_text.strip():
                    full_text = ocr_text
            
            return {
                'success': True,
                'text': full_text,
                'pages_text': pages_text,
                'images': images,
                'metadata': {
                    **basic_metadata,
                    **advanced_metadata
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'PDF extraction failed: {str(e)}'
            }

    def _extract_image_content(self, file_path: str) -> Dict:
        """Extract content from image file."""
        try:
            import cv2
            image = cv2.imread(file_path)
            ocr_result = self.ocr_processor.extract_text(image)
            if not ocr_result['success']:
                return {
                    'success': False,
                    'error': f"OCR failed: {ocr_result.get('error', 'Unknown error')}"
                }
            return {
                'success': True,
                'text': ocr_result['full_text'],
                'images': [image],
                'metadata': {
                    'ocr_confidence': np.mean([block['confidence'] for block in ocr_result['text_blocks']]) if ocr_result['text_blocks'] else 0.0,
                    'text_blocks_count': len(ocr_result['text_blocks'])
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Image extraction failed: {str(e)}'
            }

    def _process_text(self, text: str) -> Dict:
        """Process extracted text."""
        try:
            processed_data = self.text_preprocessor.preprocess_for_classification(text)
            return {
                'success': True,
                'processed_data': processed_data
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Text processing failed: {str(e)}'
            }

    def _analyze_document(self, file_path: str, text: str, processed_data: Dict, metadata: Dict, images: List) -> Dict:
        """Perform multi-modal document analysis."""
        analysis_results = {}
        # NLP Analysis
        if self.nlp_classifier:
            try:
                nlp_result = self.nlp_classifier.predict_single(text, processed_data['processed_data'])
                analysis_results['nlp'] = nlp_result
            except Exception as e:
                analysis_results['nlp'] = {'error': str(e), 'available': False}

        # Vision Analysis
        if self.vision_model and images:
            try:
                vision_results = []
                for image in images[:3]:  # Limit to first 3 images for performance
                    vision_result = self.vision_model.comprehensive_analysis(image)
                    vision_results.append(vision_result)
                avg_confidence = np.mean([r['prediction']['confidence'] for r in vision_results]) if vision_results else 0.0
                is_legal_votes = sum(1 for r in vision_results if r['prediction']['is_legal'])
                analysis_results['vision'] = {
                    'individual_results': vision_results,
                    'aggregate_confidence': avg_confidence,
                    'is_legal_consensus': is_legal_votes > len(vision_results) / 2,
                    'consensus_strength': is_legal_votes / len(vision_results) if vision_results else 0.0,
                    'available': True
                }
            except Exception as e:
                analysis_results['vision'] = {'error': str(e), 'available': False}
        
        # Mark as unavailable if models don't exist
        if not self.nlp_classifier:
            analysis_results['nlp'] = {'score': 0.5, 'confidence': 0.0, 'available': False}
        if not self.vision_model or not images:
            analysis_results['vision'] = {'score': 0.5, 'confidence': 0.0, 'available': False}
            
        return analysis_results

    def _validate_document(self, text: str, metadata: Dict, file_path: str) -> Dict:
        """Perform rule-based document validation."""
        validation_results = {}
        
        # Rules-based Validation (main rules)
        try:
            rules_result = self.rules_validator.validate(text, metadata)
            validation_results['rules'] = rules_result
        except Exception as e:
            validation_results['rules'] = {'error': str(e), 'available': False}
            
        # MRZ Validation
        try:
            mrz_result = self.mrz_validator.comprehensive_mrz_validation(text)
            validation_results['mrz'] = mrz_result
        except Exception as e:
            validation_results['mrz'] = {'error': str(e), 'available': False}

        # Legal Clause Validation
        try:
            clause_result = self.clause_validator.comprehensive_clause_validation(text)
            validation_results['clauses'] = clause_result
        except Exception as e:
            validation_results['clauses'] = {'error': str(e), 'available': False}

        # Signature Validation (for PDFs)
        if file_path.lower().endswith('.pdf'):
            try:
                signature_result = self.signature_validator.comprehensive_signature_validation(file_path, text)
                validation_results['signatures'] = signature_result
            except Exception as e:
                validation_results['signatures'] = {'error': str(e), 'available': False}
        else:
             validation_results['signatures'] = {'score': 0.5, 'confidence': 0.0, 'available': False, 'details': 'Signature validation skipped for non-PDF file'}

        return validation_results

    def _make_final_decision(self, analysis_result: Dict, validation_result: Dict, options: Dict) -> Dict:
        """Make final legality decision using ensemble approach."""
        from orchestrator.decision import DecisionFusion
        decision_maker = DecisionFusion(self.config.get('model', {}).get('ensemble', {}))
        
        # Combine all results into a single dictionary for the decision maker
        combined_results = {
            'analysis': analysis_result,
            'validation': validation_result
        }
        
        decision = decision_maker.make_decision(combined_results, options)
        return decision

    async def process_batch(self, file_paths: List[str], options: Dict = None) -> List[Dict]:
        """Process multiple documents asynchronously."""
        options = options or {}
        max_concurrent = options.get('max_concurrent', 3)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single(file_path: str) -> Dict:
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self.process_document, file_path, options
                )

        tasks = [process_single(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'file_path': file_paths[i],
                    'error': str(result),
                    'stage': 'batch_processing'
                })
            else:
                result['file_path'] = file_paths[i]
                processed_results.append(result)

        return processed_results

    def get_pipeline_status(self) -> Dict:
        """Get status of pipeline components."""
        return {
            'file_detector': self.file_detector is not None,
            'ocr_processor': self.ocr_processor is not None,
            'pdf_extractor': self.pdf_extractor is not None,
            'text_preprocessor': self.text_preprocessor is not None,
            'vectorizer': self.vectorizer is not None,
            'nlp_classifier': self.nlp_classifier is not None,
            'vision_model': self.vision_model is not None,
            'mrz_validator': self.mrz_validator is not None,
            'clause_validator': self.clause_validator is not None,
            'rules_validator': self.rules_validator is not None,
            'signature_validator': self.signature_validator is not None,
            'timestamp': time.time()
        }

