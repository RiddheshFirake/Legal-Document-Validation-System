import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from typing import Dict, List, Optional, Union
import os

class DocumentVisionInference:
    def __init__(self, model_path: str, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.get('image_size', 224), config.get('image_size', 224))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path: str) -> nn.Module:
        """Load trained model"""
        from .model import create_model
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model with same configuration
        model_config = checkpoint.get('config', {})
        model = create_model(
            model_type=model_config.get('model_type', 'cnn'),
            num_classes=model_config.get('num_classes', 2),
            backbone=model_config.get('backbone', 'resnet50'),
            pretrained=False  # Don't need pretrained weights when loading trained model
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """Preprocess image for inference"""
        # Convert to PIL Image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError("Unsupported image type")
        
        # Apply transforms
        tensor_image = self.transform(pil_image)
        
        # Add batch dimension
        tensor_image = tensor_image.unsqueeze(0)
        
        return tensor_image.to(self.device)
    
    def predict_single(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        """Predict document legality for a single image"""
        with torch.no_grad():
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Forward pass
            if hasattr(self.model, 'forward') and len(self.model.forward.__code__.co_varnames) > 2:
                outputs, attention_weights = self.model(input_tensor)
                attention_weights = attention_weights.cpu().numpy()
            else:
                outputs = self.model(input_tensor)
                attention_weights = None
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Convert to numpy
            probabilities = probabilities.cpu().numpy()[0]
            predicted = predicted.cpu().item()
            confidence = confidence.cpu().item()
            
            return {
                'prediction': predicted,
                'probabilities': probabilities.tolist(),
                'confidence': confidence,
                'is_legal': bool(predicted),
                'attention_weights': attention_weights.tolist() if attention_weights is not None else None
            }
    
    def predict_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> List[Dict]:
        """Predict document legality for multiple images"""
        results = []
        
        for image in images:
            result = self.predict_single(image)
            results.append(result)
        
        return results
    
    def extract_visual_features(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        """Extract visual features from document image"""
        # Convert to numpy array if needed
        if isinstance(image, str):
            img_array = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_array = image
        
        # Extract features using computer vision techniques
        from .dataset import extract_document_features
        features = extract_document_features(img_array)
        
        return features
    
    def analyze_document_layout(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        """Analyze document layout and structure"""
        # Convert to numpy array
        if isinstance(image, str):
            img_array = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_array = image
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Detect text regions
        # Use EAST text detector or similar
        height, width = gray.shape
        
        # Simple text detection using contours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        min_area = (width * height) * 0.001
        text_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Analyze layout
        layout_analysis = {
            'total_contours': len(contours),
            'text_regions': len(text_contours),
            'image_dimensions': {'width': width, 'height': height},
            'text_coverage': sum(cv2.contourArea(c) for c in text_contours) / (width * height),
        }
        
        # Detect potential signature regions (bottom 20% of document)
        bottom_y = int(height * 0.8)
        signature_contours = [c for c in text_contours 
                            if cv2.boundingRect(c)[1] > bottom_y]
        
        layout_analysis['potential_signatures'] = len(signature_contours)
        
        # Detect header/footer regions
        header_y = int(height * 0.1)
        footer_y = int(height * 0.9)
        
        header_contours = [c for c in text_contours 
                          if cv2.boundingRect(c)[1] < header_y]
        footer_contours = [c for c in text_contours 
                          if cv2.boundingRect(c)[1] > footer_y]
        
        layout_analysis['header_elements'] = len(header_contours)
        layout_analysis['footer_elements'] = len(footer_contours)
        
        return layout_analysis
    
    def get_prediction_confidence_factors(self, prediction_result: Dict, 
                                        visual_features: Dict, 
                                        layout_analysis: Dict) -> List[str]:
        """Analyze factors contributing to prediction confidence"""
        factors = []
        confidence = prediction_result['confidence']
        is_legal = prediction_result['is_legal']
        
        # High confidence factors
        if confidence > 0.9:
            if is_legal:
                factors.append("Strong visual indicators of legal document")
                if layout_analysis.get('potential_signatures', 0) > 0:
                    factors.append("Signature regions detected")
                if visual_features.get('text_density', 0) > 0.3:
                    factors.append("Appropriate text density for legal document")
            else:
                factors.append("Visual characteristics inconsistent with legal documents")
                if visual_features.get('text_density', 0) < 0.1:
                    factors.append("Insufficient text content")
        
        # Medium confidence factors
        elif confidence > 0.7:
            factors.append("Moderate confidence based on visual analysis")
            if layout_analysis.get('text_regions', 0) > 10:
                factors.append("Multiple text regions detected")
        
        # Low confidence factors
        else:
            factors.append("Low confidence - visual analysis inconclusive")
            factors.append("Manual review recommended")
        
        return factors
    
    def comprehensive_analysis(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        """Perform comprehensive document analysis"""
        # Get prediction
        prediction = self.predict_single(image)
        
        # Extract visual features
        visual_features = self.extract_visual_features(image)
        
        # Analyze layout
        layout_analysis = self.analyze_document_layout(image)
        
        # Get confidence factors
        confidence_factors = self.get_prediction_confidence_factors(
            prediction, visual_features, layout_analysis
        )
        
        return {
            'prediction': prediction,
            'visual_features': visual_features,
            'layout_analysis': layout_analysis,
            'confidence_factors': confidence_factors,
            'analysis_summary': {
                'is_legal': prediction['is_legal'],
                'confidence': prediction['confidence'],
                'has_signatures': layout_analysis.get('potential_signatures', 0) > 0,
                'text_coverage': visual_features.get('text_density', 0),
                'document_structure_score': min(1.0, layout_analysis.get('text_regions', 0) / 20)
            }
        }
