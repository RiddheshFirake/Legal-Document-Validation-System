import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle

class DocumentInference:
    def __init__(self, model_path: str, vectorizer_path: str):
        self.classifier = None
        self.vectorizer = None
        self.load_models(model_path, vectorizer_path)
    
    def load_models(self, model_path: str, vectorizer_path: str):
        """Load trained models and vectorizers"""
        # Load classifier
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data
        
        # Load vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
    
    def predict_single(self, text: str, processed_data: Dict) -> Dict:
        """Predict legality for a single document"""
        """Predict legality for a single document"""
        if not self.vectorizer or not self.classifier or 'processed_text' not in processed_data:
            # Return a neutral, low-confidence result if components are missing or data is malformed
            return {
                'prediction': 0, 
                'confidence': 0.0,
                'is_legal': False,
                'probability': [0.5, 0.5],
                'error': 'NLP component not ready or data is missing'
            }
        # Vectorize text
        features = self.vectorizer.transform([text], [processed_data])
        
        # Prepare features
        X = self._prepare_features(features)
        
        # Get best model
        best_model_name = self.classifier['best_model']
        model = self.classifier['models'][best_model_name]
        
        # Apply scaling if needed
        if best_model_name in ['logistic_regression', 'svm']:
            scaler = self.classifier['scalers']['main']
            X = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        # Get ensemble prediction if available
        ensemble_pred = None
        ensemble_proba = None
        
        if 'ensemble' in self.classifier['models']:
            # Get predictions from all base models
            base_predictions = []
            for model_name, base_model in self.classifier['models'].items():
                if model_name != 'ensemble':
                    if model_name in ['logistic_regression', 'svm']:
                        X_model = self.classifier['scalers']['main'].transform(X)
                    else:
                        X_model = X
                    
                    base_pred = base_model.predict_proba(X_model)[0, 1]
                    base_predictions.append(base_pred)
            
            # Get ensemble prediction
            ensemble_features = np.array(base_predictions).reshape(1, -1)
            ensemble_model = self.classifier['models']['ensemble']
            ensemble_pred = ensemble_model.predict(ensemble_features)[0]
            ensemble_proba = ensemble_model.predict_proba(ensemble_features)[0]
        
        return {
            'prediction': int(prediction),
            'probability': probability.tolist(),
            'confidence': float(max(probability)),
            'is_legal': bool(prediction),
            'ensemble_prediction': int(ensemble_pred) if ensemble_pred is not None else None,
            'ensemble_probability': ensemble_proba.tolist() if ensemble_proba is not None else None,
            'model_used': best_model_name
        }
    
    def predict_batch(self, texts: List[str], processed_data: List[Dict]) -> List[Dict]:
        """Predict legality for multiple documents"""
        results = []
        
        for text, data in zip(texts, processed_data):
            result = self.predict_single(text, data)
            results.append(result)
        
        return results
    
    def _prepare_features(self, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine all features into a single matrix"""
        feature_matrices = []
        
        for feature_name, features in feature_dict.items():
            if features is not None and len(features) > 0:
                feature_matrices.append(features)
        
        if feature_matrices:
            combined_features = np.hstack(feature_matrices)
        else:
            raise ValueError("No features provided")
        
        return combined_features
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> Dict:
        """Get feature importance for specified model"""
        if model_name is None:
            model_name = self.classifier['best_model']
        
        if model_name in self.classifier['feature_importance']:
            return {
                'model': model_name,
                'importance': self.classifier['feature_importance'][model_name].tolist()
            }
        else:
            return {'error': f'Feature importance not available for {model_name}'}
    
    def explain_prediction(self, text: str, processed_data: Dict) -> Dict:
        """Provide explanation for prediction"""
        prediction_result = self.predict_single(text, processed_data)
        
        # Get feature importance
        importance = self.get_feature_importance()
        
        # Analyze text characteristics
        features = processed_data.get('features', {})
        entities = processed_data.get('entities', {})
        
        explanation = {
            'prediction_summary': prediction_result,
            'text_analysis': {
                'word_count': features.get('word_count', 0),
                'legal_density': features.get('legal_density', 0),
                'has_signatures': len(entities.get('signatures', [])) > 0,
                'has_dates': len(entities.get('dates', [])) > 0,
                'has_parties': len(entities.get('parties', [])) > 0,
                'clause_count': len(entities.get('clauses', []))
            },
            'confidence_factors': self._analyze_confidence_factors(prediction_result, features, entities)
        }
        
        return explanation
    
    def _analyze_confidence_factors(self, prediction: Dict, features: Dict, entities: Dict) -> List[str]:
        """Analyze factors contributing to prediction confidence"""
        factors = []
        confidence = prediction['confidence']
        is_legal = prediction['is_legal']
        
        # High confidence factors
        if confidence > 0.9:
            if is_legal:
                factors.append("Strong legal language patterns detected")
                if features.get('legal_density', 0) > 0.1:
                    factors.append("High density of legal terminology")
                if len(entities.get('clauses', [])) > 5:
                    factors.append("Multiple legal clauses identified")
            else:
                factors.append("Document lacks legal document characteristics")
                if features.get('legal_density', 0) < 0.05:
                    factors.append("Low legal terminology density")
        
        # Medium confidence factors
        elif confidence > 0.7:
            factors.append("Moderate confidence based on document structure")
            if len(entities.get('signatures', [])) > 0:
                factors.append("Signature patterns detected")
            if len(entities.get('dates', [])) > 2:
                factors.append("Multiple date references found")
        
        # Low confidence factors
        else:
            factors.append("Low confidence - document has mixed characteristics")
            factors.append("Manual review recommended")
        
        return factors
