import numpy as np
from typing import Dict, List, Optional, Any
import logging

class DecisionFusion:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Default weights for different components
        self.default_weights = {
            'nlp': 0.4,
            'vision': 0.3,
            'rules': 0.3
        }
        
        # Get weights from config or use defaults
        self.weights = config.get('weights', self.default_weights)
        
        # Confidence thresholds
        self.thresholds = {
            'high_confidence': 0.9,
            'medium_confidence': 0.7,
            'low_confidence': 0.5
        }
    
    def make_decision(self, combined_results: Dict, options: Dict = None) -> Dict:
        """Make final legality decision based on all component results"""
        options = options or {}
        
        try:
            # Extract individual component scores
            component_scores = self._extract_component_scores(combined_results)
            
            # Calculate weighted ensemble score
            ensemble_score = self._calculate_ensemble_score(component_scores)
            
            # Make binary decision - ENSURE variables are properly defined
            is_legal, confidence = self._make_binary_decision(ensemble_score, component_scores)
            
            # Generate explanation
            explanation = self._generate_explanation(
                component_scores, ensemble_score, is_legal, confidence
            )
            
            # Calculate risk assessment
            risk_assessment = self._assess_risk(component_scores, confidence)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                component_scores, is_legal, confidence, risk_assessment
            )
            
            # Convert numpy types to Python types before returning
            def convert_numpy_types(obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            result = {
                'is_legal': bool(is_legal),  # Now is_legal is properly defined
                'confidence': float(confidence),  # Now confidence is properly defined
                'ensemble_score': float(ensemble_score),  # Now ensemble_score is properly defined
                'component_scores': convert_numpy_types(component_scores),
                'explanation': explanation,
                'risk_assessment': convert_numpy_types(risk_assessment),
                'recommendations': recommendations,
                'decision_metadata': {
                    'weights_used': self.weights,
                    'thresholds_used': self.thresholds,
                    'components_available': list(component_scores.keys())
                }
            }
            
            return convert_numpy_types(result)
            
        except Exception as e:
            # Fallback decision in case of errors
            self.logger.error(f"Error in decision making: {e}")
            return {
                'is_legal': False,
                'confidence': 0.0,
                'ensemble_score': 0.0,
                'component_scores': {},
                'explanation': {'error': str(e), 'decision': 'NOT LEGAL', 'confidence_level': 'LOW'},
                'risk_assessment': {'risk_level': 'HIGH', 'requires_human_review': True},
                'recommendations': ['Manual review required due to processing error'],
                'decision_metadata': {'error': str(e)}
            }

        """Make final legality decision based on all component results"""
        # ... existing code ...
        
        # Convert numpy types to Python types before returning
        def convert_numpy_types(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        result = {
            'is_legal': bool(is_legal),  # Ensure it's Python bool
            'confidence': float(confidence),  # Ensure it's Python float
            'ensemble_score': float(ensemble_score),
            'component_scores': convert_numpy_types(component_scores),
            'explanation': explanation,
            'risk_assessment': convert_numpy_types(risk_assessment),
            'recommendations': recommendations,
            'decision_metadata': {
                'weights_used': self.weights,
                'thresholds_used': self.thresholds,
                'components_available': list(component_scores.keys())
            }
        }
        
        return convert_numpy_types(result)

    
    def _extract_component_scores(self, combined_results: Dict) -> Dict:
        """Extract scores from all components"""
        component_scores = {}
        
        # NLP Analysis
        analysis = combined_results.get('analysis', {})
        nlp_result = analysis.get('nlp', {})
        
        if 'error' not in nlp_result and nlp_result:
            component_scores['nlp'] = {
                'score': 1.0 if nlp_result.get('is_legal', False) else 0.0,
                'confidence': nlp_result.get('confidence', 0.0),
                'available': True
            }
        else:
            component_scores['nlp'] = {'score': 0.5, 'confidence': 0.0, 'available': False}
        
        # Vision Analysis
        vision_result = analysis.get('vision', {})
        if 'error' not in vision_result and vision_result:
            component_scores['vision'] = {
                'score': 1.0 if vision_result.get('is_legal_consensus', False) else 0.0,
                'confidence': vision_result.get('aggregate_confidence', 0.0),
                'available': True
            }
        else:
            component_scores['vision'] = {'score': 0.5, 'confidence': 0.0, 'available': False}
        
        # --- Corrected Logic for Rule-based Validation ---
        # Rule-based Validation
        validation = combined_results.get('validation', {})
        rules_result = validation.get('rules', {})

        if 'error' not in rules_result and 'score' in rules_result:
            component_scores['rules'] = {
                'score': rules_result.get('score', 0.5),
                'confidence': rules_result.get('confidence', 0.0),
                'available': True,
            }
        else:
            component_scores['rules'] = {'score': 0.5, 'confidence': 0.0, 'available': False}
        
        return component_scores
    
    def _calculate_rules_score(self, validation: dict) -> dict:
        """Calculate aggregated score from all rule-based validators based on a corrected structure"""
        rules_overall_result = validation.get('rules', {})
        
        if not rules_overall_result or 'error' in rules_overall_result:
            # If no rules result, return a neutral score
            return {'score': 0.5, 'confidence': 0.0, 'available': False}

        # Extract the overall score and confidence from the main 'rules' result
        overall_score = rules_overall_result.get('overall_score', 0.0)
        overall_confidence = rules_overall_result.get('confidence', 0.0)

        # Return the aggregated result
        return {
            'score': overall_score,
            'confidence': overall_confidence,
            'available': True,
            'validator_results': rules_overall_result.get('rule_results', {}) # Keep detailed results if needed
        }
        """Calculate aggregated score from all rule-based validators"""
        validator_results = {}
        total_score = 0.0
        total_weight = 0.0
        
        # MRZ Validation
        mrz_result = validation.get('mrz', {})
        if 'error' not in mrz_result:
            mrz_valid = mrz_result.get('valid', False)
            mrz_confidence = mrz_result.get('validation_result', {}).get('confidence', 0.0) if mrz_result.get('has_mrz') else 0.8
            validator_results['mrz'] = {'score': 1.0 if mrz_valid else 0.0, 'confidence': mrz_confidence, 'weight': 0.2}
        
        # Clause Validation
        clause_result = validation.get('clauses', {})
        if 'error' not in clause_result:
            clause_score = clause_result.get('overall_score', 0.0)
            clause_confidence = clause_result.get('confidence', 0.0)
            validator_results['clauses'] = {'score': clause_score, 'confidence': clause_confidence, 'weight': 0.3}
        
        # Rules Validation
        rules_result = validation.get('rules', {})
        if 'error' not in rules_result:
            rules_score = rules_result.get('overall_score', 0.0)
            rules_confidence = rules_result.get('confidence', 0.0)
            validator_results['rules'] = {'score': rules_score, 'confidence': rules_confidence, 'weight': 0.3}
        
        # Signature Validation
        sig_result = validation.get('signatures', {})
        if 'error' not in sig_result:
            sig_score = 1.0 if sig_result.get('signatures_valid', False) else 0.0
            sig_confidence = sig_result.get('confidence', 0.0)
            validator_results['signatures'] = {'score': sig_score, 'confidence': sig_confidence, 'weight': 0.2}
        
        # Calculate weighted average
        for validator, result in validator_results.items():
            weight = result['weight']
            score = result['score']
            total_score += score * weight
            total_weight += weight
        
        # Average confidence
        avg_confidence = np.mean([r['confidence'] for r in validator_results.values()]) if validator_results else 0.0
        
        return {
            'score': total_score / total_weight if total_weight > 0 else 0.5,
            'confidence': avg_confidence,
            'available': len(validator_results) > 0,
            'validator_results': validator_results
        }
    
    def _calculate_ensemble_score(self, component_scores: Dict) -> float:
        """Calculate weighted ensemble score"""
        total_score = 0.0
        total_weight = 0.0
        
        for component, weight in self.weights.items():
            if component in component_scores and component_scores[component]['available']:
                score = component_scores[component]['score']
                confidence = component_scores[component]['confidence']
                
                # Weight by confidence
                effective_weight = weight * confidence
                total_score += score * effective_weight
                total_weight += effective_weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _make_binary_decision(self, ensemble_score: float, component_scores: Dict) -> tuple:
        """Make binary legality decision with confidence"""
        
        # Base decision on ensemble score - ENSURE this line exists
        is_legal = ensemble_score > 0.5
        
        # Calculate confidence based on agreement between components
        agreements = []
        confidences = []
        
        for component, result in component_scores.items():
            if result.get('available', False):  # Use .get() for safety
                component_decision = result.get('score', 0.5) > 0.5
                agreements.append(component_decision == is_legal)
                confidences.append(result.get('confidence', 0.0))
        
        # Agreement ratio
        agreement_ratio = sum(agreements) / len(agreements) if agreements else 0.5
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Overall confidence combines ensemble certainty and component agreement
        distance_from_threshold = abs(ensemble_score - 0.5) * 2  # 0 to 1
        confidence = (distance_from_threshold * 0.4 + agreement_ratio * 0.4 + avg_confidence * 0.2)
        confidence = min(max(confidence, 0.0), 1.0)
        
        return is_legal, confidence

        """Make binary legality decision with confidence"""
        
        # Base decision on ensemble score
        is_legal = ensemble_score > 0.5
        
        # Calculate confidence based on agreement between components
        agreements = []
        confidences = []
        
        for component, result in component_scores.items():
            if result['available']:
                component_decision = result['score'] > 0.5
                agreements.append(component_decision == is_legal)
                confidences.append(result['confidence'])
        
        # Agreement ratio
        agreement_ratio = sum(agreements) / len(agreements) if agreements else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Overall confidence combines ensemble certainty and component agreement
        distance_from_threshold = abs(ensemble_score - 0.5) * 2  # 0 to 1
        confidence = (distance_from_threshold * 0.4 + agreement_ratio * 0.4 + avg_confidence * 0.2)
        confidence = min(max(confidence, 0.0), 1.0)
        
        return is_legal, confidence
    
    def _generate_explanation(self, component_scores: Dict, ensemble_score: float, 
                            is_legal: bool, confidence: float) -> Dict:
        """Generate human-readable explanation of the decision"""
        
        # Component contributions
        contributions = []
        for component, result in component_scores.items():
            if result['available']:
                weight = self.weights.get(component, 0.0)
                component_decision = "supports legality" if result['score'] > 0.5 else "questions legality"
                contributions.append(f"{component.upper()} ({weight:.1%} weight): {component_decision} with {result['confidence']:.1%} confidence")
        
        # Overall assessment
        decision_text = "LEGAL" if is_legal else "NOT LEGAL"
        confidence_level = (
            "HIGH" if confidence > self.thresholds['high_confidence']
            else "MEDIUM" if confidence > self.thresholds['medium_confidence']
            else "LOW"
        )
        
        # Key factors
        key_factors = []
        
        # High impact factors
        for component, result in component_scores.items():
            if result['available'] and self.weights.get(component, 0) > 0.25:
                if result['confidence'] > 0.8:
                    if result['score'] > 0.5:
                        key_factors.append(f"Strong {component} indicators support document legality")
                    else:
                        key_factors.append(f"Strong {component} indicators question document legality")
        
        # Agreement analysis
        available_components = [c for c, r in component_scores.items() if r['available']]
        if len(available_components) > 1:
            scores = [component_scores[c]['score'] for c in available_components]
            if max(scores) - min(scores) < 0.2:
                key_factors.append("High agreement between different analysis methods")
            else:
                key_factors.append("Mixed signals from different analysis methods")
        
        return {
            'decision': decision_text,
            'confidence_level': confidence_level,
            'ensemble_score': f"{ensemble_score:.3f}",
            'component_contributions': contributions,
            'key_factors': key_factors,
            'summary': f"Document classified as {decision_text} with {confidence_level} confidence ({confidence:.1%})"
        }
    
    def _assess_risk(self, component_scores: Dict, confidence: float) -> Dict:
        """Assess various risk factors"""
        risk_factors = []
        risk_score = 0.0
        
        # Low confidence risk
        if confidence < self.thresholds['medium_confidence']:
            risk_factors.append("Low confidence in automated analysis")
            risk_score += 0.3
        
        # Component disagreement risk
        available_scores = [r['score'] for r in component_scores.values() if r['available']]
        if len(available_scores) > 1:
            score_variance = np.var(available_scores)
            if score_variance > 0.1:  # High disagreement
                risk_factors.append("Significant disagreement between analysis components")
                risk_score += 0.4
        
        # Missing component risk
        unavailable_components = [c for c, r in component_scores.items() if not r['available']]
        if unavailable_components:
            risk_factors.append(f"Analysis limited due to unavailable components: {', '.join(unavailable_components)}")
            risk_score += len(unavailable_components) * 0.1
        
        # Edge case risk
        for component, result in component_scores.items():
            if result['available'] and abs(result['score'] - 0.5) < 0.1:  # Very close to threshold
                risk_factors.append(f"{component} analysis shows borderline results")
                risk_score += 0.2
        
        risk_level = (
            "HIGH" if risk_score > 0.6
            else "MEDIUM" if risk_score > 0.3
            else "LOW"
        )
        
        return {
            'risk_level': risk_level,
            'risk_score': min(risk_score, 1.0),
            'risk_factors': risk_factors,
            'requires_human_review': risk_score > 0.5
        }
    
    def _generate_recommendations(self, component_scores: Dict, is_legal: bool, 
                                confidence: float, risk_assessment: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on risk level
        if risk_assessment['requires_human_review']:
            recommendations.append("Recommend manual review by legal expert due to analysis uncertainties")
        
        # Based on confidence
        if confidence < self.thresholds['medium_confidence']:
            recommendations.append("Consider obtaining additional document verification")
        
        # Based on specific component issues
        for component, result in component_scores.items():
            if not result['available']:
                if component == 'nlp':
                    recommendations.append("Text analysis unavailable - verify document text quality")
                elif component == 'vision':
                    recommendations.append("Visual analysis unavailable - verify document image quality")
                elif component == 'rules':
                    recommendations.append("Rule validation incomplete - manual compliance check recommended")
        
        # Based on decision outcome
        if not is_legal:
            recommendations.append("Document does not meet legality criteria - detailed legal review required")
            
            # Specific recommendations based on failing components
            for component, result in component_scores.items():
                if result['available'] and result['score'] < 0.5:
                    if component == 'rules':
                        recommendations.append("Document fails rule-based validation - check required legal elements")
                    elif component == 'nlp':
                        recommendations.append("Document language analysis indicates non-legal content")
                    elif component == 'vision':
                        recommendations.append("Visual document analysis suggests authenticity issues")
        
        # Quality recommendations
        if confidence > self.thresholds['high_confidence'] and is_legal:
            recommendations.append("Document analysis shows high confidence - suitable for automated processing")
        
        return list(set(recommendations))  # Remove duplicates
