import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import pickle
import joblib
from typing import Dict, List, Tuple, Any
import warnings
import os
import yaml

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- NLP LAYER IMPORTS (These will be needed in the main execution block) ---
from nlp.preprocess import TextPreprocessor
from nlp.vectorize import TextVectorizer

class DocumentClassifier:
    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.best_model = None
        
    def prepare_features(self, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine all features into a single matrix"""
        feature_matrices = []
        
        # Add each feature type
        for feature_name, features in feature_dict.items():
            if features is not None and len(features) > 0:
                feature_matrices.append(features)
        
        # Concatenate all features
        if feature_matrices:
            combined_features = np.hstack(feature_matrices)
        else:
            raise ValueError("No features provided")
        
        return combined_features
    
    def train_individual_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train individual classification models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        models_config = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaled': False
            },
            'xgboost': {
                'model': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                ),
                'use_scaled': False
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                ),
                'use_scaled': False
            },
            'logistic_regression': {
                'model': LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42
                ),
                'use_scaled': True
            },
            'svm': {
                'model': SVC(
                    C=1.0,
                    kernel='rbf',
                    probability=True,
                    random_state=42
                ),
                'use_scaled': True
            }
        }
        
        results = {}
        
        for model_name, config in models_config.items():
            print(f"Training {model_name}...")
            
            model = config['model']
            
            # Choose scaled or unscaled features
            if config['use_scaled']:
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            else:
                X_train_model = X_train
                X_test_model = X_test
            
            # Train model
            model.fit(X_train_model, y_train)
            
            # Predictions
            train_pred = model.predict(X_train_model)
            test_pred = model.predict(X_test_model)
            
            # Probabilities for AUC
            try:
                train_proba = model.predict_proba(X_train_model)[:, 1]
                test_proba = model.predict_proba(X_test_model)[:, 1]
                train_auc = roc_auc_score(y_train, train_proba)
                test_auc = roc_auc_score(y_test, test_proba)
            except:
                train_auc = test_auc = 0.0
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_model, y_train, cv=5)
            
            # Store results
            results[model_name] = {
                'model': model,
                'train_accuracy': (train_pred == y_train).mean(),
                'test_accuracy': (test_pred == y_test).mean(),
                'train_auc': train_auc,
                'test_auc': test_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'use_scaled': config['use_scaled']
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                self.feature_importance[model_name] = np.abs(model.coef_[0])
        
        self.models = {name: result['model'] for name, result in results.items()}
        return results
    
    def create_ensemble(self, X: np.ndarray, y: np.ndarray, models_results: Dict) -> Dict:
        """Create ensemble model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Get predictions from all models
        train_predictions = {}
        test_predictions = {}
        
        for model_name, result in models_results.items():
            model = result['model']
            use_scaled = result['use_scaled']
            
            if use_scaled:
                X_train_model = self.scalers['main'].transform(X_train)
                X_test_model = self.scalers['main'].transform(X_test)
            else:
                X_train_model = X_train
                X_test_model = X_test
            
            train_predictions[model_name] = model.predict_proba(X_train_model)[:, 1]
            test_predictions[model_name] = model.predict_proba(X_test_model)[:, 1]
        
        # Create ensemble features
        ensemble_train = np.column_stack(list(train_predictions.values()))
        ensemble_test = np.column_stack(list(test_predictions.values()))
        
        # Train meta-model
        meta_model = LogisticRegression(random_state=42)
        meta_model.fit(ensemble_train, y_train)
        
        # Evaluate ensemble
        ensemble_pred = meta_model.predict(ensemble_test)
        ensemble_proba = meta_model.predict_proba(ensemble_test)[:, 1]
        
        ensemble_accuracy = (ensemble_pred == y_test).mean()
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)
        
        self.models['ensemble'] = meta_model
        
        return {
            'meta_model': meta_model,
            'accuracy': ensemble_accuracy,
            'auc': ensemble_auc,
            'model_weights': meta_model.coef_[0]
        }
    
    def select_best_model(self, results: Dict, ensemble_result: Dict) -> str:
        """Select the best performing model"""
        best_score = 0
        best_model_name = None
        
        # Check individual models
        for model_name, result in results.items():
            score = result['test_auc']
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        # Check ensemble
        if ensemble_result['auc'] > best_score:
            best_model_name = 'ensemble'
        
        self.best_model = best_model_name
        return best_model_name
    
    def train(self, feature_dict: Dict[str, np.ndarray], labels: np.ndarray) -> Dict:
        """Complete training pipeline"""
        # Prepare features
        X = self.prepare_features(feature_dict)
        
        print(f"Training with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Train individual models
        results = self.train_individual_models(X, labels)
        
        # Create ensemble
        ensemble_result = self.create_ensemble(X, labels, results)
        
        # Select best model
        best_model = self.select_best_model(results, ensemble_result)
        
        # Print results
        print("\nModel Performance:")
        print("-" * 50)
        for name, result in results.items():
            print(f"{name}: Test AUC = {result['test_auc']:.4f}, CV = {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        print(f"Ensemble: Test AUC = {ensemble_result['auc']:.4f}")
        print(f"\nBest model: {best_model}")
        
        return {
            'individual_results': results,
            'ensemble_result': ensemble_result,
            'best_model': best_model,
            'feature_shape': X.shape
        }
    
    def save_models(self, path: str):
        """Save all trained models"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'best_model': self.best_model
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_models(self, path: str):
        """Load trained models"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_importance = model_data['feature_importance']
        self.best_model = model_data['best_model']

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Define paths
    data_dir = config['paths']['data_dir']
    models_dir = config['paths']['models_dir']
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    # NOTE: You need to have a training data CSV file here.
    # The example uses 'training_data.csv' in the 'data' directory.
    # Replace this with your actual data loading logic.
    try:
        df = pd.read_csv(os.path.join(data_dir, 'training_data.csv'))
        texts = df['text'].tolist()
        labels = df['label'].values
    except FileNotFoundError:
        print("Error: 'training_data.csv' not found. Please create it first.")
        print("Exiting NLP model training.")
        exit()

    preprocessor = TextPreprocessor()
    processed_texts = [preprocessor.preprocess_for_classification(text) for text in texts]
    
    # Vectorize data
    print("Vectorizing data...")
    vectorizer = TextVectorizer(config['model']['nlp'])
    features = vectorizer.fit_transform(texts, processed_texts)
    
    # Initialize classifier
    classifier = DocumentClassifier(config['model']['nlp'])
    
    # Run training pipeline
    print("Starting NLP training pipeline...")
    training_results = classifier.train(features, labels)
    
    # Save models
    os.makedirs(models_dir, exist_ok=True)
    classifier.save_models(os.path.join(models_dir, 'nlp_model.pkl'))
    vectorizer.save_vectorizers(os.path.join(models_dir, 'vectorizer.pkl'))
    
    print("\nNLP model training and saving completed successfully!")
