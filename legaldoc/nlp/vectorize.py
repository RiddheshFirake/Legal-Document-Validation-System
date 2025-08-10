import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict, Tuple, Optional
import pickle

class TextVectorizer:
    def __init__(self, config: dict):
        self.config = config
        self.tfidf_vectorizer = None
        self.lda_model = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.fitted = False
        
    def initialize_bert(self, model_name: str = "distilbert-base-uncased"):
        """Initialize BERT model for embeddings"""
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.bert_model.eval()
    
    def fit_tfidf(self, texts: List[str], max_features: int = 10000):
        """Fit TF-IDF vectorizer"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        self.tfidf_vectorizer.fit(texts)
    
    def fit_lda(self, texts: List[str], n_topics: int = 20):
        """Fit LDA topic model"""
        # Use CountVectorizer for LDA
        count_vectorizer = CountVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        doc_term_matrix = count_vectorizer.fit_transform(texts)
        
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        self.lda_model.fit(doc_term_matrix)
        self.count_vectorizer = count_vectorizer
    
    def get_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Get TF-IDF features"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted")
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def get_topic_features(self, texts: List[str]) -> np.ndarray:
        """Get topic distribution features"""
        if self.lda_model is None:
            raise ValueError("LDA model not fitted")
        
        doc_term_matrix = self.count_vectorizer.transform(texts)
        return self.lda_model.transform(doc_term_matrix)
    
    def get_bert_embeddings(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """Get BERT embeddings"""
        if self.bert_model is None:
            self.initialize_bert()
        
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.bert_tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True,
                    padding=True
                )
                
                # Get embeddings
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding.flatten())
        
        return np.array(embeddings)
    
    def extract_statistical_features(self, processed_data: List[Dict]) -> np.ndarray:
        """Extract statistical features from preprocessed data"""
        features = []
        
        for data in processed_data:
            feature_dict = data.get('features', {})
            entities = data.get('entities', {})
            
            # Basic text features
            text_features = [
                feature_dict.get('char_count', 0),
                feature_dict.get('word_count', 0),
                feature_dict.get('sentence_count', 0),
                feature_dict.get('avg_word_length', 0),
                feature_dict.get('avg_sentence_length', 0),
                feature_dict.get('legal_density', 0),
                feature_dict.get('lexical_diversity', 0)
            ]
            
            # Entity features
            entity_features = [
                len(entities.get('dates', [])),
                len(entities.get('currency', [])),
                len(entities.get('clauses', [])),
                len(entities.get('references', [])),
                len(entities.get('signatures', [])),
                len(entities.get('parties', []))
            ]
            
            # Named entity features
            named_entities = entities.get('named_entities', {})
            ne_features = [
                len(named_entities.get('PERSON', [])),
                len(named_entities.get('ORG', [])),
                len(named_entities.get('GPE', [])),
                len(named_entities.get('MONEY', [])),
                len(named_entities.get('DATE', []))
            ]
            
            features.append(text_features + entity_features + ne_features)
        
        return np.array(features)
    
    def fit_transform(self, texts: List[str], processed_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Fit all vectorizers and transform texts"""
        # Fit vectorizers
        self.fit_tfidf(texts)
        self.fit_lda(texts)
        
        # Get all features
        features = {
            'tfidf': self.get_tfidf_features(texts),
            'topics': self.get_topic_features(texts),
            'bert': self.get_bert_embeddings(texts),
            'statistical': self.extract_statistical_features(processed_data)
        }
        
        self.fitted = True
        return features
    
    def transform(self, texts: List[str], processed_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Transform texts using fitted vectorizers"""
        if not self.fitted:
            raise ValueError("Vectorizers not fitted")
        
        features = {
            'tfidf': self.get_tfidf_features(texts),
            'topics': self.get_topic_features(texts),
            'bert': self.get_bert_embeddings(texts),
            'statistical': self.extract_statistical_features(processed_data)
        }
        
        return features
    
    def save_vectorizers(self, path: str):
        """Save fitted vectorizers"""
        vectorizers = {
            'tfidf': self.tfidf_vectorizer,
            'lda': self.lda_model,
            'count_vectorizer': getattr(self, 'count_vectorizer', None),
            'fitted': self.fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(vectorizers, f)
    
    def load_vectorizers(self, path: str):
        """Load fitted vectorizers"""
        with open(path, 'rb') as f:
            vectorizers = pickle.load(f)
        
        self.tfidf_vectorizer = vectorizers['tfidf']
        self.lda_model = vectorizers['lda']
        self.count_vectorizer = vectorizers['count_vectorizer']
        self.fitted = vectorizers['fitted']
