import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Optional
import string

class TextPreprocessor:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Initialize tools
        self.nlp = spacy.load('en_core_web_sm')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Legal document patterns
        self.legal_patterns = {
            'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'currency': r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            'clauses': r'(?:section|clause|article|paragraph)\s+\d+',
            'references': r'(?:see|refer to|as defined in)\s+(?:section|clause|article)\s+\d+',
            'signatures': r'(?:signed|executed|dated)\s+(?:this|on)\s+\d{1,2}[^\w]*(?:day|st|nd|rd|th)',
            'parties': r'(?:party|parties|plaintiff|defendant|licensor|licensee)'
        }
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep legal punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/]', ' ', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)

        
        return text.strip()
    
    def extract_legal_entities(self, text: str) -> Dict:
        """Extract legal entities and patterns"""
        entities = {}
        
        for pattern_name, pattern in self.legal_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[pattern_name] = matches
        
        # Use spaCy for named entity recognition
        doc = self.nlp(text)
        spacy_entities = {}
        for ent in doc.ents:
            if ent.label_ not in spacy_entities:
                spacy_entities[ent.label_] = []
            spacy_entities[ent.label_].append(ent.text)
        
        entities['named_entities'] = spacy_entities
        return entities
    
    def tokenize_and_clean(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenize and clean text"""
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract and clean sentences"""
        sentences = sent_tokenize(text)
        cleaned_sentences = []
        
        for sentence in sentences:
            cleaned = self.clean_text(sentence)
            if len(cleaned) > 10:  # Filter very short sentences
                cleaned_sentences.append(cleaned)
        
        return cleaned_sentences
    
    def calculate_text_features(self, text: str) -> Dict:
        """Calculate various text features"""
        sentences = self.extract_sentences(text)
        tokens = self.tokenize_and_clean(text, remove_stopwords=False)
        
        # Basic statistics
        features = {
            'char_count': len(text),
            'word_count': len(tokens),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0,
            'avg_sentence_length': sum(len(sent.split()) for sent in sentences) / len(sentences) if sentences else 0
        }
        
        # Legal document indicators
        legal_keywords = [
            'agreement', 'contract', 'terms', 'conditions', 'hereby', 'whereas',
            'party', 'parties', 'shall', 'therefore', 'pursuant', 'provision',
            'clause', 'section', 'article', 'liability', 'damages', 'breach'
        ]
        
        legal_word_count = sum(1 for token in tokens if token in legal_keywords)
        features['legal_density'] = legal_word_count / len(tokens) if tokens else 0
        
        # Complexity measures
        features['lexical_diversity'] = len(set(tokens)) / len(tokens) if tokens else 0
        
        return features
    
    def preprocess_for_classification(self, text: str) -> Dict:
        """Complete preprocessing pipeline for classification"""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Extract features
        features = self.calculate_text_features(cleaned_text)
        
        # Extract entities
        entities = self.extract_legal_entities(cleaned_text)
        
        # Get processed tokens
        tokens = self.tokenize_and_clean(cleaned_text)
        
        # Get sentences
        sentences = self.extract_sentences(cleaned_text)
        
        return {
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'sentences': sentences,
            'features': features,
            'entities': entities,
            'processed_text': ' '.join(tokens)
        }
