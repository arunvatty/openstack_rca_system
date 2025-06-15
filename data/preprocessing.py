import pandas as pd
import numpy as np
import re
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogPreprocessor:
    """Preprocessor for OpenStack log data"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Important keywords for classification
        self.important_keywords = [
            'error', 'critical', 'failed', 'exception', 'timeout',
            'connection_lost', 'unavailable', 'denied', 'rejected',
            'spawning', 'terminating', 'destroyed', 'claim', 'resource',
            'attempting', 'successful', 'instance', 'vm', 'hypervisor'
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers (keep alphanumeric and spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text"""
        if not text:
            return []
        
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def create_importance_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create binary labels for log importance"""
        labels = []
        
        for _, row in df.iterrows():
            message = str(row.get('message', '')).lower()
            level = str(row.get('level', '')).lower()
            
            # Check for important keywords
            has_important_keyword = any(keyword in message for keyword in self.important_keywords)
            
            # Check log level
            is_important_level = level in ['error', 'critical', 'warning']
            
            # Check for specific patterns
            has_instance_action = any(pattern in message for pattern in [
                'spawning', 'terminating', 'destroyed', 'instance spawned',
                'instance destroyed', 'vm started', 'vm stopped', 'vm paused'
            ])
            
            has_resource_issue = any(pattern in message for pattern in [
                'claim', 'resource', 'disk', 'memory', 'vcpu', 'attempting claim'
            ])
            
            # Label as important if any condition is met
            is_important = (has_important_keyword or is_important_level or 
                          has_instance_action or has_resource_issue)
            
            labels.append(1 if is_important else 0)
        
        return pd.Series(labels)
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from log data"""
        logger.info("Extracting features from log data...")
        
        features_df = df.copy()
        
        # Text features
        features_df['cleaned_message'] = features_df['message'].apply(self.clean_text)
        features_df['message_length'] = features_df['message'].str.len()
        features_df['word_count'] = features_df['cleaned_message'].str.split().str.len()
        
        # Temporal features
        if 'timestamp' in features_df.columns:
            features_df['hour'] = pd.to_datetime(features_df['timestamp']).dt.hour
            features_df['day_of_week'] = pd.to_datetime(features_df['timestamp']).dt.dayofweek
            features_df['minute'] = pd.to_datetime(features_df['timestamp']).dt.minute
        
        # Service features
        if 'service_type' not in features_df.columns:
            features_df['service_type'] = 'unknown'
        
        # Categorical encoding
        categorical_columns = ['level', 'service_type', 'module']
        for col in categorical_columns:
            if col in features_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        features_df[col].fillna('unknown')
                    )
                else:
                    features_df[f'{col}_encoded'] = self.label_encoders[col].transform(
                        features_df[col].fillna('unknown')
                    )
        
        # Boolean features
        features_df['has_instance_id'] = features_df['instance_id'].notna().astype(int)
        features_df['has_request_id'] = features_df['request_id'].notna().astype(int)
        
        # Keyword features
        for keyword in self.important_keywords[:10]:  # Top 10 keywords
            features_df[f'has_{keyword}'] = features_df['cleaned_message'].str.contains(
                keyword, case=False, na=False
            ).astype(int)
        
        logger.info(f"Extracted {len(features_df.columns)} features")
        return features_df
    
    def prepare_lstm_data(self, df: pd.DataFrame, max_sequence_length: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model"""
        logger.info("Preparing data for LSTM model...")
        
        # Create importance labels
        labels = self.create_importance_labels(df)
        
        # Prepare text sequences
        texts = df['message'].fillna('').astype(str).tolist()
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Fit TF-IDF vectorizer if not already fitted
        if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(cleaned_texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(cleaned_texts)
        
        # Convert to dense array and pad sequences
        tfidf_dense = tfidf_matrix.toarray()
        
        # Pad or truncate sequences to max_sequence_length
        if tfidf_dense.shape[1] > max_sequence_length:
            sequences = tfidf_dense[:, :max_sequence_length]
        else:
            sequences = np.pad(
                tfidf_dense, 
                ((0, 0), (0, max_sequence_length - tfidf_dense.shape[1])), 
                mode='constant'
            )
        
        logger.info(f"Prepared {len(sequences)} sequences for LSTM training")
        logger.info(f"Positive samples: {labels.sum()}, Negative samples: {len(labels) - labels.sum()}")
        
        return sequences, labels.values
    
    def prepare_features_for_training(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for traditional ML models"""
        logger.info("Preparing features for traditional ML models...")
        
        # Extract features
        features_df = self.extract_features(df)
        
        # Select numerical features
        numerical_features = [
            'message_length', 'word_count', 'hour', 'day_of_week', 'minute',
            'level_encoded', 'service_type_encoded', 'module_encoded',
            'has_instance_id', 'has_request_id'
        ]
        
        # Add keyword features
        keyword_features = [f'has_{keyword}' for keyword in self.important_keywords[:10]]
        numerical_features.extend(keyword_features)
        
        # Filter existing features
        available_features = [col for col in numerical_features if col in features_df.columns]
        
        X = features_df[available_features].fillna(0)
        y = self.create_importance_labels(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Prepared {X_scaled.shape[1]} features for training")
        
        return X_scaled, y.values