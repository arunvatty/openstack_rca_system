import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    GEMINI_SERVICE_ACCOUNT_PATH = os.getenv('GEMINI_SERVICE_ACCOUNT_PATH', 'gemini-service-account.json')
    
    # AI Provider Selection (claude or gemini)
    AI_PROVIDER = os.getenv('AI_PROVIDER', 'claude').lower()
    
    # File paths
    DATA_DIR = 'logs'
    MODELS_DIR = 'models'
    CACHE_DIR = 'data/cache'
    VECTOR_DB_DIR = 'data/vector_db'
    
    # LSTM Model Configuration
    LSTM_CONFIG = {
        'max_sequence_length': 100,
        'embedding_dim': 128,
        'lstm_units': 64,
        'dropout_rate': 0.2,
        'batch_size': 32,
        'epochs': 50,
        'validation_split': 0.2
    }
    
    # Log Processing Configuration
    LOG_CONFIG = {
        'important_keywords': [
            'ERROR', 'CRITICAL', 'FAILED', 'EXCEPTION', 'TIMEOUT',
            'CONNECTION_LOST', 'UNAVAILABLE', 'DENIED', 'REJECTED',
            'SPAWNING', 'TERMINATING', 'DESTROYED', 'CLAIM', 'RESOURCE'
        ],
        'service_patterns': {
            'nova-api': r'nova-api\.log',
            'nova-compute': r'nova-compute\.log',
            'nova-scheduler': r'nova-scheduler\.log'
        }
    }
    
    # RCA Configuration
    RCA_CONFIG = {
        'similarity_threshold': 0.7,
        'max_context_logs': 50,
        'time_window_minutes': 30,
        'historical_context_size': 10,  # Number of historical logs to include in context
        'max_historical_context_chars': 2000  # Maximum characters for historical context
    }
    
    # NEW: Vector DB Configuration
    VECTOR_DB_CONFIG = {
        'type': 'chroma',
        'embedding_model': 'all-MiniLM-L12-v2',  # Upgraded to L12 for better semantic understanding
        'collection_name': 'openstack_logs',
        'similarity_threshold': 0.7,
        'top_k_results': 20,
        'persist_directory': VECTOR_DB_DIR,
        
        # Additional parameters for enhanced configuration
        'chunk_size': 512,  # For text chunking if needed
        'chunk_overlap': 50,  # Overlap between chunks
        'embedding_dimensions': 384,  # Explicit dimension setting
        'distance_metric': 'cosine',  # Distance metric (cosine, euclidean, etc.)
        'max_text_length': 1000,  # Maximum text length for embedding
    }
    
    # AI Model Configuration
    AI_CONFIG = {
        'claude': {
            'model': 'claude-3-5-sonnet-20241022',
            'max_tokens': 2000,
            'temperature': 0.1
        },
        'gemini': {
            'model': 'gemini-1.5-pro',
            'max_tokens': 2000,
            'temperature': 0.1
        }
    }
    
    # Streamlit Configuration
    STREAMLIT_CONFIG = {
        'page_title': 'CloudTracer RCA Assistant',
        'page_icon': '🔍',
        'layout': 'wide'
    }