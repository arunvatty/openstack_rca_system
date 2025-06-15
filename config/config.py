import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    
    # File paths
    DATA_DIR = 'logs'
    MODELS_DIR = 'saved_models'
    
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
        'time_window_minutes': 30
    }
    
    # Streamlit Configuration
    STREAMLIT_CONFIG = {
        'page_title': 'CloudTracer RCA Assistant',
        'page_icon': '🔍',
        'layout': 'wide'
    }