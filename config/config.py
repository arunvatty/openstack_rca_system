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
    
    # MLflow Configuration
    MLFLOW_CONFIG = {
        # MLflow Tracking Server
        'tracking_uri': os.getenv('MLFLOW_TRACKING_URI'),
        'experiment_name': 'openstack_rca_system_staging',
        
        # S3 Artifact Store (set these environment variables)
        'artifact_root': os.getenv('MLFLOW_ARTIFACT_ROOT', 's3://chandanbam-bucket/group6-capstone'),
        's3_endpoint_url': os.getenv('MLFLOW_S3_ENDPOINT_URL'),
        'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        's3_bucket': os.getenv('MLFLOW_S3_BUCKET', 'chandanbam-bucket'),
        
        # Model Registry & Versioning
        'model_registry_uri': os.getenv('MLFLOW_MODEL_REGISTRY_URI'),
        'model_name_prefix': 'openstack_rca',
        'auto_register_model': True,  # Automatically register models
        'default_model_stage': 'Staging',  # Default stage for new models
        'auto_promote_threshold': 0.85,  # Auto-promote to Production if accuracy > threshold
        
        # Model Lifecycle Management
        'max_model_versions': 10,  # Maximum versions to keep per model
        'archive_old_versions': True,  # Auto-archive old versions
        'production_approval_required': False,  # Require manual approval for Production stage
        
        # Run Configuration
        'auto_log': True,  # Enable automatic logging
        'log_models': True,  # Log models automatically
        'log_artifacts': True,  # Log artifacts automatically
        'tags': {
            'project': 'openstack_rca_system',
            'team': 'mlops',
            'version': '2.0.0',
            'environment': os.getenv('ENVIRONMENT', 'development')
        },
        
        # Enhanced Logging Configuration
        'log_system_metrics': True,  # Log system metrics (CPU, memory, etc.)
        'log_input_examples': True,  # Log input examples for model inference
        'log_model_signature': True,  # Log model signature
        'log_conda_env': True,  # Log conda environment
        'log_pip_requirements': True,  # Log pip requirements
        
        # Performance Tracking
        'track_training_time': True,
        'track_inference_time': True,
        'log_feature_importance': True,
        'log_model_metrics': True,  # Log detailed model metrics
        'log_hyperparameters': True,  # Log all hyperparameters
        
        # Integration Settings
        'enable_ui_integration': True,  # Enable MLflow in Streamlit UI
        'enable_auto_logging': True,  # Enable automatic parameter/metric logging
        'enable_model_serving': True,  # Enable model serving endpoints
        
        # Versioning Strategy
        'versioning_strategy': 'semantic',  # 'semantic', 'timestamp', or 'incremental'
        'version_format': 'v{major}.{minor}.{patch}',  # Version format template
        'auto_increment': True,  # Auto-increment version numbers
        
        # S3 Configuration
        's3_model_prefix': 'models/openstack_rca',  # S3 prefix for model artifacts
        's3_experiment_prefix': 'experiments',  # S3 prefix for experiment artifacts
        's3_backup_enabled': True,  # Enable S3 backup for critical models
        
        # Model Deployment
        'deployment_targets': ['staging', 'production'],  # Available deployment targets
        'auto_deploy_staging': True,  # Auto-deploy to staging
        'auto_deploy_production': False,  # Require manual deployment to production
        
        # Monitoring & Alerts
        'enable_model_monitoring': True,  # Enable model performance monitoring
        'alert_on_drift': True,  # Alert on model drift
        'performance_threshold': 0.80,  # Alert if performance drops below threshold
        
        # Backup & Recovery
        'backup_frequency': 'daily',  # Backup frequency (daily, weekly, monthly)
        'backup_retention_days': 30,  # Days to retain backups
        'enable_disaster_recovery': True,  # Enable disaster recovery procedures
    }
    
    # MLflow Tracking URI (for backward compatibility)
    MLFLOW_TRACKING_URI = MLFLOW_CONFIG['tracking_uri']
    
    # Streamlit Configuration
    STREAMLIT_CONFIG = {
        'page_title': 'CloudTracer RCA Assistant',
        'page_icon': '🔍',
        'layout': 'wide'
    }