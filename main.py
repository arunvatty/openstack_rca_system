"""
OpenStack RCA System - Main Entry Point
Interactive Root Cause Analysis system for OpenStack logs using LSTM and Claude API
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Disable ChromaDB telemetry to prevent errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import Config
from data.log_ingestion import LogIngestionManager
from data.preprocessing import LogPreprocessor
from models.lstm_classifier import LSTMLogClassifier
from models.rca_analyzer import RCAAnalyzer
from utils.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories"""
    directories = [
        Config.DATA_DIR,
        Config.MODELS_DIR,
        'logs/sample_logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def train_model_pipeline(log_files_path: str = None, clean_vector_db: bool = False):
    """Train the LSTM model pipeline with optional ChromaDB cleanup"""
    logger.info("Starting model training pipeline...")
    
    # Step 1: Initialize components
    ingestion_manager = LogIngestionManager(Config.DATA_DIR)
    preprocessor = LogPreprocessor()
    feature_engineer = FeatureEngineer()
    
    # NEW: Clean ChromaDB if requested
    if clean_vector_db:
        logger.info("Cleaning ChromaDB before training...")
        try:
            from services.vector_db_service import VectorDBService
            vector_db = VectorDBService()
            vector_db.clear_collection()
            logger.info("✅ ChromaDB cleaned successfully")
            
            # Reinitialize the ingestion manager to get fresh vector DB connection
            ingestion_manager = LogIngestionManager(Config.DATA_DIR)
            logger.info("✅ Reinitialized ingestion manager with clean ChromaDB")
            
        except Exception as e:
            logger.warning(f"Failed to clean ChromaDB: {e}")
    
    # Step 2: Ingest and preprocess data
    logger.info("Ingesting log data...")
    if log_files_path:
        df = ingestion_manager.ingest_from_directory(log_files_path)
    else:
        df = ingestion_manager.ingest_multiple_files()
    
    if df.empty:
        logger.error("No log data found for training")
        return None
    
    # Apply feature engineering
    logger.info("Applying feature engineering...")
    df = feature_engineer.engineer_all_features(df)
    
    # Step 3: Prepare data for LSTM
    logger.info("Preparing data for LSTM training...")
    X, y = preprocessor.prepare_lstm_data(df, Config.LSTM_CONFIG['max_sequence_length'])
    
    # Step 4: Train LSTM model
    logger.info("Training LSTM model...")
    lstm_classifier = LSTMLogClassifier(Config.LSTM_CONFIG)
    results = lstm_classifier.train(X, y)
    
    # Step 5: Save model
    model_path = os.path.join(Config.MODELS_DIR, 'lstm_log_classifier.keras')
    lstm_classifier.save_model(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Print training results
    logger.info("Training Results:")
    logger.info(f"Validation Accuracy: {results['val_accuracy']:.4f}")
    logger.info(f"Validation Precision: {results['val_precision']:.4f}")
    logger.info(f"Validation Recall: {results['val_recall']:.4f}")
    
    # NEW: Print ChromaDB status
    try:
        from services.vector_db_service import VectorDBService
        vector_db = VectorDBService()
        stats = vector_db.get_collection_stats()
        logger.info(f"ChromaDB Status: {stats['total_documents']} documents")
    except Exception as e:
        logger.warning(f"Could not get ChromaDB stats: {e}")
    
    return lstm_classifier

def run_rca_analysis(issue_description: str, log_files_path: str = None, fast_mode: bool = False):
    """Run RCA analysis on a specific issue using Hybrid RCA Analyzer"""
    logger.info("Starting hybrid RCA analysis...")
    
    # Check for API key
    if not Config.ANTHROPIC_API_KEY:
        logger.error("Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable.")
        return
    
    # Initialize log cache
    from utils.log_cache import LogCache
    log_cache = LogCache()
    
    # Load or train LSTM model
    model_path = os.path.join(Config.MODELS_DIR, 'lstm_log_classifier.keras')
    lstm_model = None
    
    if os.path.exists(model_path):
        logger.info("Loading existing LSTM model...")
        lstm_model = LSTMLogClassifier(Config.LSTM_CONFIG)
        lstm_model.load_model(model_path)
    else:
        logger.warning("No trained model found. Training new model...")
        lstm_model = train_model_pipeline(log_files_path)
    
    # Initialize Hybrid RCA analyzer
    from models.hybrid_rca_analyzer import HybridRCAAnalyzer
    rca_analyzer = HybridRCAAnalyzer(Config.ANTHROPIC_API_KEY, lstm_model)
    
    # Get cached logs or load from files
    if log_files_path:
        logs_df = log_cache.get_cached_logs(log_files_path)
    else:
        logs_df = log_cache.get_cached_logs(Config.DATA_DIR)
    
    if logs_df.empty:
        logger.error("No log data found for analysis.")
        return
    
    # Perform RCA analysis
    logger.info(f"Analyzing issue: {issue_description}")
    results = rca_analyzer.analyze_issue(issue_description, logs_df, fast_mode=fast_mode)
    
    # Display results
    print("\n" + "="*50)
    print("HYBRID RCA ANALYSIS RESULTS")
    print("="*50)
    print(f"Issue: {issue_description}")
    print(f"Category: {results['issue_category']}")
    print(f"Relevant Logs: {results['relevant_logs_count']}")
    print(f"Analysis Mode: {results['analysis_mode']}")
    
    # Display performance metrics
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        print(f"\nPerformance Metrics:")
        print(f"- Processing Time: {metrics.get('processing_time', 0):.2f} seconds")
        print(f"- Total Logs: {metrics.get('total_logs', 0)}")
        print(f"- Filtered Logs: {metrics.get('filtered_logs', 0)}")
        print(f"- LSTM Available: {metrics.get('lstm_available', False)}")
        print(f"- Vector DB Available: {metrics.get('vector_db_available', False)}")
    
    print("\n" + results['root_cause_analysis'])
    
    if results['recommendations']:
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
    
    # Display top filtered logs with scores
    if 'filtered_logs' in results and not results['filtered_logs'].empty:
        print(f"\nTOP RELEVANT LOGS (showing first 5):")
        top_logs = results['filtered_logs'].head(5)
        for _, log in top_logs.iterrows():
            score = log.get('combined_score', log.get('lstm_importance', 0))
            print(f"- [{log.get('level', 'INFO')}] {log.get('service_type', 'unknown')}: {log.get('message', '')[:100]}... (Score: {score:.3f})")
    
    return results

def setup_openstack_log():
    """Set up the OpenStack_2k.log file for processing"""
    # Check if OpenStack_2k.log exists in the project directory
    openstack_log_path = Path('OpenStack_2k.log')
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    target_path = logs_dir / 'OpenStack_2k.log'
    
    if openstack_log_path.exists():
        # Copy the file to logs directory if it exists in project root
        import shutil
        shutil.copy2(openstack_log_path, target_path)
        logger.info(f"Copied OpenStack_2k.log to {target_path}")
        return str(target_path)
    elif target_path.exists():
        # File already exists in logs directory
        logger.info(f"Using existing OpenStack_2k.log at {target_path}")
        return str(target_path)
    else:
        logger.error("OpenStack_2k.log not found. Please ensure the file is in the project directory.")
        logger.info("Expected locations:")
        logger.info(f"  - {openstack_log_path.absolute()}")
        logger.info(f"  - {target_path.absolute()}")
        return None

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='OpenStack RCA System')
    parser.add_argument('--mode', choices=['train', 'analyze', 'streamlit', 'setup', 'vector-db'], 
                       default='streamlit', help='Operation mode')
    parser.add_argument('--logs', type=str, help='Path to log files directory')
    parser.add_argument('--issue', type=str, 
                       help='Issue description for RCA analysis')
    parser.add_argument('--api-key', type=str, 
                       help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--clean-vector-db', action='store_true',
                       help='Clean ChromaDB before training (removes all existing data)')
    parser.add_argument('--vector-db-action', choices=['status', 'clean', 'reset'],
                       help='Vector DB action (for vector-db mode)')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Use fast mode for analysis (skips vector DB for speed)')
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ['ANTHROPIC_API_KEY'] = args.api_key
    
    # Setup directories
    setup_directories()
    
    if args.mode == 'setup':
        logger.info("Setting up OpenStack_2k.log file...")
        log_file = setup_openstack_log()
        if log_file:
            logger.info(f"OpenStack log file ready at: {log_file}")
            logger.info("You can now run: python main.py --mode train --logs logs")
        else:
            logger.error("Setup failed. Please place OpenStack_2k.log in the project directory.")
        
    elif args.mode == 'vector-db':
        logger.info("Vector Database Management Mode")
        
        try:
            from services.vector_db_service import VectorDBService
            vector_db = VectorDBService()
            
            if args.vector_db_action == 'status':
                stats = vector_db.get_collection_stats()
                print("\n" + "="*50)
                print("CHROMADB STATUS")
                print("="*50)
                print(f"Collection Name: {stats.get('collection_name', 'N/A')}")
                print(f"Total Documents: {stats.get('total_documents', 0)}")
                print(f"Chunked Documents: {stats.get('chunked_documents', 0)}")
                print(f"Non-chunked Documents: {stats.get('non_chunked_documents', 0)}")
                print(f"Embedding Model: {stats.get('embedding_model', 'N/A')}")
                print(f"Embedding Dimensions: {stats.get('embedding_dimensions', 'N/A')}")
                print(f"Distance Metric: {stats.get('distance_metric', 'N/A')}")
                print("="*50)
                
            elif args.vector_db_action == 'clean':
                logger.info("Cleaning ChromaDB collection...")
                vector_db.clear_collection()
                logger.info("✅ ChromaDB collection cleaned successfully")
                
            elif args.vector_db_action == 'reset':
                logger.info("Resetting ChromaDB database...")
                vector_db._reset_chroma_db()
                logger.info("✅ ChromaDB database reset successfully")
                
            else:
                logger.error("Please specify --vector-db-action: status, clean, or reset")
                logger.info("Examples:")
                logger.info("  python main.py --mode vector-db --vector-db-action status")
                logger.info("  python main.py --mode vector-db --vector-db-action clean")
                logger.info("  python main.py --mode vector-db --vector-db-action reset")
                
        except Exception as e:
            logger.error(f"Vector DB operation failed: {e}")
    
    elif args.mode == 'train':
        logger.info("Starting model training...")
        
        # If no logs path specified, try to use the OpenStack_2k.log
        if not args.logs:
            log_file = setup_openstack_log()
            if log_file:
                args.logs = 'logs'  # Use the logs directory
            else:
                logger.error("No log files found. Please run: python main.py --mode setup")
                return
        
        # NEW: Show ChromaDB cleanup status
        if args.clean_vector_db:
            logger.info("🔄 ChromaDB will be cleaned before training")
        else:
            logger.info("📊 ChromaDB will retain existing data (use --clean-vector-db to reset)")
        
        model = train_model_pipeline(args.logs, clean_vector_db=args.clean_vector_db)
        if model:
            logger.info("Model training completed successfully!")
        else:
            logger.error("Model training failed!")
    
    elif args.mode == 'analyze':
        logger.info("Starting RCA analysis...")
        
        if not args.issue:
            logger.error("Please provide an issue description with --issue")
            return
        
        # If no logs path specified, try to use the OpenStack_2k.log
        if not args.logs:
            log_file = setup_openstack_log()
            if log_file:
                args.logs = 'logs'
            else:
                logger.error("No log files found. Please run: python main.py --mode setup")
                return
        
        results = run_rca_analysis(args.issue, args.logs, args.fast_mode)
        if results:
            logger.info("RCA analysis completed successfully!")
        else:
            logger.error("RCA analysis failed!")
    
    elif args.mode == 'streamlit':
        logger.info("Starting Streamlit application...")
        
        # Ensure OpenStack log is available
        setup_openstack_log()
        
        try:
            import subprocess
            import sys
            
            # Run streamlit app
            cmd = [sys.executable, '-m', 'streamlit', 'run', 
                   'streamlit_app/chatbot.py', '--server.port', '8501']
            subprocess.run(cmd)
            
        except Exception as e:
            logger.error(f"Failed to start Streamlit app: {e}")
            logger.info("Please run manually: streamlit run streamlit_app/chatbot.py")
    
    else:
        logger.error(f"Unknown mode: {args.mode}")
        parser.print_help()

if __name__ == "__main__":
    main()