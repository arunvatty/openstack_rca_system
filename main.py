"""
OpenStack RCA System - Main Entry Point
Interactive Root Cause Analysis system for OpenStack logs using LSTM and Claude API
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path

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

def train_model_pipeline(log_files_path: str = None):
    """Complete pipeline for training LSTM model"""
    logger.info("Starting model training pipeline...")
    
    # Initialize components
    ingestion_manager = LogIngestionManager(Config.DATA_DIR)
    preprocessor = LogPreprocessor()
    feature_engineer = FeatureEngineer()
    
    # Step 1: Ingest log files
    if log_files_path:
        logger.info(f"Ingesting logs from: {log_files_path}")
        df = ingestion_manager.ingest_from_directory(log_files_path)
    else:
        logger.info("Ingesting logs from default directory...")
        df = ingestion_manager.ingest_multiple_files()
    
    if df.empty:
        logger.error("No log data found. Please check log files path.")
        return None
    
    logger.info(f"Ingested {len(df)} log entries")
    
    # Step 2: Feature engineering
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
    model_path = os.path.join(Config.MODELS_DIR, 'lstm_log_classifier.h5')
    lstm_classifier.save_model(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Print training results
    logger.info("Training Results:")
    logger.info(f"Validation Accuracy: {results['val_accuracy']:.4f}")
    logger.info(f"Validation Precision: {results['val_precision']:.4f}")
    logger.info(f"Validation Recall: {results['val_recall']:.4f}")
    
    return lstm_classifier

def run_rca_analysis(issue_description: str, log_files_path: str = None):
    """Run RCA analysis on a specific issue"""
    logger.info("Starting RCA analysis...")
    
    # Check for API key
    if not Config.ANTHROPIC_API_KEY:
        logger.error("Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable.")
        return
    
    # Initialize components
    ingestion_manager = LogIngestionManager(Config.DATA_DIR)
    feature_engineer = FeatureEngineer()
    
    # Load or train LSTM model
    model_path = os.path.join(Config.MODELS_DIR, 'lstm_log_classifier.h5')
    lstm_model = None
    
    if os.path.exists(model_path):
        logger.info("Loading existing LSTM model...")
        lstm_model = LSTMLogClassifier(Config.LSTM_CONFIG)
        lstm_model.load_model(model_path)
    else:
        logger.warning("No trained model found. Training new model...")
        lstm_model = train_model_pipeline(log_files_path)
    
    # Initialize RCA analyzer
    rca_analyzer = RCAAnalyzer(Config.ANTHROPIC_API_KEY, lstm_model)
    
    # Ingest logs
    if log_files_path:
        df = ingestion_manager.ingest_from_directory(log_files_path)
    else:
        df = ingestion_manager.ingest_multiple_files()
    
    if df.empty:
        logger.error("No log data found for analysis.")
        return
    
    # Apply feature engineering
    df = feature_engineer.engineer_all_features(df)
    
    # Perform RCA analysis
    logger.info(f"Analyzing issue: {issue_description}")
    results = rca_analyzer.analyze_issue(issue_description, df)
    
    # Display results
    print("\n" + "="*50)
    print("ROOT CAUSE ANALYSIS RESULTS")
    print("="*50)
    print(f"Issue: {issue_description}")
    print(f"Category: {results['issue_category']}")
    print(f"Relevant Logs: {results['relevant_logs_count']}")
    print("\n" + results['root_cause_analysis'])
    
    if results['recommendations']:
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
    
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
    parser.add_argument('--mode', choices=['train', 'analyze', 'streamlit', 'setup'], 
                       default='streamlit', help='Operation mode')
    parser.add_argument('--logs', type=str, help='Path to log files directory')
    parser.add_argument('--issue', type=str, 
                       help='Issue description for RCA analysis')
    parser.add_argument('--api-key', type=str, 
                       help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    
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
        
        model = train_model_pipeline(args.logs)
        if model:
            logger.info("Model training completed successfully!")
        else:
            logger.error("Model training failed!")
    
    elif args.mode == 'analyze':
        if not args.issue:
            logger.error("Please provide an issue description with --issue")
            return
        
        logger.info("Starting RCA analysis...")
        
        # If no logs path specified, try to use the OpenStack_2k.log
        if not args.logs:
            log_file = setup_openstack_log()
            if log_file:
                args.logs = 'logs'
            else:
                logger.error("No log files found. Please run: python main.py --mode setup")
                return
        
        results = run_rca_analysis(args.issue, args.logs)
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

if __name__ == "__main__":
    main()