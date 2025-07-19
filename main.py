"""
OpenStack RCA System - Main Entry Point
Interactive Root Cause Analysis system for OpenStack logs using LSTM and Claude API
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Disable ChromaDB telemetry to prevent errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import Config
from data.log_ingestion import LogIngestionManager
from data.preprocessing import LogPreprocessor
from lstm.lstm_classifier import LSTMLogClassifier
from lstm.rca_analyzer import RCAAnalyzer
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

def train_model_pipeline(clean_vector_db: bool = False):
    """Train the LSTM model pipeline with optional ChromaDB cleanup. No log ingestion here."""
    logger.info("Starting model training pipeline...")
    
    # Step 1: Initialize components
    preprocessor = LogPreprocessor()
    feature_engineer = FeatureEngineer()
    
    # NEW: Check ChromaDB status before training
    try:
        from services.vector_db_service import VectorDBService
        vector_db = VectorDBService()
        stats = vector_db.get_collection_stats()
        logger.info(f"ChromaDB Status before training: {stats['total_documents']} documents")
        
        # Only clean if explicitly requested AND there are documents
        if clean_vector_db and stats['total_documents'] > 0:
            logger.info("🔄 Cleaning ChromaDB before training (explicitly requested)...")
            vector_db.clear_collection()
            logger.info("✅ ChromaDB cleaned successfully")
        elif clean_vector_db and stats['total_documents'] == 0:
            logger.info("📊 ChromaDB is already empty, no cleaning needed")
        else:
            logger.info("📊 ChromaDB will retain existing data (use --clean-vector-db to reset)")
    except Exception as e:
        logger.warning(f"Failed to check ChromaDB status: {e}")
    
    # Step 2: Load already-ingested logs from vector DB for training
    logger.info("Loading logs from vector DB for training...")
    try:
        from services.vector_db_service import VectorDBService
        vector_db = VectorDBService()
        # Get all documents from vector DB
        results = vector_db.collection.get()
        if not results['documents']:
            logger.error("No log data found in vector DB for training")
            return None
        import pandas as pd
        # Reconstruct DataFrame from metadatas and documents
        df = pd.DataFrame(results['metadatas'])
        df['message'] = results['documents']
    except Exception as e:
        logger.error(f"Failed to load logs from vector DB: {e}")
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
    
    # NEW: Print ChromaDB status after training
    try:
        from services.vector_db_service import VectorDBService
        vector_db = VectorDBService()
        stats = vector_db.get_collection_stats()
        logger.info(f"ChromaDB Status after training: {stats['total_documents']} documents")
        logger.info(f"Database type: {stats['database_type']}")
        
        if stats['total_documents'] > 0:
            logger.info("✅ ChromaDB has data and is ready for RCA analysis")
        else:
            logger.warning("⚠️ ChromaDB is empty - RCA analysis will use fast mode only")
            
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
    model_path = os.path.join('data/model', 'lstm_log_classifier.keras')
    lstm_model = None
    
    if os.path.exists(model_path):
        logger.info("Loading existing LSTM model...")
        lstm_model = LSTMLogClassifier(Config.LSTM_CONFIG)
        lstm_model.load_model(model_path)
    else:
        logger.warning("No trained model found. Training new model...")
        lstm_model = train_model_pipeline(clean_vector_db=False) # Pass False as it's not for ingestion
    
    # Initialize Hybrid RCA analyzer
    rca_analyzer = RCAAnalyzer(Config.ANTHROPIC_API_KEY, lstm_model)
    
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

def test_model_performance(custom_query: str = None, log_files_path: str = None, iterations: int = 3):
    """Test model performance with custom queries"""
    logger.info("Starting model performance testing...")
    
    # Check for API key
    if not Config.ANTHROPIC_API_KEY:
        logger.error("Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable.")
        return
    
    # Initialize log cache
    from utils.log_cache import LogCache
    log_cache = LogCache()
    
    # Load or train LSTM model
    model_path = os.path.join('data/model', 'lstm_log_classifier.keras')
    lstm_model = None
    
    if os.path.exists(model_path):
        logger.info("Loading existing LSTM model...")
        lstm_model = LSTMLogClassifier(Config.LSTM_CONFIG)
        lstm_model.load_model(model_path)
    else:
        logger.warning("No trained model found. Training new model...")
        lstm_model = train_model_pipeline(clean_vector_db=False)
    
    # Initialize RCA analyzer
    rca_analyzer = RCAAnalyzer(Config.ANTHROPIC_API_KEY, lstm_model)
    
    # Get cached logs or load from files
    if log_files_path:
        logs_df = log_cache.get_cached_logs(log_files_path)
    else:
        logs_df = log_cache.get_cached_logs(Config.DATA_DIR)
    
    if logs_df.empty:
        logger.error("No log data found for testing.")
        return
    
    # Define test queries
    if custom_query:
        test_queries = [custom_query]
        # For custom query, show actual results
        show_results = True
    else:
        test_queries = [
            "Instance launch failures",
            "Network connectivity issues", 
            "Authentication problems",
            "Resource allocation failures",
            "Service startup errors"
        ]
        show_results = False
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE TESTING")
    print("="*60)
    print(f"Total logs available: {len(logs_df)}")
    print(f"Test queries: {len(test_queries)}")
    print(f"Iterations per query: {iterations}")
    print("="*60)
    
    # Performance metrics storage
    all_metrics = []
    
    for query in test_queries:
        print(f"\n🔍 Testing Query: '{query}'")
        print("-" * 50)
        
        query_metrics = []
        
        for i in range(iterations):
            print(f"  Iteration {i+1}/{iterations}...")
            
            try:
                # Test hybrid mode
                start_time = datetime.now()
                results = rca_analyzer.analyze_issue(query, logs_df, fast_mode=False)
                hybrid_time = (datetime.now() - start_time).total_seconds()
                
                # Test fast mode
                start_time = datetime.now()
                fast_results = rca_analyzer.analyze_issue(query, logs_df, fast_mode=True)
                fast_time = (datetime.now() - start_time).total_seconds()
                
                # Store metrics
                iteration_metrics = {
                    'query': query,
                    'iteration': i+1,
                    'hybrid_time': hybrid_time,
                    'fast_time': fast_time,
                    'hybrid_logs': results.get('relevant_logs_count', 0),
                    'fast_logs': fast_results.get('relevant_logs_count', 0),
                    'hybrid_category': results.get('issue_category', 'unknown'),
                    'fast_category': fast_results.get('issue_category', 'unknown')
                }
                
                query_metrics.append(iteration_metrics)
                
                print(f"    Hybrid: {hybrid_time:.2f}s ({results.get('relevant_logs_count', 0)} logs)")
                print(f"    Fast:   {fast_time:.2f}s ({fast_results.get('relevant_logs_count', 0)} logs)")
                print(f"    Speedup: {hybrid_time/fast_time:.1f}x")
                
                # Show actual results for custom query (only on first iteration)
                if show_results and i == 0:
                    print(f"\n📋 HYBRID MODE RESULTS:")
                    print("="*50)
                    print(f"Issue: {results.get('issue_description', query)}")
                    print(f"Category: {results.get('issue_category', 'unknown')}")
                    print(f"Relevant Logs: {results.get('relevant_logs_count', 0)}")
                    print(f"Analysis Mode: {results.get('analysis_mode', 'hybrid')}")
                    
                    # Display performance metrics
                    if 'performance_metrics' in results:
                        metrics = results['performance_metrics']
                        print(f"\nPerformance Metrics:")
                        print(f"- Processing Time: {metrics.get('processing_time', 0):.2f} seconds")
                        print(f"- Total Logs: {metrics.get('total_logs', 0)}")
                        print(f"- Filtered Logs: {metrics.get('filtered_logs', 0)}")
                        print(f"- LSTM Available: {metrics.get('lstm_available', False)}")
                        print(f"- Vector DB Available: {metrics.get('vector_db_available', False)}")
                    
                    # Show prompt only (not the full RCA response) - ONLY ONCE for hybrid mode
                    # REMOVED: Duplicate prompt printing since LSTM analyzer already logs it
                    # if 'prompt' in results and results['prompt']:
                    #     print(f"\n📄 PROMPT SENT TO LLM:")
                    #     print("="*50)
                    #     print(results['prompt'][:500] + "..." if len(results['prompt']) > 500 else results['prompt'])
                    
                    # Display top filtered logs with scores
                    if 'filtered_logs' in results and not results['filtered_logs'].empty:
                        print(f"\n🔍 TOP RELEVANT LOGS (showing first 5):")
                        top_logs = results['filtered_logs'].head(5)
                        for _, log in top_logs.iterrows():
                            score = log.get('combined_score', log.get('lstm_importance', 0))
                            print(f"- [{log.get('level', 'INFO')}] {log.get('service_type', 'unknown')}: {log.get('message', '')[:100]}... (Score: {score:.3f})")
                    
                    print(f"\n📋 FAST MODE RESULTS:")
                    print("="*50)
                    print(f"Issue: {fast_results.get('issue_description', query)}")
                    print(f"Category: {fast_results.get('issue_category', 'unknown')}")
                    print(f"Relevant Logs: {fast_results.get('relevant_logs_count', 0)}")
                    print(f"Analysis Mode: {fast_results.get('analysis_mode', 'fast')}")
                    
                    # REMOVED: No longer showing fast mode prompt to avoid duplication
                
            except Exception as e:
                logger.error(f"Test iteration failed: {e}")
                continue
        
        # Calculate averages for this query
        if query_metrics:
            avg_hybrid_time = sum(m['hybrid_time'] for m in query_metrics) / len(query_metrics)
            avg_fast_time = sum(m['fast_time'] for m in query_metrics) / len(query_metrics)
            avg_speedup = avg_hybrid_time / avg_fast_time
            
            print(f"\n  📊 Query Summary:")
            print(f"    Average Hybrid Time: {avg_hybrid_time:.2f}s")
            print(f"    Average Fast Time:   {avg_fast_time:.2f}s")
            print(f"    Average Speedup:     {avg_speedup:.1f}x")
            
            all_metrics.extend(query_metrics)
    
    # Overall summary (only show if not custom query)
    if all_metrics and not show_results:
        print("\n" + "="*60)
        print("OVERALL PERFORMANCE SUMMARY")
        print("="*60)
        
        total_hybrid_time = sum(m['hybrid_time'] for m in all_metrics)
        total_fast_time = sum(m['fast_time'] for m in all_metrics)
        overall_speedup = total_hybrid_time / total_fast_time
        
        print(f"Total Tests: {len(all_metrics)}")
        print(f"Total Hybrid Time: {total_hybrid_time:.2f}s")
        print(f"Total Fast Time:   {total_fast_time:.2f}s")
        print(f"Overall Speedup:   {overall_speedup:.1f}x")
        
        # Show best and worst cases
        best_hybrid = min(all_metrics, key=lambda x: x['hybrid_time'])
        worst_hybrid = max(all_metrics, key=lambda x: x['hybrid_time'])
        
        print(f"\nBest Hybrid Performance:  {best_hybrid['hybrid_time']:.2f}s ({best_hybrid['query']})")
        print(f"Worst Hybrid Performance: {worst_hybrid['hybrid_time']:.2f}s ({worst_hybrid['query']})")
        
        # Category distribution
        hybrid_categories = [m['hybrid_category'] for m in all_metrics]
        fast_categories = [m['fast_category'] for m in all_metrics]
        
        print(f"\nCategory Consistency:")
        print(f"Hybrid categories: {set(hybrid_categories)}")
        print(f"Fast categories:   {set(fast_categories)}")
        
        # Log count comparison
        avg_hybrid_logs = sum(m['hybrid_logs'] for m in all_metrics) / len(all_metrics)
        avg_fast_logs = sum(m['fast_logs'] for m in all_metrics) / len(all_metrics)
        
        print(f"\nLog Filtering:")
        print(f"Average Hybrid Logs: {avg_hybrid_logs:.1f}")
        print(f"Average Fast Logs:   {avg_fast_logs:.1f}")
    
    print("\n✅ Performance testing completed!")
    return all_metrics

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='OpenStack RCA System')
    parser.add_argument('--mode', choices=['train', 'analyze', 'streamlit', 'setup', 'vector-db', 'categories', 'test-ml-model'], 
                       default='streamlit', help='Operation mode')
    parser.add_argument('--logs', type=str, help='Path to log files directory (for ingest in vector-db mode)')
    parser.add_argument('--issue', type=str, 
                       help='Issue description for RCA analysis')
    parser.add_argument('--api-key', type=str, 
                       help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--clean-vector-db', action='store_true',
                       help='Clean ChromaDB before training (removes all existing data)')
    parser.add_argument('--action', choices=['status', 'clean', 'reset', 'ingest'],
                       help='Action for vector-db mode (status, clean, reset, ingest)')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Use fast mode for analysis (skips vector DB for speed)')
    parser.add_argument('--custom-query', type=str,
                       help='Custom query for model performance testing')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of iterations for model performance testing')
    
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
            logger.info("You can now run: python main.py --mode vector-db --action ingest --logs logs")
        else:
            logger.error("Setup failed. Please place OpenStack_2k.log in the project directory.")
        
    elif args.mode == 'vector-db':
        logger.info("Vector Database Management Mode")
        
        try:
            from services.vector_db_service import VectorDBService
            vector_db = VectorDBService()
            
            if args.action == 'status':
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
                
            elif args.action == 'clean':
                logger.info("Cleaning ChromaDB collection...")
                vector_db.clear_collection()
                logger.info("✅ ChromaDB collection cleaned successfully")
                
            elif args.action == 'reset':
                logger.info("Resetting ChromaDB database...")
                from services.vector_db_service import VectorDBService
                reset_success = VectorDBService.static_reset_chroma_db(Config.VECTOR_DB_CONFIG)
                if reset_success:
                    logger.info("✅ ChromaDB database reset successfully")
                else:
                    logger.error("❌ ChromaDB database reset failed - using in-memory fallback")
                    logger.info("💡 Try running the reset command again, or check ChromaDB installation")
                
            elif args.action == 'ingest':
                logger.info("Ingesting logs into ChromaDB collection...")
                if not args.logs:
                    logger.error("Please specify --logs <log_dir> for ingestion.")
                    return
                from data.log_ingestion import LogIngestionManager
                import os
                ingestion_manager = LogIngestionManager()
                import pandas as pd
                # List files in the directory for debug
                if os.path.isdir(args.logs):
                    files = [f for f in os.listdir(args.logs) if os.path.isfile(os.path.join(args.logs, f))]
                    logger.info(f"Found {len(files)} files in directory {args.logs}: {files}")
                    logs_df = ingestion_manager.ingest_from_directory(args.logs)
                else:
                    logger.info(f"Treating {args.logs} as a single file")
                    logs_df = ingestion_manager.ingest_multiple_files([args.logs])
                logger.info(f"Loaded DataFrame with {len(logs_df)} rows")
                if logs_df.empty:
                    logger.error(f"No valid logs found in {args.logs}")
                    return
                added = vector_db.add_logs(logs_df)
                logger.info(f"✅ Ingested {added} logs into ChromaDB collection")
                stats = vector_db.get_collection_stats()
                logger.info(f"ChromaDB now contains {stats['total_documents']} documents")
                
            else:
                logger.error("Please specify --action: status, clean, reset, or ingest")
                logger.info("Examples:")
                logger.info("  python main.py --mode vector-db --action status")
                logger.info("  python main.py --mode vector-db --action clean")
                logger.info("  python main.py --mode vector-db --action reset")
                logger.info("  python main.py --mode vector-db --action ingest --logs logs/")
                
        except Exception as e:
            logger.error(f"Vector DB operation failed: {e}")
        
    elif args.mode == 'train':
        logger.info("Starting model training...")
        
        # NEW: Show ChromaDB cleanup status
        if args.clean_vector_db:
            logger.info("🔄 ChromaDB will be cleaned before training")
        else:
            logger.info("📊 ChromaDB will retain existing data (use --clean-vector-db to reset)")
        
        model = train_model_pipeline(clean_vector_db=args.clean_vector_db)
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
            
            # Run streamlit app (no PRINT_LLM_PROMPT env var)
            cmd = [sys.executable, '-m', 'streamlit', 'run', 
                   'streamlit_app/chatbot.py', '--server.port', '8501']
            subprocess.run(cmd)
            
        except Exception as e:
            logger.error(f"Failed to start Streamlit app: {e}")
            logger.info("Please run manually: streamlit run streamlit_app/chatbot.py")
    
    elif args.mode == 'categories':
        logger.info("Displaying available issue categories and keywords...")
        try:
            # Initialize RCA analyzer to get categories
            rca_analyzer = RCAAnalyzer('dummy-key')  # Dummy key for categories only
            categories = rca_analyzer.get_available_categories()
            
            print("\n" + "="*60)
            print("AVAILABLE ISSUE CATEGORIES")
            print("="*60)
            for category_name, keywords in categories.items():
                print(f"📋 {category_name.upper().replace('_', ' ')}:")
                print(f"   Keywords: {', '.join(keywords)}")
                print()
            
            print("="*60)
            print("USAGE: When describing issues, use keywords from these categories")
            print("Example: 'Instance launch failures due to resource constraints'")
            print("         (matches 'instance_issues' and 'resource_shortage' categories)")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Failed to retrieve issue categories: {e}")
    
    elif args.mode == 'test-ml-model':
        logger.info("Starting model performance testing...")
        test_model_performance(custom_query=args.custom_query, iterations=args.iterations)
    
    else:
        logger.error(f"Unknown mode: {args.mode}")
        parser.print_help()

if __name__ == "__main__":
    main()