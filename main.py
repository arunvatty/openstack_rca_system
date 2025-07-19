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

def train_model_pipeline(clean_vector_db: bool = False, mlflow_manager=None):
    """Train the LSTM model pipeline with optional ChromaDB cleanup and MLflow tracking."""
    logger.info("Starting model training pipeline...")
    
    # Start MLflow run if enabled
    run_id = None
    if mlflow_manager and mlflow_manager.is_enabled:
        experiment_name = mlflow_manager.experiment_name.replace('_', '-')
        run_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_id = mlflow_manager.start_run(
            run_name=run_name,
            tags={
                'model_type': 'lstm',
                'training_mode': 'hybrid',
                'clean_vector_db': str(clean_vector_db)
            }
        )
        logger.info(f"🚀 Started MLflow run: {run_id}")
    
    try:
        # Step 1: Initialize components
        preprocessor = LogPreprocessor()
        feature_engineer = FeatureEngineer()
        
        # NEW: Check ChromaDB status before training
        try:
            from services.vector_db_service import VectorDBService
            vector_db = VectorDBService()
            stats = vector_db.get_collection_stats()
            logger.info(f"ChromaDB Status before training: {stats['total_documents']} documents")
            
            # Log ChromaDB stats to MLflow
            if mlflow_manager and mlflow_manager.is_enabled:
                mlflow_manager.log_metrics({
                    'chromadb_documents_before': stats['total_documents'],
                    'chromadb_chunked_docs': stats.get('chunked_documents', 0),
                    'chromadb_non_chunked_docs': stats.get('non_chunked_documents', 0)
                })
            
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
                if mlflow_manager and mlflow_manager.is_enabled:
                    mlflow_manager.end_run(status="FAILED")
                return None
            import pandas as pd
            # Reconstruct DataFrame from metadatas and documents
            df = pd.DataFrame(results['metadatas'])
            df['message'] = results['documents']
            
            # Log data statistics to MLflow
            if mlflow_manager and mlflow_manager.is_enabled:
                mlflow_manager.log_metrics({
                    'total_log_entries': len(df),
                    'unique_services': df['service_type'].nunique() if 'service_type' in df.columns else 0,
                    'error_entries': len(df[df['level'].str.upper() == 'ERROR']) if 'level' in df.columns else 0
                })
                
        except Exception as e:
            logger.error(f"Failed to load logs from vector DB: {e}")
            if mlflow_manager and mlflow_manager.is_enabled:
                mlflow_manager.end_run(status="FAILED")
            return None
        
        # Apply feature engineering
        logger.info("Applying feature engineering...")
        df = feature_engineer.engineer_all_features(df)
        
        # Log feature engineering stats to MLflow
        if mlflow_manager and mlflow_manager.is_enabled:
            mlflow_manager.log_metrics({
                'engineered_features': len(df.columns),
                'feature_engineered_entries': len(df)
            })
        
        # Step 3: Prepare data for LSTM
        logger.info("Preparing data for LSTM training...")
        X, y = preprocessor.prepare_lstm_data(df, Config.LSTM_CONFIG['max_sequence_length'])
        
        # Log LSTM configuration to MLflow
        if mlflow_manager and mlflow_manager.is_enabled:
            lstm_params = {
                'max_sequence_length': Config.LSTM_CONFIG['max_sequence_length'],
                'embedding_dim': Config.LSTM_CONFIG['embedding_dim'],
                'lstm_units': Config.LSTM_CONFIG['lstm_units'],
                'dropout_rate': Config.LSTM_CONFIG['dropout_rate'],
                'batch_size': Config.LSTM_CONFIG['batch_size'],
                'epochs': Config.LSTM_CONFIG['epochs'],
                'validation_split': Config.LSTM_CONFIG['validation_split'],
                'training_samples': len(X),
                'sequence_length': X.shape[1] if len(X.shape) > 1 else 0,
                'num_classes': len(set(y)) if len(y) > 0 else 0
            }
            mlflow_manager.log_params(lstm_params)
        
        # Step 4: Train LSTM model
        logger.info("Training LSTM model...")
        lstm_classifier = LSTMLogClassifier(Config.LSTM_CONFIG)
        
        # Record training start time
        training_start_time = datetime.now()
        
        results = lstm_classifier.train(X, y)
        
        # Calculate training time
        training_time = (datetime.now() - training_start_time).total_seconds()
        
        # Step 5: Save model
        model_path = os.path.join(Config.MODELS_DIR, 'lstm_log_classifier.keras')
        lstm_classifier.save_model(model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Log training results and model to MLflow
        if mlflow_manager and mlflow_manager.is_enabled:
            # Log training metrics
            training_metrics = {
                'val_accuracy': results['val_accuracy'],
                'val_precision': results['val_precision'],
                'val_recall': results['val_recall'],
                'training_time_seconds': training_time,
                'final_loss': results.get('loss', 0.0),
                'final_val_loss': results.get('val_loss', 0.0)
            }
            mlflow_manager.log_metrics(training_metrics)
            
            # Log the trained model
            try:
                model_artifacts = {
                    'model_path': model_path,
                    'config_file': 'config/config.py'
                }
                model_version = mlflow_manager.log_model_with_versioning(
                    lstm_classifier.model,
                    model_name="lstm_model",
                    model_type="tensorflow",
                    model_stage="Staging",
                    artifacts=model_artifacts
                )
                
                if model_version:
                    logger.info(f"📊 Model v{model_version} logged to MLflow and S3 successfully")
                    
                    # Log model registry information
                    registry_info = mlflow_manager.get_model_registry_info()
                    if registry_info:
                        logger.info(f"🗃️ Model Registry Status:")
                        for model_name, model_info in registry_info['models'].items():
                            logger.info(f"   - {model_name}: {model_info['total_versions']} versions, latest: v{model_info['latest_version']}")
                else:
                    logger.warning("⚠️ Model logging to MLflow failed")
                    
            except Exception as e:
                logger.warning(f"Failed to log model to MLflow: {e}")
        else:
            logger.info("📝 No MLflow manager - model saved locally only")
        
        # Print training results
        logger.info("Training Results:")
        logger.info(f"Validation Accuracy: {results['val_accuracy']:.4f}")
        logger.info(f"Validation Precision: {results['val_precision']:.4f}")
        logger.info(f"Validation Recall: {results['val_recall']:.4f}")
        logger.info(f"Training Time: {training_time:.2f} seconds")
        
        # NEW: Print ChromaDB status after training
        try:
            from services.vector_db_service import VectorDBService
            vector_db = VectorDBService()
            stats = vector_db.get_collection_stats()
            logger.info(f"ChromaDB Status after training: {stats['total_documents']} documents")
            
            # Log final ChromaDB stats to MLflow
            if mlflow_manager and mlflow_manager.is_enabled:
                mlflow_manager.log_metrics({
                    'chromadb_documents_after': stats['total_documents']
                })
                
        except Exception as e:
            logger.warning(f"Failed to check ChromaDB status after training: {e}")
        
        # End MLflow run successfully
        if mlflow_manager and mlflow_manager.is_enabled:
            mlflow_manager.end_run(status="FINISHED")
            logger.info("✅ MLflow run completed successfully")
            
            # Print MLflow run info
            run_info = mlflow_manager.get_run_info()
            if run_info:
                logger.info(f"📊 MLflow Run ID: {run_info['run_id']}")
                logger.info(f"🔗 Artifact URI: {run_info['artifact_uri']}")
        
        return lstm_classifier
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        
        # End MLflow run with failure status
        if mlflow_manager and mlflow_manager.is_enabled:
            mlflow_manager.end_run(status="FAILED")
            logger.info("❌ MLflow run ended with failure status")
        
        return None

def run_rca_analysis(issue_description: str, log_files_path: str = None, fast_mode: bool = False, enable_mlflow: bool = False, mlflow_uri: str = None, experiment_name: str = 'openstack_rca_system_staging'):
    """Run RCA analysis on a specific issue using Hybrid RCA Analyzer with optional MLflow model loading"""
    logger.info("Starting hybrid RCA analysis...")
    
    # Check for API key
    if not Config.ANTHROPIC_API_KEY:
        logger.error("Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable.")
        return
    
    # Initialize MLflow if enabled for model loading
    mlflow_manager = None
    if enable_mlflow:
        try:
            from mlflow_integration.mlflow_manager import MLflowManager
            mlflow_manager = MLflowManager(
                tracking_uri=mlflow_uri,
                experiment_name=experiment_name,
                enable_mlflow=True
            )
            
            if mlflow_manager.is_enabled:
                logger.info("✅ MLflow enabled for model loading")
            else:
                logger.warning("⚠️ MLflow initialization failed - falling back to local model")
                mlflow_manager = None
                
        except ImportError:
            logger.warning("⚠️ MLflow not installed - using local model")
            mlflow_manager = None
        except Exception as e:
            logger.warning(f"⚠️ MLflow setup failed: {e} - using local model")
            mlflow_manager = None
    
    # Initialize log cache
    from utils.log_cache import LogCache
    log_cache = LogCache()
    
    # Load LSTM model with MLflow support
    lstm_model = None
    
    # Try loading from MLflow first if enabled
    if mlflow_manager and mlflow_manager.is_enabled:
        try:
            logger.info("🔍 Attempting to load model from MLflow/S3...")
            model_result = mlflow_manager.load_model_with_versioning(
                model_name="lstm_model",
                version="latest",
                stage=None  # Load latest version regardless of stage
            )
            
            if model_result is not None:
                logger.info("✅ Successfully loaded LSTM model from MLflow/S3")
                
                # Wrap the MLflow model in our classifier
                lstm_model = LSTMLogClassifier(Config.LSTM_CONFIG)
                lstm_model.model = model_result
                
                # Store model metadata for inference tracking
                model_metadata_for_inference = {
                    'model_source': 'mlflow_s3'
                }
            else:
                logger.warning("⚠️ No model found in MLflow registry")
                model_metadata_for_inference = {'model_source': 'none'}
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load model from MLflow: {e}")
            logger.error(f"❌ FALLBACK: MLflow model loading failed - {str(e)}")
            logger.error("❌ Will attempt to load from local file or train new model")
            model_metadata_for_inference = {'model_source': 'failed', 'error': str(e)}
            
    # Final fallback to local model if MLflow failed
        if lstm_model is None:
            model_path = os.path.join('data/model', 'lstm_log_classifier.keras')
            
            if os.path.exists(model_path):
                logger.info("📁 Loading existing LSTM model from local file...")
                logger.warning("⚠️ FALLBACK: Using local model file instead of MLflow")
                lstm_model = LSTMLogClassifier(Config.LSTM_CONFIG)
                lstm_model.load_model(model_path)
                model_metadata_for_inference = {'model_source': 'local_file', 'model_path': model_path}
            else:
                logger.warning("⚠️ No trained model found locally. Training new model...")
                logger.error("❌ FALLBACK: Both MLflow and local model failed - training new model")
                lstm_model = train_model_pipeline(clean_vector_db=False) # Pass False as it's not for ingestion
                model_metadata_for_inference = {'model_source': 'freshly_trained'}
    
    # Start MLflow run for inference tracking if enabled
    inference_run_id = None
    if mlflow_manager and mlflow_manager.is_enabled:
        try:
            run_name = f"rca_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            inference_run_id = mlflow_manager.start_run(
                run_name=run_name,
                tags={
                    'task': 'inference',
                    'issue_type': 'rca_analysis',
                    'fast_mode': str(fast_mode)
                }
            )
            
            # Log inference parameters including model metadata
            inference_params = {
                'issue_description': issue_description,
                'fast_mode': fast_mode,
                'log_files_path': log_files_path or 'default'
            }
            
            # Add model metadata to parameters
            if 'model_metadata_for_inference' in locals():
                inference_params.update(model_metadata_for_inference)
            
            mlflow_manager.log_params(inference_params)
            
            logger.info(f"🚀 Started MLflow inference run: {inference_run_id}")
            
        except Exception as e:
            logger.warning(f"Failed to start MLflow inference run: {e}")
    
    # Initialize Hybrid RCA analyzer
    rca_analyzer = RCAAnalyzer(Config.ANTHROPIC_API_KEY, lstm_model)
    
    # Get cached logs or load from files
    if log_files_path:
        logs_df = log_cache.get_cached_logs(log_files_path)
    else:
        logs_df = log_cache.get_cached_logs(Config.DATA_DIR)
    
    if logs_df.empty:
        logger.error("No log data found for analysis.")
        if mlflow_manager and mlflow_manager.is_enabled:
            mlflow_manager.end_run(status="FAILED")
        return
    
    # Record inference start time
    inference_start_time = datetime.now()
    
    # Perform RCA analysis
    logger.info(f"Analyzing issue: {issue_description}")
    results = rca_analyzer.analyze_issue(issue_description, logs_df, fast_mode=fast_mode)
    
    # Calculate inference time
    inference_time = (datetime.now() - inference_start_time).total_seconds()
    
    # Log inference metrics to MLflow
    if mlflow_manager and mlflow_manager.is_enabled:
        try:
            inference_metrics = {
                'inference_time_seconds': inference_time,
                'relevant_logs_count': results['relevant_logs_count'],
                'total_logs_analyzed': len(logs_df),
                'lstm_available': 1 if lstm_model else 0,
                'vector_db_available': 1 if results.get('vector_db_used', False) else 0
            }
            mlflow_manager.log_metrics(inference_metrics)
            
            # Log analysis results as artifacts
            analysis_summary = {
                'issue_description': issue_description,
                'issue_category': results['issue_category'],
                'analysis_mode': results['analysis_mode'],
                'relevant_logs_count': results['relevant_logs_count'],
                'inference_time': inference_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log analysis summary as MLflow parameters (no temp file)
            try:
                mlflow_manager.log_params({
                    'analysis_issue': analysis_summary['issue_description'][:100],
                    'analysis_category': analysis_summary['issue_category'],
                    'analysis_mode': analysis_summary['analysis_mode'],
                    'relevant_logs_count': analysis_summary['relevant_logs_count']
                })
            except Exception as e:
                logger.warning(f"Failed to log analysis summary: {e}")
            
            # Clean up temporary file
            if os.path.exists(summary_path):
                os.remove(summary_path)
                
            mlflow_manager.end_run(status="FINISHED")
            logger.info("✅ MLflow inference run completed successfully")
            
        except Exception as e:
            logger.warning(f"Failed to log inference metrics to MLflow: {e}")
            if mlflow_manager and mlflow_manager.is_enabled:
                mlflow_manager.end_run(status="FAILED")
    
    # Display results
    print("\n" + "="*50)
    print("HYBRID RCA ANALYSIS RESULTS")
    print("="*50)
    print(f"Issue: {issue_description}")
    print(f"Category: {results['issue_category']}")
    print(f"Relevant Logs: {results['relevant_logs_count']}")
    print(f"Analysis Mode: {results['analysis_mode']}")
    print(f"Inference Time: {inference_time:.2f} seconds")
    if mlflow_manager and mlflow_manager.is_enabled:
        print(f"MLflow Run ID: {inference_run_id}")
    print("="*50)
    
    print("\nRoot Cause Analysis:")
    print(results.get('root_cause_analysis', 'Analysis not available'))
    
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
    
    # MLflow arguments (MLflow is ENABLED BY DEFAULT)
    parser.add_argument('--enable-mlflow', action='store_true',
                       help='Explicitly enable MLflow (redundant - enabled by default)')
    parser.add_argument('--disable-mlflow', action='store_true',
                       help='Disable MLflow experiment tracking and model logging')
    parser.add_argument('--mlflow-uri', type=str,
                       help='MLflow tracking URI (overrides config/env settings)')
    parser.add_argument('--mlflow-experiment', type=str, default='openstack_rca_system_staging',
                       help='MLflow experiment name')
    
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
                from config.config import Config
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
        
        # Determine MLflow settings - AUTO-ENABLE by default from config
        enable_mlflow = True  # Default to enabled
        
        if args.disable_mlflow:
            enable_mlflow = False
            logger.info("🚫 MLflow explicitly disabled via --disable-mlflow")
        elif args.enable_mlflow:
            enable_mlflow = True
            logger.info("✅ MLflow explicitly enabled via --enable-mlflow")
        else:
            # Use config default (auto_log = True)
            try:
                from config.config import Config
                enable_mlflow = Config.MLFLOW_CONFIG.get('auto_log', True)
                if enable_mlflow:
                    logger.info("✅ MLflow auto-enabled from configuration")
                else:
                    logger.info("📝 MLflow disabled in configuration")
            except:
                enable_mlflow = True  # Default to enabled if config fails
                logger.info("✅ MLflow enabled by default")
        
        # Set MLflow tracking URI if provided
        mlflow_uri = args.mlflow_uri
        if not mlflow_uri:
            try:
                from config.config import Config
                mlflow_uri = Config.MLFLOW_TRACKING_URI
            except:
                mlflow_uri = None
        
        # Auto-generate experiment name with versioning
        if args.mlflow_experiment == 'openstack_rca_system_staging':  # Default value
            # Use auto-versioned experiment name from config
            experiment_name = Config.MLFLOW_CONFIG.get('experiment_name', 'openstack_rca_system_staging')
            logger.info(f"🎯 Using auto-configured experiment: {experiment_name}")
        else:
            # Use user-provided experiment name
            experiment_name = args.mlflow_experiment
            logger.info(f"🎯 Using custom experiment: {experiment_name}")
        
        # Initialize MLflow if enabled
        mlflow_manager = None
        if enable_mlflow:
            try:
                from mlflow_integration.mlflow_manager import MLflowManager
                mlflow_manager = MLflowManager(
                    tracking_uri=mlflow_uri,
                    experiment_name=experiment_name,
                    enable_mlflow=True
                )
                
                if mlflow_manager.is_enabled:
                    logger.info("✅ MLflow tracking enabled")
                    logger.info(f"📊 Experiment: {experiment_name}")
                    logger.info(f"🔗 Tracking URI: {mlflow_manager.tracking_uri}")
                    logger.info("🔄 Auto-versioning: ENABLED (models will be versioned automatically)")
                else:
                    logger.warning("⚠️ MLflow initialization failed - will try S3 direct download")
                    mlflow_manager = None
                    
            except ImportError:
                logger.warning("⚠️ MLflow not installed - using local model")
                logger.info("💡 Install MLflow with: pip install mlflow")
                mlflow_manager = None
            except Exception as e:
                logger.warning(f"⚠️ MLflow setup failed: {e} - will try S3 direct download")
                # Create minimal MLflow manager for S3 operations
                try:
                    mlflow_manager = MLflowManager(
                        tracking_uri=mlflow_uri,
                        experiment_name=experiment_name,
                        enable_mlflow=False  # Disable MLflow but keep S3 functionality
                    )
                    logger.info("🔧 Created minimal MLflow manager for S3 operations")
                except:
                    logger.error("❌ Could not create MLflow manager for S3 operations")
                    mlflow_manager = None
        else:
            logger.info("📝 MLflow tracking disabled")
        
        # NEW: Show ChromaDB cleanup status
        if args.clean_vector_db:
            logger.info("🔄 ChromaDB will be cleaned before training")
        else:
            logger.info("📊 ChromaDB will retain existing data (use --clean-vector-db to reset)")
        
        model = train_model_pipeline(
            clean_vector_db=args.clean_vector_db,
            mlflow_manager=mlflow_manager
        )
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
        
        # Determine MLflow settings for analysis - DEFAULT TO ENABLED
        enable_mlflow = True  # Default to enabled
        if args.disable_mlflow:
            enable_mlflow = False
            logger.info("🚫 MLflow explicitly disabled via --disable-mlflow")
        elif args.enable_mlflow:
            enable_mlflow = True
            logger.info("✅ MLflow explicitly enabled via --enable-mlflow")
        else:
            # Use config default (now defaults to True)
            try:
                from config.config import Config
                enable_mlflow = getattr(Config, 'MLFLOW_CONFIG', {}).get('auto_log', True)
                if enable_mlflow:
                    logger.info("✅ MLflow auto-enabled by default")
                else:
                    logger.info("📝 MLflow disabled in configuration")
            except:
                enable_mlflow = True  # Default to enabled if config fails
                logger.info("✅ MLflow enabled by default")
        
        # Set MLflow tracking URI if provided
        mlflow_uri = args.mlflow_uri
        if not mlflow_uri:
            try:
                from config.config import Config
                mlflow_uri = getattr(Config, 'MLFLOW_TRACKING_URI', None)
            except:
                mlflow_uri = None
        
        # Auto-generate experiment name with versioning
        if args.mlflow_experiment == 'openstack_rca_system_staging':  # Default value
            # Use auto-versioned experiment name from config
            try:
                from config.config import Config
                experiment_name = Config.MLFLOW_CONFIG.get('experiment_name', 'openstack_rca_system_staging')
                logger.info(f"🎯 Using auto-configured experiment: {experiment_name}")
            except:
                experiment_name = 'openstack_rca_system_staging'
                logger.info(f"🎯 Using default experiment: {experiment_name}")
        else:
            # Use user-provided experiment name
            experiment_name = args.mlflow_experiment
            logger.info(f"🎯 Using custom experiment: {experiment_name}")
        
        results = run_rca_analysis(args.issue, args.logs, args.fast_mode, enable_mlflow, mlflow_uri, experiment_name)
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