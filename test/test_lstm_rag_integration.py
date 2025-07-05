#!/usr/bin/env python3
"""
Test script for LSTM + RAG integration with ChromaDB
Tests the complete workflow from log ingestion to RCA analysis
"""

import os
import sys
import logging
import pandas as pd

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.log_ingestion import LogIngestionManager
from models.rca_analyzer import RCAAnalyzer
from models.lstm_classifier import LSTMLogClassifier
from config.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vector_db_initialization():
    """Test vector database initialization"""
    logger.info("Testing vector database initialization...")
    
    try:
        from services.vector_db_service import VectorDBService
        
        # Test initialization
        vector_db = VectorDBService()
        
        # Test collection stats
        stats = vector_db.get_collection_stats()
        logger.info(f"Vector DB stats: {stats}")
        
        logger.info("✓ Vector database initialization successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Vector database initialization failed: {e}")
        return False

def test_log_ingestion_with_vector_db():
    """Test log ingestion with vector database integration"""
    logger.info("Testing log ingestion with vector database...")
    
    try:
        # Initialize ingestion manager
        ingestion_manager = LogIngestionManager(Config.DATA_DIR)
        
        # Check if vector DB is available
        if not ingestion_manager.vector_db:
            logger.warning("Vector DB not available in ingestion manager")
            return False
        
        # Ingest logs from sample directory
        logs_df = ingestion_manager.ingest_from_directory('logs')
        
        if logs_df.empty:
            logger.error("No logs ingested")
            return False
        
        logger.info(f"✓ Successfully ingested {len(logs_df)} logs with vector DB integration")
        
        # Get ingestion stats
        stats = ingestion_manager.get_ingestion_stats(logs_df)
        logger.info(f"Ingestion stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Log ingestion with vector DB failed: {e}")
        return False

def test_lstm_rag_filtering():
    """Test LSTM + RAG filtering workflow"""
    logger.info("Testing LSTM + RAG filtering workflow...")
    
    try:
        # Load LSTM model
        model_path = os.path.join(Config.MODELS_DIR, 'lstm_log_classifier.h5')
        if not os.path.exists(model_path):
            logger.warning("LSTM model not found, testing without LSTM")
            lstm_model = None
        else:
            lstm_model = LSTMLogClassifier(Config.LSTM_CONFIG)
            lstm_model.load_model(model_path)
            logger.info("✓ LSTM model loaded successfully")
        
        # Initialize RCA analyzer
        rca_analyzer = RCAAnalyzer(
            anthropic_api_key=Config.ANTHROPIC_API_KEY,
            lstm_model=lstm_model
        )
        
        # Check if vector DB is available
        if not rca_analyzer.vector_db:
            logger.warning("Vector DB not available in RCA analyzer")
            return False
        
        logger.info("✓ RCA analyzer with LSTM + RAG initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ LSTM + RAG filtering test failed: {e}")
        return False

def test_complete_workflow():
    """Test complete LSTM + RAG workflow"""
    logger.info("Testing complete LSTM + RAG workflow...")
    
    try:
        # Step 1: Ingest logs
        ingestion_manager = LogIngestionManager(Config.DATA_DIR)
        logs_df = ingestion_manager.ingest_from_directory('logs')
        
        if logs_df.empty:
            logger.error("No logs available for testing")
            return False
        
        # Step 2: Load LSTM model
        model_path = os.path.join(Config.MODELS_DIR, 'lstm_log_classifier.h5')
        lstm_model = None
        if os.path.exists(model_path):
            lstm_model = LSTMLogClassifier(Config.LSTM_CONFIG)
            lstm_model.load_model(model_path)
            logger.info("✓ LSTM model loaded")
        
        # Step 3: Initialize RCA analyzer
        rca_analyzer = RCAAnalyzer(
            anthropic_api_key=Config.ANTHROPIC_API_KEY,
            lstm_model=lstm_model
        )
        
        # Step 4: Test filtering
        issue_description = "Instance launch failures"
        relevant_logs = rca_analyzer._filter_relevant_logs(logs_df, issue_description)
        
        logger.info(f"✓ Filtering completed: {len(relevant_logs)} relevant logs found")
        
        # Check for similarity scores
        if 'vector_similarity' in relevant_logs.columns:
            logger.info("✓ Vector similarity scores available")
        if 'tfidf_similarity' in relevant_logs.columns:
            logger.info("✓ TF-IDF similarity scores available")
        if 'combined_similarity' in relevant_logs.columns:
            logger.info("✓ Combined similarity scores available")
        
        # Step 5: Test historical context
        historical_context = rca_analyzer._get_historical_context(issue_description)
        if historical_context:
            logger.info("✓ Historical context retrieved")
            logger.info(f"Historical context length: {len(historical_context)} characters")
        else:
            logger.info("No historical context available (expected for first run)")
        
        # Step 6: Test issue categorization
        category = rca_analyzer._categorize_issue(issue_description)
        logger.info(f"✓ Issue categorized as: {category}")
        
        # Step 7: Test pattern analysis
        patterns = rca_analyzer._analyze_patterns(relevant_logs, category)
        logger.info(f"✓ Pattern analysis completed: {len(patterns)} pattern types")
        
        # Step 8: Test timeline extraction
        timeline = rca_analyzer._extract_timeline(relevant_logs)
        logger.info(f"✓ Timeline extracted: {len(timeline)} events")
        
        logger.info("✓ Complete LSTM + RAG workflow test successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Complete workflow test failed: {e}")
        return False

def test_vector_db_operations():
    """Test vector database operations"""
    logger.info("Testing vector database operations...")
    
    try:
        from services.vector_db_service import VectorDBService
        
        vector_db = VectorDBService()
        
        # Test similarity search
        test_query = "Instance launch failures"
        similar_logs = vector_db.search_similar_logs(test_query, top_k=5)
        
        logger.info(f"✓ Similarity search completed: {len(similar_logs)} results")
        
        # Test context retrieval
        context = vector_db.get_context_for_issue(test_query, top_k=3)
        
        if context:
            logger.info("✓ Context retrieval successful")
            logger.info(f"Context length: {len(context)} characters")
        else:
            logger.info("No context available (expected for first run)")
        
        # Test collection stats
        stats = vector_db.get_collection_stats()
        logger.info(f"✓ Collection stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Vector DB operations test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("LSTM + RAG INTEGRATION TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Vector DB Initialization", test_vector_db_initialization),
        ("Log Ingestion with Vector DB", test_log_ingestion_with_vector_db),
        ("LSTM + RAG Filtering", test_lstm_rag_filtering),
        ("Vector DB Operations", test_vector_db_operations),
        ("Complete Workflow", test_complete_workflow)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:30} : {status}")
        if not result:
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("🎉 ALL LSTM + RAG INTEGRATION TESTS PASSED!")
        logger.info("The system is ready for enhanced RCA analysis")
    else:
        logger.error("❌ SOME TESTS FAILED - Please check the issues above")
    
    logger.info("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1) 