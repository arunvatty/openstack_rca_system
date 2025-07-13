#!/usr/bin/env python3
"""
Test script for Hybrid RCA Analyzer
Demonstrates the performance and quality improvements of the hybrid approach
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.rca_analyzer import RCAAnalyzer
from models.lstm_classifier import LSTMLogClassifier
from utils.log_cache import LogCache
from config.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hybrid_vs_original():
    """Compare Hybrid RCA Analyzer with original RCA Analyzer"""
    logger.info("="*60)
    logger.info("HYBRID RCA ANALYZER TEST")
    logger.info("="*60)
    
    # Check for API key
    if not Config.ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set. Please set the environment variable.")
        return
    
    # Initialize log cache
    log_cache = LogCache()
    
    # Load LSTM model
    model_path = os.path.join(Config.MODELS_DIR, 'lstm_log_classifier.keras')
    lstm_model = None
    
    if os.path.exists(model_path):
        logger.info("Loading LSTM model...")
        lstm_model = LSTMLogClassifier(Config.LSTM_CONFIG)
        lstm_model.load_model(model_path)
        logger.info("✓ LSTM model loaded")
    else:
        logger.warning("LSTM model not found. Some features will be limited.")
    
    # Get cached logs
    logger.info("Loading logs from cache...")
    logs_df = log_cache.get_cached_logs(Config.DATA_DIR)
    
    if logs_df.empty:
        logger.error("No logs available for testing")
        return
    
    logger.info(f"✓ Loaded {len(logs_df)} logs for testing")
    
    # Test queries
    test_queries = [
        "Instance launch failures",
        "Resource allocation problems",
        "Network connectivity issues",
        "Authentication errors"
    ]
    
    # Initialize analyzers
    hybrid_analyzer = RCAAnalyzer(Config.ANTHROPIC_API_KEY, lstm_model)
    original_analyzer = RCAAnalyzer(Config.ANTHROPIC_API_KEY, lstm_model)
    
    logger.info("="*60)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("="*60)
    
    for query in test_queries:
        logger.info(f"\nTesting query: '{query}'")
        logger.info("-" * 40)
        
        # Test Hybrid Analyzer
        logger.info("Testing Hybrid RCA Analyzer...")
        start_time = time.time()
        
        try:
            hybrid_results = hybrid_analyzer.analyze_issue(query, logs_df, fast_mode=False)
            hybrid_time = time.time() - start_time
            
            logger.info(f"✓ Hybrid Analysis completed in {hybrid_time:.2f}s")
            logger.info(f"  - Relevant logs: {hybrid_results['relevant_logs_count']}")
            logger.info(f"  - Analysis mode: {hybrid_results['analysis_mode']}")
            
            # Show performance metrics
            if 'performance_metrics' in hybrid_results:
                metrics = hybrid_results['performance_metrics']
                logger.info(f"  - Processing time: {metrics.get('processing_time', 0):.2f}s")
                logger.info(f"  - LSTM available: {metrics.get('lstm_available', False)}")
                logger.info(f"  - Vector DB available: {metrics.get('vector_db_available', False)}")
            
        except Exception as e:
            logger.error(f"✗ Hybrid analysis failed: {e}")
            hybrid_time = float('inf')
            hybrid_results = None
        
        # Test Original Analyzer (if available)
        logger.info("\nTesting Original RCA Analyzer...")
        start_time = time.time()
        
        try:
            original_results = original_analyzer.analyze_issue(query, logs_df, fast_mode=False)
            original_time = time.time() - start_time
            
            logger.info(f"✓ Original Analysis completed in {original_time:.2f}s")
            logger.info(f"  - Relevant logs: {original_results['relevant_logs_count']}")
            logger.info(f"  - Analysis mode: {original_results['analysis_mode']}")
            
        except Exception as e:
            logger.error(f"✗ Original analysis failed: {e}")
            original_time = float('inf')
            original_results = None
        
        # Compare results
        if hybrid_results and original_results:
            logger.info("\nComparison:")
            logger.info(f"  - Speed improvement: {original_time/hybrid_time:.1f}x faster")
            logger.info(f"  - Log filtering: {hybrid_results['relevant_logs_count']} vs {original_results['relevant_logs_count']}")
        
        logger.info("-" * 40)
    
    # Test fast mode
    logger.info("\n" + "="*60)
    logger.info("FAST MODE TEST")
    logger.info("="*60)
    
    query = "Instance launch failures"
    logger.info(f"Testing fast mode with query: '{query}'")
    
    start_time = time.time()
    fast_results = hybrid_analyzer.analyze_issue(query, logs_df, fast_mode=True)
    fast_time = time.time() - start_time
    
    logger.info(f"✓ Fast mode completed in {fast_time:.2f}s")
    logger.info(f"  - Relevant logs: {fast_results['relevant_logs_count']}")
    logger.info(f"  - Analysis mode: {fast_results['analysis_mode']}")
    
    # Show top filtered logs
    if 'filtered_logs' in fast_results and not fast_results['filtered_logs'].empty:
        logger.info("\nTop 3 filtered logs:")
        top_logs = fast_results['filtered_logs'].head(3)
        for i, (_, log) in enumerate(top_logs.iterrows(), 1):
            score = log.get('combined_score', log.get('lstm_importance', 0))
            logger.info(f"  {i}. [{log.get('level', 'INFO')}] {log.get('message', '')[:80]}... (Score: {score:.3f})")
    
    # Cache statistics
    logger.info("\n" + "="*60)
    logger.info("CACHE STATISTICS")
    logger.info("="*60)
    
    cache_stats = log_cache.get_cache_stats()
    logger.info(f"Cache directory: {cache_stats['cache_dir']}")
    logger.info(f"Total entries: {cache_stats['total_entries']}")
    logger.info(f"Cache size: {cache_stats['cache_size_mb']:.2f} MB")
    
    if cache_stats['oldest_entry']:
        logger.info(f"Oldest entry: {cache_stats['oldest_entry']}")
    if cache_stats['newest_entry']:
        logger.info(f"Newest entry: {cache_stats['newest_entry']}")
    
    logger.info("\n" + "="*60)
    logger.info("TEST COMPLETED")
    logger.info("="*60)

def test_hybrid_features():
    """Test specific features of the Hybrid RCA Analyzer"""
    logger.info("="*60)
    logger.info("HYBRID FEATURES TEST")
    logger.info("="*60)
    
    # Check for API key
    if not Config.ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set. Please set the environment variable.")
        return
    
    # Initialize components
    log_cache = LogCache()
    
    # Load LSTM model
    model_path = os.path.join(Config.MODELS_DIR, 'lstm_log_classifier.keras')
    lstm_model = None
    
    if os.path.exists(model_path):
        lstm_model = LSTMLogClassifier(Config.LSTM_CONFIG)
        lstm_model.load_model(model_path)
        logger.info("✓ LSTM model loaded")
    
    # Initialize Hybrid analyzer
    hybrid_analyzer = RCAAnalyzer(Config.ANTHROPIC_API_KEY, lstm_model)
    
    # Get logs
    logs_df = log_cache.get_cached_logs(Config.DATA_DIR)
    
    if logs_df.empty:
        logger.error("No logs available for testing")
        return
    
    # Test LSTM filtering
    logger.info("\nTesting LSTM filtering...")
    try:
        lstm_filtered = hybrid_analyzer._lstm_filter_logs(logs_df, "test query")
        logger.info(f"✓ LSTM filtered {len(lstm_filtered)} logs from {len(logs_df)}")
        
        if 'lstm_importance' in lstm_filtered.columns:
            importance_range = f"{lstm_filtered['lstm_importance'].min():.3f} - {lstm_filtered['lstm_importance'].max():.3f}"
            logger.info(f"  - Importance score range: {importance_range}")
        
    except Exception as e:
        logger.error(f"✗ LSTM filtering failed: {e}")
    
    # Test Vector DB search
    logger.info("\nTesting Vector DB search...")
    try:
        if lstm_filtered is not None and not lstm_filtered.empty:
            vector_results = hybrid_analyzer._vector_db_search(lstm_filtered, "test query")
            logger.info(f"✓ Vector DB found {len(vector_results)} similar logs")
        else:
            logger.info("Skipping Vector DB test (no LSTM filtered logs)")
    except Exception as e:
        logger.error(f"✗ Vector DB search failed: {e}")
    
    # Test issue categorization
    logger.info("\nTesting issue categorization...")
    test_issues = [
        "Instance launch failures",
        "Memory allocation problems", 
        "Network timeout issues",
        "Authentication token expired"
    ]
    
    for issue in test_issues:
        category = hybrid_analyzer._categorize_issue(issue)
        logger.info(f"  - '{issue}' → {category}")
    
    # Test pattern analysis
    logger.info("\nTesting pattern analysis...")
    if lstm_filtered is not None and not lstm_filtered.empty:
        patterns = hybrid_analyzer._analyze_patterns(lstm_filtered, "resource_shortage")
        logger.info(f"✓ Pattern analysis completed")
        logger.info(f"  - Error count: {patterns.get('error_count', 0)}")
        logger.info(f"  - Service distribution: {len(patterns.get('service_distribution', {}))} services")
        
        if 'resource_patterns' in patterns:
            resource_patterns = patterns['resource_patterns']
            logger.info(f"  - Resource patterns: {resource_patterns}")
    
    logger.info("\n" + "="*60)
    logger.info("FEATURES TEST COMPLETED")
    logger.info("="*60)

def main():
    """Main test function"""
    logger.info("Starting Hybrid RCA Analyzer tests...")
    
    # Test 1: Performance comparison
    test_hybrid_vs_original()
    
    # Test 2: Feature testing
    test_hybrid_features()
    
    logger.info("All tests completed!")

if __name__ == "__main__":
    main() 