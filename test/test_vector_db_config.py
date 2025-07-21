#!/usr/bin/env python3
"""
Test script for enhanced Vector Database Configuration
Demonstrates chunking, configuration parameters, and new features
"""

import os
import sys
import pandas as pd
from datetime import datetime
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from services.vector_db_service import VectorDBService
from data.log_ingestion import LogIngestionManager
from utils.vector_db_query import VectorDBQueryTool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_configuration_parameters():
    """Test the new configuration parameters"""
    print("🔧 Testing Vector Database Configuration Parameters")
    print("=" * 60)
    
    # Initialize vector database service
    vector_db = VectorDBService()
    
    # Get configuration info
    config_info = vector_db.get_config_info()
    
    print("\n📋 Configuration Parameters:")
    for key, value in config_info.items():
        print(f"   {key}: {value}")
    
    # Validate embedding dimensions
    print(f"\n✅ Embedding dimensions validated: {config_info['embedding_dimensions']}")
    print(f"✅ Distance metric: {config_info['distance_metric']}")
    print(f"✅ Chunk size: {config_info['chunk_size']}")
    print(f"✅ Chunk overlap: {config_info['chunk_overlap']}")
    print(f"✅ Max text length: {config_info['max_text_length']}")

def test_text_chunking():
    """Test text chunking functionality"""
    print("\n📄 Testing Text Chunking Functionality")
    print("=" * 60)
    
    vector_db = VectorDBService()
    
    # Test with short text (should not be chunked)
    short_text = "This is a short log message"
    chunks = vector_db._chunk_text(short_text)
    print(f"\nShort text: '{short_text}'")
    print(f"Chunks: {len(chunks)} (should be 1)")
    print(f"Content: {chunks}")
    
    # Test with long text (should be chunked)
    long_text = """
    This is a very long log message that contains detailed information about an OpenStack error.
    The error occurred during instance creation and involved multiple services including nova-api,
    nova-scheduler, and nova-compute. The error message indicates that there were insufficient
    resources available on the compute nodes, specifically related to memory allocation and
    disk space. The scheduler attempted to find a suitable host but failed after multiple
    attempts. This type of error typically occurs when the OpenStack cluster is under heavy
    load or when there are resource constraints that prevent new instances from being created.
    The error also includes stack trace information and debugging details that help identify
    the root cause of the failure.
    """ * 3  # Repeat to make it very long
    
    chunks = vector_db._chunk_text(long_text)
    print(f"\nLong text length: {len(long_text)} characters")
    print(f"Chunks: {len(chunks)} (should be > 1)")
    print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")
    
    # Test text truncation
    truncated = vector_db._truncate_text(long_text)
    print(f"\nTruncated text length: {len(truncated)} characters")
    print(f"Truncated preview: {truncated[:100]}...")

def test_chunked_ingestion():
    """Test ingestion with chunking enabled"""
    print("\n📥 Testing Chunked Log Ingestion")
    print("=" * 60)
    
    # Create sample log data
    sample_logs = [
        {
            'timestamp': datetime.now(),
            'service_type': 'nova-api',
            'level': 'ERROR',
            'message': 'Short error message',
            'instance_id': 'instance-1'
        },
        {
            'timestamp': datetime.now(),
            'service_type': 'nova-compute',
            'level': 'ERROR',
            'message': 'This is a very long error message that contains detailed information about an OpenStack error. The error occurred during instance creation and involved multiple services including nova-api, nova-scheduler, and nova-compute. The error message indicates that there were insufficient resources available on the compute nodes, specifically related to memory allocation and disk space. The scheduler attempted to find a suitable host but failed after multiple attempts. This type of error typically occurs when the OpenStack cluster is under heavy load or when there are resource constraints that prevent new instances from being created. The error also includes stack trace information and debugging details that help identify the root cause of the failure.',
            'instance_id': 'instance-2'
        }
    ]
    
    df = pd.DataFrame(sample_logs)
    
    # Test without chunking
    print("\n🔄 Testing without chunking:")
    vector_db_no_chunk = VectorDBService()
    vector_db_no_chunk.clear_collection()
    logs_added = vector_db_no_chunk.add_logs(df, enable_chunking=False)
    stats = vector_db_no_chunk.get_collection_stats()
    print(f"   Logs added: {logs_added}")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Chunked documents: {stats['chunked_documents']}")
    print(f"   Non-chunked documents: {stats['non_chunked_documents']}")
    
    # Test with chunking
    print("\n🔄 Testing with chunking:")
    vector_db_chunk = VectorDBService()
    vector_db_chunk.clear_collection()
    logs_added = vector_db_chunk.add_logs(df, enable_chunking=True)
    stats = vector_db_chunk.get_collection_stats()
    print(f"   Logs added: {logs_added}")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Chunked documents: {stats['chunked_documents']}")
    print(f"   Non-chunked documents: {stats['non_chunked_documents']}")

def test_enhanced_search():
    """Test enhanced search functionality"""
    print("\n🔍 Testing Enhanced Search Functionality")
    print("=" * 60)
    
    vector_db = VectorDBService()
    
    # Test search with and without chunks
    query = "OpenStack error insufficient resources"
    
    print(f"\nSearch query: '{query}'")
    
    # Search including chunks
    results_with_chunks = vector_db.search_similar_logs(query, top_k=5, include_chunks=True)
    print(f"\nResults with chunks (total: {len(results_with_chunks)}):")
    for i, result in enumerate(results_with_chunks[:3]):
        print(f"   {i+1}. Similarity: {result['similarity']:.3f}")
        print(f"      Document: {result['document'][:100]}...")
        print(f"      Chunked: {result['metadata']['is_chunked']}")
    
    # Search excluding chunks
    results_no_chunks = vector_db.search_similar_logs(query, top_k=5, include_chunks=False)
    print(f"\nResults without chunks (total: {len(results_no_chunks)}):")
    for i, result in enumerate(results_no_chunks[:3]):
        print(f"   {i+1}. Similarity: {result['similarity']:.3f}")
        print(f"      Document: {result['document'][:100]}...")
        print(f"      Chunked: {result['metadata']['is_chunked']}")

def test_query_tool_enhancements():
    """Test the enhanced query tool"""
    print("\n🛠️ Testing Enhanced Query Tool")
    print("=" * 60)
    
    tool = VectorDBQueryTool()
    
    # Test config action
    print("\n📋 Configuration Information:")
    config_info = tool.get_config_info()
    for key, value in config_info.items():
        print(f"   {key}: {value}")
    
    # Test enhanced stats
    print("\n📊 Enhanced Collection Statistics:")
    stats = tool.get_collection_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

def main():
    """Main test function"""
    print("🚀 Vector Database Configuration Enhancement Tests")
    print("=" * 80)
    
    try:
        # Test configuration parameters
        test_configuration_parameters()
        
        # Test text chunking
        test_text_chunking()
        
        # Test chunked ingestion
        test_chunked_ingestion()
        
        # Test enhanced search
        test_enhanced_search()
        
        # Test query tool enhancements
        test_query_tool_enhancements()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main() 