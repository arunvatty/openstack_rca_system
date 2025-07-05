#!/usr/bin/env python3
"""
Vector Database Query Tool
Standalone script for querying ChromaDB vector database
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.vector_db_service import VectorDBService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Vector Database Query Tool')
    parser.add_argument('--action', choices=['stats', 'search', 'clear'], 
                       default='stats', help='Action to perform')
    parser.add_argument('--query', type=str, help='Search query (for search action)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--collection', type=str, default='openstack_logs', 
                       help='Collection name')
    
    args = parser.parse_args()
    
    try:
        # Initialize vector DB service
        vector_db = VectorDBService(collection_name=args.collection)
        
        if args.action == 'stats':
            # Get collection statistics
            stats = vector_db.get_collection_stats()
            
            print("\n" + "="*50)
            print("VECTOR DATABASE STATISTICS")
            print("="*50)
            print(f"Collection Name: {stats.get('collection_name', 'N/A')}")
            print(f"Total Documents: {stats.get('total_documents', 0)}")
            print(f"Chunked Documents: {stats.get('chunked_documents', 0)}")
            print(f"Non-chunked Documents: {stats.get('non_chunked_documents', 0)}")
            print(f"Embedding Model: {stats.get('embedding_model', 'N/A')}")
            print(f"Embedding Dimensions: {stats.get('embedding_dimensions', 'N/A')}")
            print(f"Distance Metric: {stats.get('distance_metric', 'N/A')}")
            print("="*50)
            
        elif args.action == 'search':
            if not args.query:
                print("Error: --query is required for search action")
                return
            
            # Search for similar logs
            similar_logs = vector_db.search_similar_logs(args.query, top_k=args.top_k)
            
            print(f"\nSearch Results for: '{args.query}'")
            print("="*50)
            
            if similar_logs:
                for i, log in enumerate(similar_logs, 1):
                    print(f"\n{i}. Similarity: {log['similarity']:.3f}")
                    print(f"   Document: {log['document'][:200]}...")
                    print(f"   Service: {log['metadata']['service_type']}")
                    print(f"   Level: {log['metadata']['level']}")
                    print(f"   Timestamp: {log['metadata']['timestamp']}")
            else:
                print("No similar logs found.")
                
        elif args.action == 'clear':
            # Clear collection
            confirm = input(f"Are you sure you want to clear collection '{args.collection}'? (y/N): ")
            if confirm.lower() == 'y':
                vector_db.clear_collection()
                print(f"Collection '{args.collection}' cleared successfully.")
            else:
                print("Operation cancelled.")
                
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 