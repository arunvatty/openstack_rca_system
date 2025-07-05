#!/usr/bin/env python3
"""
Vector Database Query Utility
A command-line tool to query and explore the ChromaDB vector database
"""

import os
import sys
import argparse
import json
import pandas as pd
from typing import List, Dict, Optional
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vector_db_service import VectorDBService
from config.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDBQueryTool:
    """Utility tool for querying the vector database"""
    
    def __init__(self):
        try:
            self.vector_db = VectorDBService()
            logger.info("VectorDBQueryTool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VectorDBQueryTool: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            stats = self.vector_db.get_collection_stats()
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def search_similar_logs(self, query: str, top_k: int = 10, 
                          filter_metadata: Dict = None, include_chunks: bool = False) -> List[Dict]:
        """Search for logs similar to the query"""
        try:
            similar_logs = self.vector_db.search_similar_logs(
                query, top_k=top_k, filter_metadata=filter_metadata, include_chunks=include_chunks
            )
            return similar_logs
        except Exception as e:
            logger.error(f"Failed to search similar logs: {e}")
            return []
    
    def get_historical_context(self, issue_description: str, top_k: int = 5) -> str:
        """Get historical context for an issue"""
        try:
            context = self.vector_db.get_context_for_issue(issue_description, top_k=top_k)
            return context
        except Exception as e:
            logger.error(f"Failed to get historical context: {e}")
            return ""
    
    def search_by_service(self, service_type: str, top_k: int = 20) -> List[Dict]:
        """Search for logs by service type"""
        try:
            filter_metadata = {'service_type': service_type}
            similar_logs = self.vector_db.search_similar_logs(
                "", top_k=top_k, filter_metadata=filter_metadata
            )
            return similar_logs
        except Exception as e:
            logger.error(f"Failed to search by service: {e}")
            return []
    
    def search_by_level(self, level: str, top_k: int = 20) -> List[Dict]:
        """Search for logs by log level"""
        try:
            filter_metadata = {'level': level}
            similar_logs = self.vector_db.search_similar_logs(
                "", top_k=top_k, filter_metadata=filter_metadata
            )
            return similar_logs
        except Exception as e:
            logger.error(f"Failed to search by level: {e}")
            return []
    
    def search_by_instance(self, instance_id: str, top_k: int = 20) -> List[Dict]:
        """Search for logs by instance ID"""
        try:
            filter_metadata = {'instance_id': instance_id}
            similar_logs = self.vector_db.search_similar_logs(
                "", top_k=top_k, filter_metadata=filter_metadata
            )
            return similar_logs
        except Exception as e:
            logger.error(f"Failed to search by instance: {e}")
            return []
    
    def export_collection_to_csv(self, output_file: str = "vector_db_export.csv"):
        """Export all logs from the collection to CSV"""
        try:
            # Get all documents from collection
            results = self.vector_db.collection.get()
            
            if not results['documents']:
                logger.warning("No documents found in collection")
                return False
            
            # Create DataFrame
            data = []
            for i, doc in enumerate(results['documents']):
                metadata = results['metadatas'][i] if results['metadatas'] else {}
                data.append({
                    'document': doc,
                    'id': results['ids'][i],
                    'timestamp': metadata.get('timestamp', ''),
                    'service_type': metadata.get('service_type', ''),
                    'level': metadata.get('level', ''),
                    'instance_id': metadata.get('instance_id', ''),
                    'original_index': metadata.get('original_index', '')
                })
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(df)} logs to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export collection: {e}")
            return False
    
    def get_service_distribution(self) -> Dict:
        """Get distribution of logs by service type"""
        try:
            results = self.vector_db.collection.get()
            
            if not results['metadatas']:
                return {}
            
            service_counts = {}
            for metadata in results['metadatas']:
                service = metadata.get('service_type', 'unknown')
                service_counts[service] = service_counts.get(service, 0) + 1
            
            return service_counts
            
        except Exception as e:
            logger.error(f"Failed to get service distribution: {e}")
            return {}
    
    def get_level_distribution(self) -> Dict:
        """Get distribution of logs by level"""
        try:
            results = self.vector_db.collection.get()
            
            if not results['metadatas']:
                return {}
            
            level_counts = {}
            for metadata in results['metadatas']:
                level = metadata.get('level', 'unknown')
                level_counts[level] = level_counts.get(level, 0) + 1
            
            return level_counts
            
        except Exception as e:
            logger.error(f"Failed to get level distribution: {e}")
            return {}
    
    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            self.vector_db.clear_collection()
            logger.info("Collection cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
    
    def get_config_info(self) -> Dict:
        """Get vector database configuration information"""
        try:
            config_info = self.vector_db.get_config_info()
            return config_info
        except Exception as e:
            logger.error(f"Failed to get config info: {e}")
            return {}

def print_similar_logs(similar_logs: List[Dict], query: str):
    """Pretty print similar logs"""
    if not similar_logs:
        print("No similar logs found.")
        return
    
    print(f"\n🔍 Similar logs for query: '{query}'")
    print("=" * 80)
    
    for i, log in enumerate(similar_logs, 1):
        similarity = 1 - log['distance']
        print(f"\n📋 Log {i} (Similarity: {similarity:.3f})")
        print(f"   ID: {log['id']}")
        print(f"   Document: {log['document']}")
        print(f"   Service: {log['metadata']['service_type']}")
        print(f"   Level: {log['metadata']['level']}")
        print(f"   Timestamp: {log['metadata']['timestamp']}")
        if log['metadata']['instance_id']:
            print(f"   Instance: {log['metadata']['instance_id']}")

def print_collection_stats(stats: Dict):
    """Pretty print collection statistics"""
    print("\n📊 Collection Statistics")
    print("=" * 40)
    for key, value in stats.items():
        if key == 'total_documents':
            print(f"   Total Documents: {value}")
        elif key == 'chunked_documents':
            print(f"   Chunked Documents: {value}")
        elif key == 'non_chunked_documents':
            print(f"   Non-chunked Documents: {value}")
        elif key == 'embedding_dimensions':
            print(f"   Embedding Dimensions: {value}")
        elif key == 'distance_metric':
            print(f"   Distance Metric: {value}")
        else:
            print(f"   {key}: {value}")

def print_config_info(config_info: Dict):
    """Pretty print configuration information"""
    print("\n⚙️ Vector Database Configuration")
    print("=" * 50)
    for key, value in config_info.items():
        if key == 'embedding_dimensions':
            print(f"   Embedding Dimensions: {value}")
        elif key == 'distance_metric':
            print(f"   Distance Metric: {value}")
        elif key == 'chunk_size':
            print(f"   Chunk Size: {value}")
        elif key == 'chunk_overlap':
            print(f"   Chunk Overlap: {value}")
        elif key == 'max_text_length':
            print(f"   Max Text Length: {value}")
        elif key == 'similarity_threshold':
            print(f"   Similarity Threshold: {value}")
        elif key == 'top_k_results':
            print(f"   Top K Results: {value}")
        else:
            print(f"   {key}: {value}")

def print_distribution(title: str, distribution: Dict):
    """Pretty print distribution data"""
    if not distribution:
        print(f"\n❌ No {title.lower()} data available")
        return
    
    print(f"\n📈 {title}")
    print("=" * 40)
    total = sum(distribution.values())
    for item, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"   {item}: {count} ({percentage:.1f}%)")

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description="Vector Database Query Utility")
    parser.add_argument("--action", choices=[
        "stats", "search", "context", "service", "level", "instance", 
        "export", "service-dist", "level-dist", "clear", "config"
    ], required=True, help="Action to perform")
    
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--service", type=str, help="Service type filter")
    parser.add_argument("--level", type=str, help="Log level filter")
    parser.add_argument("--instance", type=str, help="Instance ID filter")
    parser.add_argument("--output", type=str, default="vector_db_export.csv", help="Output file for export")
    parser.add_argument("--include-chunks", action="store_true", help="Include chunked documents in search")
    
    args = parser.parse_args()
    
    try:
        tool = VectorDBQueryTool()
        
        if args.action == "stats":
            stats = tool.get_collection_stats()
            print_collection_stats(stats)
            
        elif args.action == "config":
            config_info = tool.get_config_info()
            print_config_info(config_info)
            
        elif args.action == "search":
            if not args.query:
                print("❌ Error: --query is required for search action")
                return
            similar_logs = tool.search_similar_logs(args.query, args.top_k, include_chunks=args.include_chunks)
            print_similar_logs(similar_logs, args.query)
            
        elif args.action == "context":
            if not args.query:
                print("❌ Error: --query is required for context action")
                return
            context = tool.get_historical_context(args.query, args.top_k)
            if context:
                print(f"\n📚 Historical Context for: '{args.query}'")
                print("=" * 80)
                print(context)
            else:
                print("❌ No historical context found")
                
        elif args.action == "service":
            if not args.service:
                print("❌ Error: --service is required for service action")
                return
            similar_logs = tool.search_by_service(args.service, args.top_k)
            print_similar_logs(similar_logs, f"service={args.service}")
            
        elif args.action == "level":
            if not args.level:
                print("❌ Error: --level is required for level action")
                return
            similar_logs = tool.search_by_level(args.level, args.top_k)
            print_similar_logs(similar_logs, f"level={args.level}")
            
        elif args.action == "instance":
            if not args.instance:
                print("❌ Error: --instance is required for instance action")
                return
            similar_logs = tool.search_by_instance(args.instance, args.top_k)
            print_similar_logs(similar_logs, f"instance={args.instance}")
            
        elif args.action == "export":
            success = tool.export_collection_to_csv(args.output)
            if success:
                print(f"✅ Successfully exported to {args.output}")
            else:
                print("❌ Failed to export collection")
                
        elif args.action == "service-dist":
            distribution = tool.get_service_distribution()
            print_distribution("Service Distribution", distribution)
            
        elif args.action == "level-dist":
            distribution = tool.get_level_distribution()
            print_distribution("Log Level Distribution", distribution)
            
        elif args.action == "clear":
            confirm = input("⚠️  Are you sure you want to clear the collection? (yes/no): ")
            if confirm.lower() == 'yes':
                tool.clear_collection()
                print("✅ Collection cleared successfully")
            else:
                print("❌ Operation cancelled")
                
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 