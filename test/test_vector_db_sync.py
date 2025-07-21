#!/usr/bin/env python3
"""
Test script to verify VectorDB sync functionality
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.vector_db_service import VectorDBService

def test_vector_db_sync():
    """Test VectorDB sync functionality"""
    print("🧪 Testing VectorDB Sync Functionality")
    print("=" * 50)
    
    try:
        # Initialize VectorDB service
        vector_db = VectorDBService()
        
        # Get collection stats
        stats = vector_db.get_collection_stats()
        print(f"📊 VectorDB Status: {stats['total_documents']} documents")
        
        if stats['total_documents'] == 0:
            print("❌ No documents in VectorDB. Please ingest logs first.")
            return
        
        # Test loading data from VectorDB
        print("\n🔍 Testing data loading from VectorDB...")
        
        # Get all documents
        results = vector_db.collection.get()
        
        if not results['documents']:
            print("❌ No documents retrieved from VectorDB")
            return
        
        print(f"✅ Retrieved {len(results['documents'])} documents from VectorDB")
        
        # Convert to DataFrame (same logic as in the UI)
        logs_data = []
        for i, doc in enumerate(results['documents']):
            metadata = results['metadatas'][i] if results['metadatas'] else {}
            
            log_entry = {
                'message': doc,
                'timestamp': metadata.get('timestamp'),
                'level': metadata.get('level', 'INFO'),
                'service_type': metadata.get('service_type', 'unknown'),
                'instance_id': metadata.get('instance_id'),
                'request_id': metadata.get('request_id'),
                'source_file': metadata.get('source_file', 'vector_db')
            }
            logs_data.append(log_entry)
        
        df = pd.DataFrame(logs_data)
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Successfully converted to DataFrame: {len(df)} rows")
        
        # Show data statistics
        print(f"\n📊 Data Statistics:")
        print(f"   - Total logs: {len(df)}")
        
        if 'level' in df.columns:
            level_counts = df['level'].value_counts()
            print(f"   - Log levels: {dict(level_counts)}")
        
        if 'service_type' in df.columns:
            service_counts = df['service_type'].value_counts()
            print(f"   - Top services: {dict(service_counts.head(3))}")
        
        if 'timestamp' in df.columns:
            print(f"   - Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        print(f"\n✅ VectorDB sync test completed successfully!")
        print(f"🎯 UI can now load {len(df)} logs from VectorDB (same as RCA analysis)")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_db_sync() 