#!/usr/bin/env python3
"""
Test script to verify prompt functionality and ERROR log prioritization in RCA analyzer
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lstm.rca_analyzer import RCAAnalyzer
from config.config import Config

def create_sample_logs():
    """Create sample log data for testing with mixed log levels"""
    sample_data = [
        {
            'timestamp': datetime.now(),
            'level': 'ERROR',
            'service_type': 'nova-compute',
            'message': 'No valid host was found for instance launch',
            'instance_id': 'test-instance-1'
        },
        {
            'timestamp': datetime.now(),
            'level': 'INFO',
            'service_type': 'nova-api',
            'message': 'GET /v2/servers/detail HTTP/1.1 status: 200',
            'instance_id': None
        },
        {
            'timestamp': datetime.now(),
            'level': 'ERROR',
            'service_type': 'nova-scheduler',
            'message': 'Failed to schedule instance: insufficient resources',
            'instance_id': 'test-instance-2'
        },
        {
            'timestamp': datetime.now(),
            'level': 'WARNING',
            'service_type': 'nova-compute',
            'message': 'High memory usage detected on compute node',
            'instance_id': None
        },
        {
            'timestamp': datetime.now(),
            'level': 'INFO',
            'service_type': 'nova-compute',
            'message': 'Instance spawned successfully',
            'instance_id': 'test-instance-3'
        }
    ]
    
    return pd.DataFrame(sample_data)

def test_rca_analysis():
    """Test RCA analysis with improved ERROR log identification"""
    print("🧪 Testing RCA Analysis with ERROR Log Prioritization")
    print("=" * 60)
    
    # Create sample data
    logs_df = create_sample_logs()
    print(f"📊 Created {len(logs_df)} sample logs:")
    print(f"   - ERROR logs: {len(logs_df[logs_df['level'] == 'ERROR'])}")
    print(f"   - WARNING logs: {len(logs_df[logs_df['level'] == 'WARNING'])}")
    print(f"   - INFO logs: {len(logs_df[logs_df['level'] == 'INFO'])}")
    
    # Test issue description that should trigger ERROR log identification
    issue_description = "I'm having trouble launching VMs. The system keeps saying 'No valid host was found' and there are resource allocation failures."
    
    print(f"\n🔍 Testing issue: {issue_description}")
    
    try:
        # Initialize RCA analyzer (without API key for testing)
        rca_analyzer = RCAAnalyzer('dummy-key')
        
        # Perform analysis
        results = rca_analyzer.analyze_issue(issue_description, logs_df, fast_mode=False)
        
        # Check results
        print(f"\n✅ Analysis completed!")
        print(f"📋 Issue Category: {results.get('issue_category', 'Unknown')}")
        print(f"🔍 Relevant Logs Found: {results.get('relevant_logs_count', 0)}")
        
        # Check if ERROR logs were identified
        if 'filtered_logs' in results and not results['filtered_logs'].empty:
            filtered_logs = results['filtered_logs']
            print(f"\n📊 Filtered Logs Breakdown:")
            
            if 'level' in filtered_logs.columns:
                level_counts = filtered_logs['level'].value_counts()
                for level, count in level_counts.items():
                    print(f"   - {level}: {count} logs")
                
                # Check if ERROR logs are in the filtered results
                error_count = level_counts.get('ERROR', 0)
                if error_count > 0:
                    print(f"✅ SUCCESS: {error_count} ERROR logs identified as important!")
                else:
                    print("⚠️ WARNING: No ERROR logs identified as important")
            
            # Show top filtered logs
            print(f"\n🔝 Top Filtered Logs:")
            for i, (_, log) in enumerate(filtered_logs.head(3).iterrows(), 1):
                level = log.get('level', 'UNKNOWN')
                message = log.get('message', '')[:80]
                importance = log.get('lstm_importance', 0)
                print(f"   {i}. [{level}] {message}... (Importance: {importance:.3f})")
        
        # Check prompt
        prompt = results.get('prompt', '')
        print(f"\n📝 Prompt Length: {len(prompt)} characters")
        if prompt:
            print("✅ Prompt captured successfully")
        else:
            print("⚠️ No prompt captured")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rca_analysis() 