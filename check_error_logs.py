#!/usr/bin/env python3
"""
Simple script to check for ERROR logs in the data
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.log_cache import LogCache
from config.config import Config

def check_error_logs():
    """Check for ERROR logs in the cached data"""
    print("🔍 Checking for ERROR logs in cached data...")
    
    cache = LogCache()
    logs_df = cache.get_cached_logs(Config.DATA_DIR)
    
    if logs_df.empty:
        print("❌ No logs found in cache")
        return
    
    print(f"📊 Total logs: {len(logs_df)}")
    
    # Check log levels
    if 'level' in logs_df.columns:
        level_counts = logs_df['level'].value_counts()
        print("\n📋 Log Level Distribution:")
        for level, count in level_counts.items():
            print(f"  {level}: {count}")
        
        # Check for ERROR logs
        error_logs = logs_df[logs_df['level'].str.upper() == 'ERROR']
        print(f"\n🚨 ERROR Logs Found: {len(error_logs)}")
        
        if len(error_logs) > 0:
            print("\n📝 Sample ERROR Logs:")
            for i, (_, log) in enumerate(error_logs.head(3).iterrows(), 1):
                message = log.get('message', '')[:100]
                service = log.get('service_type', 'unknown')
                print(f"  {i}. [{service}] {message}...")
        else:
            print("⚠️ No ERROR logs found in the data")
            print("   This explains why ERROR logs don't appear in RCA results")
    
    # Check for error-related keywords in messages
    if 'message' in logs_df.columns:
        error_keywords = ['error', 'failed', 'failure', 'exception', 'timeout', 'no valid host']
        print(f"\n🔍 Checking for error-related keywords in messages:")
        
        for keyword in error_keywords:
            matches = logs_df[logs_df['message'].str.contains(keyword, case=False, na=False)]
            if len(matches) > 0:
                print(f"  '{keyword}': {len(matches)} matches")
                # Show sample
                sample = matches.iloc[0]['message'][:80]
                print(f"    Sample: {sample}...")

if __name__ == "__main__":
    check_error_logs() 