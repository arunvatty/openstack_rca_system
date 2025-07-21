#!/usr/bin/env python3
"""
Log Cache Check Utility
Check the number of log entries in cache and show detailed information
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to path (parent of utils directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.log_cache import LogCache
from config.config import Config

def check_cache_logs(log_path: str = None):
    """Check log entries in cache for a specific path or all paths"""
    if log_path is None:
        log_path = Config.DATA_DIR
    
    print(f"🔍 Checking cache for: {log_path}")
    print("=" * 60)
    
    cache = LogCache()
    
    try:
        # Get cached logs
        logs_df = cache.get_cached_logs(log_path)
        
        if logs_df.empty:
            print("❌ No logs found in cache for this path")
            return
        
        print(f"✅ Found {len(logs_df)} log entries in cache")
        print()
        
        # Show basic statistics
        print("📊 LOG STATISTICS:")
        print("-" * 30)
        print(f"Total Logs: {len(logs_df)}")
        
        # Show log levels distribution
        if 'level' in logs_df.columns:
            level_counts = logs_df['level'].value_counts()
            print(f"\n📋 Log Levels:")
            for level, count in level_counts.items():
                print(f"  {level}: {count}")
        
        # Show service types distribution
        if 'service_type' in logs_df.columns:
            service_counts = logs_df['service_type'].value_counts()
            print(f"\n🔧 Service Types:")
            for service, count in service_counts.head(10).items():
                print(f"  {service}: {count}")
            if len(service_counts) > 10:
                print(f"  ... and {len(service_counts) - 10} more services")
        
        # Show time range
        if 'timestamp' in logs_df.columns:
            print(f"\n⏰ Time Range:")
            print(f"  Start: {logs_df['timestamp'].min()}")
            print(f"  End:   {logs_df['timestamp'].max()}")
        
        # Show sample logs
        print(f"\n📝 SAMPLE LOGS (first 5):")
        print("-" * 30)
        for i, (_, log) in enumerate(logs_df.head(5).iterrows(), 1):
            level = log.get('level', 'UNKNOWN')
            service = log.get('service_type', 'unknown')
            message = log.get('message', '')[:80]
            timestamp = log.get('timestamp', '')
            
            print(f"{i}. [{level}] {service}: {message}...")
            print(f"   Timestamp: {timestamp}")
            print()
        
        # Show error logs if any
        if 'level' in logs_df.columns:
            error_logs = logs_df[logs_df['level'].str.upper() == 'ERROR']
            if not error_logs.empty:
                print(f"🚨 ERROR LOGS ({len(error_logs)} found):")
                print("-" * 30)
                for i, (_, log) in enumerate(error_logs.head(3).iterrows(), 1):
                    service = log.get('service_type', 'unknown')
                    message = log.get('message', '')[:100]
                    timestamp = log.get('timestamp', '')
                    
                    print(f"{i}. [{service}] {message}...")
                    print(f"   Timestamp: {timestamp}")
                    print()
                
                if len(error_logs) > 3:
                    print(f"   ... and {len(error_logs) - 3} more ERROR logs")
        
        # Show cache file info
        cache_key = cache._get_cache_key(log_path)
        cache_file = cache._get_cache_file(cache_key)
        
        if cache_file.exists():
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"\n💾 CACHE FILE INFO:")
            print(f"  File: {cache_file}")
            print(f"  Size: {size_mb:.2f} MB")
            print(f"  Cache Key: {cache_key}")
        
    except Exception as e:
        print(f"❌ Error checking cache: {e}")

def check_all_cache_entries():
    """Check all cache entries"""
    print("🔍 Checking all cache entries")
    print("=" * 60)
    
    cache = LogCache()
    stats = cache.get_cache_stats()
    
    print(f"📊 CACHE OVERVIEW:")
    print(f"  Total Entries: {stats['total_entries']}")
    print(f"  Cache Size: {stats['cache_size_mb']:.2f} MB")
    print(f"  Cache Directory: {stats['cache_dir']}")
    
    if stats['oldest_entry']:
        print(f"  Oldest Entry: {stats['oldest_entry']}")
    if stats['newest_entry']:
        print(f"  Newest Entry: {stats['newest_entry']}")
    
    print()
    
    # Check each cache entry
    cache_dir = Path(stats['cache_dir'])
    if cache_dir.exists():
        cache_files = list(cache_dir.glob('logs_*.pkl'))
        
        if not cache_files:
            print("❌ No cache files found")
            return
        
        print(f"📁 CACHE FILES ({len(cache_files)} found):")
        print("-" * 40)
        
        total_logs = 0
        for cache_file in cache_files:
            try:
                # Try to load the cache file
                with open(cache_file, 'rb') as f:
                    import pickle
                    cached_data = pickle.load(f)
                    logs_df = cached_data.get('logs_df', pd.DataFrame())
                    log_count = len(logs_df)
                    total_logs += log_count
                    
                    size_mb = cache_file.stat().st_size / (1024 * 1024)
                    print(f"  • {cache_file.name}")
                    print(f"    Size: {size_mb:.2f} MB")
                    print(f"    Logs: {log_count}")
                    
                    # Show log levels if available
                    if not logs_df.empty and 'level' in logs_df.columns:
                        level_counts = logs_df['level'].value_counts()
                        level_summary = ", ".join([f"{level}:{count}" for level, count in level_counts.items()])
                        print(f"    Levels: {level_summary}")
                    
                    print()
                    
            except Exception as e:
                print(f"  • {cache_file.name} - Error reading: {e}")
                print()
        
        print(f"📈 TOTAL LOGS IN CACHE: {total_logs}")

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Log Cache Check Utility')
    parser.add_argument('--path', type=str, help='Specific log path to check (default: all)')
    parser.add_argument('--all', action='store_true', help='Check all cache entries')
    
    args = parser.parse_args()
    
    if args.all:
        check_all_cache_entries()
    else:
        check_cache_logs(args.path)

if __name__ == "__main__":
    main() 