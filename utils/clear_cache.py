#!/usr/bin/env python3
"""
Log Cache Clearing Utility
Provides easy ways to clear and manage the log cache
"""

import os
import sys
from pathlib import Path

# Add project root to path (parent of utils directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.log_cache import LogCache
from config.config import Config

def clear_all_cache():
    """Clear all log cache"""
    print("🗑️ Clearing all log cache...")
    cache = LogCache()
    cache.clear_cache()
    print("✅ All cache cleared!")

def clear_specific_cache(log_path: str):
    """Clear cache for specific log path"""
    print(f"🗑️ Clearing cache for: {log_path}")
    cache = LogCache()
    
    # Get cache key for the path
    cache_key = cache._get_cache_key(log_path)
    cache.clear_cache(cache_key)
    print(f"✅ Cache cleared for: {log_path}")

def show_cache_stats():
    """Show cache statistics"""
    print("📊 Cache Statistics:")
    print("=" * 40)
    
    cache = LogCache()
    stats = cache.get_cache_stats()
    
    print(f"Cache Directory: {stats['cache_dir']}")
    print(f"Total Entries: {stats['total_entries']}")
    print(f"Cache Size: {stats['cache_size_mb']:.2f} MB")
    
    if stats['oldest_entry']:
        print(f"Oldest Entry: {stats['oldest_entry']}")
    if stats['newest_entry']:
        print(f"Newest Entry: {stats['newest_entry']}")
    
    # Show individual cache files
    cache_dir = Path(stats['cache_dir'])
    if cache_dir.exists():
        print(f"\nCache Files:")
        for cache_file in cache_dir.glob('logs_*.pkl'):
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"  • {cache_file.name}: {size_mb:.2f} MB")

def cleanup_expired_cache():
    """Remove expired cache entries"""
    print("🧹 Cleaning up expired cache entries...")
    cache = LogCache()
    cache.cleanup_expired_cache()
    print("✅ Expired cache cleanup completed!")

def force_reload_cache(log_path: str = None):
    """Force reload cache by clearing and rebuilding"""
    if log_path is None:
        log_path = Config.DATA_DIR
    
    print(f"🔄 Force reloading cache for: {log_path}")
    
    # Clear existing cache
    cache = LogCache()
    cache_key = cache._get_cache_key(log_path)
    cache.clear_cache(cache_key)
    
    # Force reload
    logs_df = cache.get_cached_logs(log_path, force_reload=True)
    print(f"✅ Cache reloaded with {len(logs_df)} logs!")

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Log Cache Management Utility')
    parser.add_argument('--action', choices=['clear-all', 'clear-specific', 'stats', 'cleanup', 'reload'], 
                       default='stats', help='Action to perform')
    parser.add_argument('--log-path', type=str, help='Log path for specific operations')
    
    args = parser.parse_args()
    
    if args.action == 'clear-all':
        clear_all_cache()
    
    elif args.action == 'clear-specific':
        if not args.log_path:
            print("❌ Please specify --log-path for clear-specific action")
            return
        clear_specific_cache(args.log_path)
    
    elif args.action == 'stats':
        show_cache_stats()
    
    elif args.action == 'cleanup':
        cleanup_expired_cache()
    
    elif args.action == 'reload':
        force_reload_cache(args.log_path)

if __name__ == "__main__":
    main() 