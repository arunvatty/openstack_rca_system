#!/usr/bin/env python3
"""
Consolidated Cache Management Utility

Combines functionality for checking, creating, and clearing log cache.
Replaces separate utilities: check_cache_logs.py, clear_cache.py, create_cache.py
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.log_cache import LogCache
from config.config import Config

def check_cache_status(log_path: str = None):
    """Check cache status and display detailed information"""
    print("🔍 Checking Log Cache Status")
    print("=" * 50)
    
    cache = LogCache()
    
    if log_path:
        # Check specific path
        print(f"📁 Checking cache for: {log_path}")
        
        if cache.is_cache_valid(log_path):
            print("✅ Cache is valid and up-to-date")
            
            # Try to get cache info
            try:
                df = cache.get_cached_logs(log_path)
                print(f"📊 Cached logs: {len(df)} entries")
                
                if 'level' in df.columns:
                    level_dist = df['level'].value_counts()
                    print("📈 Log level distribution:")
                    for level, count in level_dist.items():
                        print(f"   {level}: {count}")
                        
            except Exception as e:
                print(f"⚠️ Could not load cache details: {e}")
        else:
            print("❌ Cache is invalid or doesn't exist")
    else:
        # Check all cache entries
        try:
            stats = cache.get_cache_stats()
            print(f"📂 Cache directory: {stats['cache_dir']}")
            print(f"📊 Total cached datasets: {stats['total_datasets']}")
            print(f"💾 Cache size: {stats['cache_size_mb']:.2f} MB")
            
            if stats['total_datasets'] > 0:
                metadata = cache.load_cache_metadata()
                print("\n📋 Cache files:")
                for entry in metadata:
                    size_mb = entry.get('file_size', 0) / (1024 * 1024)
                    print(f"   {entry['hash'][:12]}...pkl "
                          f"({entry.get('logs_count', 0)} logs, {size_mb:.1f} MB)")
            else:
                print("📭 No cache entries found")
                
        except Exception as e:
            print(f"❌ Error checking cache: {e}")

def create_cache(log_path: str = None, force: bool = False):
    """Force cache creation for specified path"""
    if log_path is None:
        log_path = Config.DATA_DIR
        
    print(f"🔄 Creating cache for: {log_path}")
    
    cache = LogCache()
    
    try:
        if force and cache.is_cache_valid(log_path):
            print("🗑️ Clearing existing cache (force mode)")
            cache.clear_cache_for_path(log_path)
        
        print("📊 Processing logs and creating cache...")
        df = cache.get_cached_logs(log_path)
        
        if not df.empty:
            print(f"✅ Cache created successfully!")
            print(f"📊 Cached {len(df)} log entries")
            
            # Show basic stats
            if 'level' in df.columns:
                level_counts = df['level'].value_counts()
                print("📈 Log levels:")
                for level, count in level_counts.head().items():
                    print(f"   {level}: {count}")
        else:
            print("⚠️ No logs found to cache")
            
    except Exception as e:
        print(f"❌ Failed to create cache: {e}")

def clear_cache(target: str = "all", confirm: bool = False):
    """Clear cache files"""
    cache = LogCache()
    
    if target == "all":
        print("🗑️ Clearing ALL cache files...")
        if not confirm:
            response = input("⚠️ This will delete all cache files. Continue? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("❌ Operation cancelled")
                return
        
        try:
            cache.clear_cache()
            print("✅ All cache files cleared successfully")
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")
            
    elif target.startswith("hash:"):
        # Clear specific hash
        hash_prefix = target[5:]
        print(f"🗑️ Clearing cache entry: {hash_prefix}...")
        
        try:
            metadata = cache.load_cache_metadata()
            matching_entries = [entry for entry in metadata if entry['hash'].startswith(hash_prefix)]
            
            if not matching_entries:
                print(f"❌ No cache entry found matching: {hash_prefix}")
                return
                
            for entry in matching_entries:
                cache.clear_cache_for_hash(entry['hash'])
                print(f"✅ Cleared cache entry: {entry['hash']}")
                
        except Exception as e:
            print(f"❌ Failed to clear specific cache: {e}")
            
    elif os.path.exists(target):
        # Clear cache for specific path
        print(f"🗑️ Clearing cache for path: {target}")
        
        try:
            cache.clear_cache_for_path(target)
            print(f"✅ Cache cleared for: {target}")
        except Exception as e:
            print(f"❌ Failed to clear cache for path: {e}")
    else:
        print(f"❌ Invalid target: {target}")

def optimize_cache():
    """Optimize cache by removing old/unused entries"""
    print("⚡ Optimizing cache...")
    cache = LogCache()
    
    try:
        initial_stats = cache.get_cache_stats()
        print(f"📊 Initial cache size: {initial_stats['cache_size_mb']:.2f} MB")
        
        # Clean old cache entries (older than 7 days)
        cleaned = cache.cleanup_old_cache(max_age_days=7)
        
        final_stats = cache.get_cache_stats()
        saved_mb = initial_stats['cache_size_mb'] - final_stats['cache_size_mb']
        
        print(f"✅ Optimization complete!")
        print(f"🗑️ Removed {cleaned} old entries")
        print(f"💾 Freed {saved_mb:.2f} MB of disk space")
        print(f"📊 Final cache size: {final_stats['cache_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"❌ Cache optimization failed: {e}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Log Cache Management Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s check                    # Check all cache status
  %(prog)s check --path logs/       # Check specific path cache
  %(prog)s create --path logs/      # Create cache for path
  %(prog)s create --force           # Force recreate cache
  %(prog)s clear --target all       # Clear all cache
  %(prog)s clear --target hash:abc123  # Clear specific hash
  %(prog)s clear --target logs/     # Clear path cache
  %(prog)s optimize                 # Optimize cache
        """
    )
    
    subparsers = parser.add_subparsers(dest='action', help='Available actions')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check cache status')
    check_parser.add_argument('--path', help='Specific log path to check')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create cache')
    create_parser.add_argument('--path', help='Log path to cache')
    create_parser.add_argument('--force', action='store_true', 
                              help='Force recreation of existing cache')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear cache')
    clear_parser.add_argument('--target', default='all',
                             help='Target to clear: "all", "hash:PREFIX", or path')
    clear_parser.add_argument('--confirm', action='store_true',
                             help='Skip confirmation prompt')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize cache')
    
    args = parser.parse_args()
    
    if args.action == 'check':
        check_cache_status(args.path)
    elif args.action == 'create':
        create_cache(args.path, args.force)
    elif args.action == 'clear':
        clear_cache(args.target, args.confirm)
    elif args.action == 'optimize':
        optimize_cache()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 