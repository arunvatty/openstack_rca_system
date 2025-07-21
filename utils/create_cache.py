#!/usr/bin/env python3
"""
Utility to force creation of the log cache for a given log path.
Prints cache stats after creation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.log_cache import LogCache
from config.config import Config

def create_cache(log_path=None):
    if log_path is None:
        log_path = Config.DATA_DIR
    print(f"🔄 Forcing cache creation for: {log_path}")
    cache = LogCache()
    logs_df = cache.get_cached_logs(log_path, force_reload=True)
    print(f"✅ Cache created with {len(logs_df)} logs!")
    stats = cache.get_cache_stats()
    print("\n📊 Cache Stats:")
    print(f"  Directory: {stats['cache_dir']}")
    print(f"  Total Entries: {stats['total_entries']}")
    print(f"  Cache Size: {stats['cache_size_mb']:.2f} MB")
    if stats['oldest_entry']:
        print(f"  Oldest Entry: {stats['oldest_entry']}")
    if stats['newest_entry']:
        print(f"  Newest Entry: {stats['newest_entry']}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Force log cache creation utility')
    parser.add_argument('--log-path', type=str, default=None, help='Path to logs directory (default: logs/)')
    args = parser.parse_args()
    create_cache(args.log_path)

if __name__ == "__main__":
    main() 