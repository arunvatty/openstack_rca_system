# Cache Operations Manual

Complete guide for log caching system, cache management, and performance optimization in the OpenStack RCA system.

## 🚀 Overview

The cache system provides high-performance log processing by storing preprocessed log data, eliminating the need for repeated file parsing and feature engineering.

### Key Benefits
- **50x faster data loading** (0.1s vs 5-10s)
- **Consistent feature engineering** across sessions
- **Memory efficiency** with pickle serialization
- **Automatic invalidation** on file changes
- **Metadata tracking** for cache management

## 🗄️ Cache Architecture

```
data/cache/
├── cache_metadata.pkl          # Cache index and metadata
├── logs_<hash1>.pkl           # Cached log data (dataset 1)
├── logs_<hash2>.pkl           # Cached log data (dataset 2)
└── ...
```

### Cache Components
- **Metadata File**: Index of all cached datasets with hashes and timestamps
- **Data Files**: Serialized pandas DataFrames with processed logs
- **Hash System**: MD5 hashes based on log directory and file modification times

## 🔧 Cache Management

### 1. Check Cache Status
```bash
# Check all cache status
python3 utils/cache_manager.py check

# Check specific path cache
python3 utils/cache_manager.py check --path logs/

# Output example:
# 🔍 Checking Log Cache Status
# ==================================================
# 📂 Cache directory: data/cache
# 📊 Total cached datasets: 2
# 💾 Cache size: 4.20 MB
# 
# 📋 Cache files:
#   05d4be440b84...pkl (568 logs, 2.1 MB)
#   2165e4fa5bdd...pkl (568 logs, 2.1 MB)
```

### 2. Create/Update Cache
```bash
# Create cache for specific directory
python3 utils/cache_manager.py create --path logs/

# Force recreate existing cache
python3 utils/cache_manager.py create --path logs/ --force

# Create cache with VectorDB integration
python3 main.py --mode train  # Automatically creates cache
```

### 3. Clear Cache
```bash
# Clear all cache files
python3 utils/cache_manager.py clear --target all

# Clear specific cache entry by hash prefix
python3 utils/cache_manager.py clear --target hash:05d4be44

# Clear cache for specific path
python3 utils/cache_manager.py clear --target logs/

# Skip confirmation prompt
python3 utils/cache_manager.py clear --target all --confirm
```

## 🔍 Cache Usage in Code

### Automatic Cache Usage
```python
from utils.log_cache import LogCache

# Initialize cache system
cache = LogCache()

# Load logs (uses cache if available)
df = cache.get_cached_logs('logs/')
# If cache exists and valid → loads in ~0.1s
# If cache invalid → processes files and caches → ~5-10s
```

### Manual Cache Control
```python
# Check if cache exists and is valid
cache_exists = cache.is_cache_valid('logs/')
print(f"Cache valid: {cache_exists}")

# Force cache refresh
cache.clear_cache_for_path('logs/')
df = cache.get_cached_logs('logs/')  # Will reprocess

# Get cache statistics
stats = cache.get_cache_stats()
print(f"Cache size: {stats['cache_size_mb']:.2f} MB")
print(f"Total datasets: {stats['total_datasets']}")
```

## 📊 Cache Performance

### Performance Comparison

| Operation | Without Cache | With Cache | Improvement |
|-----------|---------------|------------|-------------|
| **Data Loading** | 5-10 seconds | 0.1 seconds | **50-100x faster** |
| **Feature Engineering** | 2-3 seconds | Included | **Instant** |
| **Memory Usage** | Variable | Optimized | **Consistent** |
| **Disk I/O** | High | Minimal | **90% reduction** |

### Cache Hit Rate
- **First Run**: Cache miss (processes files)
- **Subsequent Runs**: Cache hit (loads instantly)
- **File Changes**: Automatic cache invalidation
- **Memory Footprint**: ~2-4 MB per 1000 log entries

## 🛠️ Cache Configuration

### Cache Settings (config/config.py)
```python
CACHE_CONFIG = {
    'cache_dir': 'data/cache',
    'max_cache_age_hours': 24,
    'max_cache_size_gb': 5,
    'enable_compression': True,
    'cache_file_prefix': 'logs_'
}
```

### Environment Variables
```bash
# Override cache directory
export RCA_CACHE_DIR="/custom/cache/path"

# Disable caching
export RCA_DISABLE_CACHE=true

# Set cache expiry (hours)
export RCA_CACHE_MAX_AGE=48
```

## 🔄 Cache Lifecycle

### 1. Cache Creation Process
```python
def create_cache(log_directory):
    # 1. Calculate directory hash
    dir_hash = calculate_directory_hash(log_directory)
    
    # 2. Process log files
    df = ingest_and_process_logs(log_directory)
    
    # 3. Apply feature engineering
    df = apply_feature_engineering(df)
    
    # 4. Store in cache
    cache_file = f"data/cache/logs_{dir_hash}.pkl"
    df.to_pickle(cache_file)
    
    # 5. Update metadata
    update_cache_metadata(dir_hash, log_directory, len(df))
```

### 2. Cache Validation
```python
def validate_cache(log_directory):
    # 1. Calculate current directory hash
    current_hash = calculate_directory_hash(log_directory)
    
    # 2. Check if cache file exists
    cache_file = f"data/cache/logs_{current_hash}.pkl"
    if not exists(cache_file):
        return False
    
    # 3. Check cache age
    cache_age = get_file_age(cache_file)
    if cache_age > MAX_CACHE_AGE:
        return False
    
    # 4. Validate metadata
    return validate_cache_metadata(current_hash)
```

### 3. Cache Loading
```python
def load_from_cache(log_directory):
    # 1. Calculate directory hash
    dir_hash = calculate_directory_hash(log_directory)
    
    # 2. Load cached DataFrame
    cache_file = f"data/cache/logs_{dir_hash}.pkl"
    df = pd.read_pickle(cache_file)
    
    # 3. Validate loaded data
    validate_dataframe_structure(df)
    
    return df
```

## 📈 Cache Monitoring

### Cache Statistics
```bash
# View cache statistics
python3 -c "
from utils.log_cache import LogCache
cache = LogCache()
stats = cache.get_cache_stats()
print(f'Cache Directory: {stats[\"cache_dir\"]}')
print(f'Total Datasets: {stats[\"total_datasets\"]}')
print(f'Cache Size: {stats[\"cache_size_mb\"]:.2f} MB')
print(f'Available Space: {stats[\"available_space_gb\"]:.2f} GB')
"
```

### Cache Health Check
```bash
# Check cache health and detailed status
python3 utils/cache_manager.py check

# Sample output:
# 🔍 Checking Log Cache Status
# ==================================================
# ✅ Cache directory exists
# ✅ Metadata file is valid
# ✅ All cache files accessible
# 📊 Total cached datasets: 2
# 💾 Cache size: 4.20 MB
# ✅ No corrupted cache entries
```

## 🧹 Cache Maintenance

### Automatic Cleanup
The cache system automatically handles:
- **Stale cache removal**: Removes caches older than 24 hours
- **Size management**: Prevents cache from exceeding 5 GB
- **Corruption detection**: Validates cache integrity

### Manual Maintenance
```bash
# Optimize cache (clean old entries, compact)
python3 utils/cache_manager.py optimize

# Clear all cache files
python3 utils/cache_manager.py clear --target all

# Recreate cache from scratch
python3 utils/cache_manager.py create --path logs/ --force
```

### Cache Repair
```bash
# Check for issues and status
python3 utils/cache_manager.py check

# Force rebuild all caches
python3 utils/cache_manager.py clear --target all --confirm
python3 utils/cache_manager.py create --path logs/
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Cache Not Loading
```bash
# Check cache validity
python3 -c "
from utils.log_cache import LogCache
cache = LogCache()
valid = cache.is_cache_valid('logs/')
print(f'Cache valid: {valid}')
"

# Solution: Clear and recreate
python3 utils/cache_manager.py clear --target all
python3 main.py --mode train
```

#### 2. Cache Size Issues
```bash
# Check disk space
df -h data/cache/

# Clean all cache files
python3 utils/cache_manager.py clear --target all
```

#### 3. Permission Issues
```bash
# Fix permissions
sudo chown -R $USER:$USER data/cache/
chmod -R 755 data/cache/
```

#### 4. Corrupted Cache
```bash
# Check cache status
python3 utils/cache_manager.py check

# Repair or rebuild
python3 utils/cache_manager.py clear --target all
python3 main.py --mode train  # Rebuilds cache
```

### Performance Issues

#### 1. Slow Cache Loading
- **Cause**: Large cache files on slow storage
- **Solution**: Move cache to SSD or increase memory

#### 2. High Memory Usage
- **Cause**: Multiple large datasets in memory
- **Solution**: Process datasets sequentially

#### 3. Cache Misses
- **Cause**: Frequent file changes or incorrect hashing
- **Solution**: Check file stability and hash algorithm

## ⚡ Performance Tuning

### Optimize Cache Settings
```python
# config/config.py
CACHE_CONFIG = {
    'cache_dir': '/fast/ssd/cache',      # Use SSD storage
    'max_cache_age_hours': 48,           # Longer cache life
    'enable_compression': True,          # Reduce file size
    'pickle_protocol': 4,               # Latest pickle protocol
}
```

### Memory Optimization
```python
# Load only required columns
cache = LogCache()
df = cache.get_cached_logs('logs/', columns=['timestamp', 'level', 'message'])

# Use chunked processing for large datasets
for chunk in cache.get_cached_logs_chunked('logs/', chunk_size=1000):
    process_chunk(chunk)
```

### Disk Optimization
```bash
# Use tmpfs for ultra-fast cache (RAM disk)
sudo mkdir /tmp/cache
sudo mount -t tmpfs -o size=2G tmpfs /tmp/cache
export RCA_CACHE_DIR="/tmp/cache"
```

## 🔍 Cache Utilities

### Available Utilities

| Script | Purpose | Usage |
|--------|---------|-------|
| `cache_manager.py` | **Consolidated cache management** | `python3 utils/cache_manager.py <action>` |
| `log_cache.py` | Core cache library | Import in Python code |

### Cache Manager Actions

| Action | Purpose | Usage Example |
|--------|---------|---------------|
| `check` | Inspect cache status | `python3 utils/cache_manager.py check` |
| `create` | Create/update cache | `python3 utils/cache_manager.py create --path logs/` |
| `clear` | Clean cache files | `python3 utils/cache_manager.py clear --target all` |
| `optimize` | Clean old entries & optimize | `python3 utils/cache_manager.py optimize` |

### Complete Usage Examples
```bash
# Get help and see all options
python3 utils/cache_manager.py --help

# Check cache status for specific path
python3 utils/cache_manager.py check --path logs/

# Create cache with force rebuild
python3 utils/cache_manager.py create --path logs/ --force

# Clear cache by hash prefix
python3 utils/cache_manager.py clear --target hash:05d4be44

# Clear cache for specific path
python3 utils/cache_manager.py clear --target logs/

# Optimize cache (removes old entries automatically)
python3 utils/cache_manager.py optimize
```

### Advanced Usage

#### Batch Cache Operations
```bash
# Cache multiple directories
for dir in logs1/ logs2/ logs3/; do
    python3 utils/cache_manager.py create --path $dir
done

# Check status of multiple paths
for dir in logs*/; do
    echo "=== $dir ==="
    python3 utils/cache_manager.py check --path $dir
done
```

#### Cache Analysis
```python
from utils.log_cache import LogCache
import pandas as pd

cache = LogCache()
metadata = cache.load_cache_metadata()

# Analyze cache usage patterns
cache_usage = []
for entry in metadata:
    cache_usage.append({
        'hash': entry['hash'],
        'logs_count': entry['logs_count'],
        'file_size_mb': entry['file_size'] / (1024*1024),
        'age_hours': entry['age_hours']
    })

df = pd.DataFrame(cache_usage)
print(df.describe())
```

## 🏷️ Cache Best Practices

### 1. Development
- Use cache for consistent test data
- Clear cache when changing feature engineering
- Monitor cache size in CI/CD

### 2. Production
- Place cache on fast storage (SSD)
- Monitor cache hit rates
- Set appropriate cache expiry
- Implement cache warming strategies

### 3. Debugging
- Disable cache to test raw processing
- Compare cached vs non-cached results
- Use cache metadata for troubleshooting

The cache system is a critical component for high-performance log analysis! 🚀 