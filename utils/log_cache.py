import os
import pandas as pd
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import pickle

from data.log_ingestion import LogIngestionManager
from utils.feature_engineering import FeatureEngineer
from config.config import Config

logger = logging.getLogger(__name__)

class LogCache:
    """Cache for processed log data to avoid repeated file loading"""
    
    def __init__(self, cache_dir: str = None, max_cache_age_hours: int = 24):
        # Use config cache directory if not specified
        if cache_dir is None:
            cache_dir = Config.CACHE_DIR
            
        self.cache_dir = Path(cache_dir)
        self.max_cache_age = timedelta(hours=max_cache_age_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata
        self.cache_metadata = {}
        self.metadata_file = self.cache_dir / 'cache_metadata.pkl'
        self._load_metadata()
        
        logger.info(f"LogCache initialized with cache directory: {self.cache_dir}")
    
    def _load_metadata(self):
        """Load cache metadata from disk"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'rb') as f:
                    self.cache_metadata = pickle.load(f)
                logger.info(f"Loaded cache metadata: {len(self.cache_metadata)} entries")
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            self.cache_metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata to disk"""
        try:
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.cache_metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, log_files_path: str) -> str:
        """Generate cache key based on log files path"""
        # Create hash of the path
        return hashlib.md5(log_files_path.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path for a given key"""
        return self.cache_dir / f"logs_{cache_key}.pkl"
    
    def _get_files_hash(self, log_files_path: str) -> str:
        """Calculate hash of log files to detect changes"""
        try:
            log_dir = Path(log_files_path)
            if not log_dir.exists():
                return ""
            
            # Get all log files
            log_files = []
            for ext in ['.log', '.txt']:
                log_files.extend(log_dir.glob(f'**/*{ext}'))
            
            # Calculate combined hash of file sizes and modification times
            file_info = []
            for log_file in sorted(log_files):
                stat = log_file.stat()
                file_info.append(f"{log_file}:{stat.st_size}:{stat.st_mtime}")
            
            combined_info = "|".join(file_info)
            return hashlib.md5(combined_info.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Failed to calculate files hash: {e}")
            return ""
    
    def get_cached_logs(self, log_files_path: str, force_reload: bool = False) -> Optional[pd.DataFrame]:
        """Get cached logs or load from files if cache is invalid"""
        cache_key = self._get_cache_key(log_files_path)
        cache_file = self._get_cache_file(cache_key)
        
        # Check if cache exists and is valid
        if not force_reload and self._is_cache_valid(cache_key, log_files_path):
            try:
                logger.info(f"Loading cached logs from {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                logger.info(f"✓ Loaded {len(cached_data['logs_df'])} cached logs")
                return cached_data['logs_df']
                
            except Exception as e:
                logger.warning(f"Failed to load cached logs: {e}")
        
        # Cache is invalid or doesn't exist, load from files
        logger.info(f"Loading logs from files: {log_files_path}")
        logs_df = self._load_and_cache_logs(log_files_path, cache_key)
        
        return logs_df
    
    def _is_cache_valid(self, cache_key: str, log_files_path: str) -> bool:
        """Check if cache is valid (exists and not expired)"""
        if cache_key not in self.cache_metadata:
            return False
        
        cache_info = self.cache_metadata[cache_key]
        cache_file = self._get_cache_file(cache_key)
        
        # Check if cache file exists
        if not cache_file.exists():
            logger.info("Cache file does not exist")
            return False
        
        # Check cache age
        cache_age = datetime.now() - cache_info['created_at']
        if cache_age > self.max_cache_age:
            logger.info(f"Cache expired (age: {cache_age})")
            return False
        
        # Check if files have changed
        current_files_hash = self._get_files_hash(log_files_path)
        if current_files_hash != cache_info['files_hash']:
            logger.info("Log files have changed, cache invalid")
            return False
        
        logger.info("Cache is valid")
        return True
    
    def _load_and_cache_logs(self, log_files_path: str, cache_key: str) -> pd.DataFrame:
        """Load logs from files and cache them"""
        try:
            # Load logs using ingestion manager
            ingestion_manager = LogIngestionManager()
            
            if os.path.isdir(log_files_path):
                logs_df = ingestion_manager.ingest_from_directory(log_files_path)
            else:
                logs_df = ingestion_manager.ingest_multiple_files([log_files_path])
            
            if logs_df.empty:
                logger.warning("No logs loaded from files")
                return pd.DataFrame()
            
            # Apply feature engineering
            feature_engineer = FeatureEngineer()
            logs_df = feature_engineer.engineer_all_features(logs_df)
            
            # Cache the processed logs
            self._save_to_cache(cache_key, logs_df, log_files_path)
            
            logger.info(f"✓ Loaded and cached {len(logs_df)} logs")
            return logs_df
            
        except Exception as e:
            logger.error(f"Failed to load logs from files: {e}")
            return pd.DataFrame()
    
    def _save_to_cache(self, cache_key: str, logs_df: pd.DataFrame, log_files_path: str):
        """Save logs to cache"""
        try:
            cache_file = self._get_cache_file(cache_key)
            
            # Prepare cache data
            cache_data = {
                'logs_df': logs_df,
                'created_at': datetime.now(),
                'log_files_path': log_files_path,
                'log_count': len(logs_df)
            }
            
            # Save to disk
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Update metadata
            self.cache_metadata[cache_key] = {
                'created_at': datetime.now(),
                'files_hash': self._get_files_hash(log_files_path),
                'log_count': len(logs_df),
                'cache_file': str(cache_file)
            }
            
            self._save_metadata()
            
            logger.info(f"✓ Cached {len(logs_df)} logs to {cache_file}")
            
        except Exception as e:
            logger.error(f"Failed to save logs to cache: {e}")
    
    def clear_cache(self, cache_key: str = None):
        """Clear cache entries"""
        try:
            if cache_key:
                # Clear specific cache
                cache_file = self._get_cache_file(cache_key)
                if cache_file.exists():
                    cache_file.unlink()
                
                if cache_key in self.cache_metadata:
                    del self.cache_metadata[cache_key]
                
                logger.info(f"Cleared cache for key: {cache_key}")
            else:
                # Clear all cache
                for cache_file in self.cache_dir.glob('logs_*.pkl'):
                    cache_file.unlink()
                
                self.cache_metadata = {}
                logger.info("Cleared all cache")
            
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'cache_dir': str(self.cache_dir),
            'total_entries': len(self.cache_metadata),
            'cache_size_mb': 0,
            'oldest_entry': None,
            'newest_entry': None
        }
        
        try:
            # Calculate cache size
            total_size = 0
            for cache_file in self.cache_dir.glob('logs_*.pkl'):
                total_size += cache_file.stat().st_size
            
            stats['cache_size_mb'] = total_size / (1024 * 1024)
            
            # Find oldest and newest entries
            if self.cache_metadata:
                created_times = [info['created_at'] for info in self.cache_metadata.values()]
                stats['oldest_entry'] = min(created_times)
                stats['newest_entry'] = max(created_times)
            
        except Exception as e:
            logger.warning(f"Failed to calculate cache stats: {e}")
        
        return stats
    
    def cleanup_expired_cache(self):
        """Remove expired cache entries"""
        try:
            expired_keys = []
            current_time = datetime.now()
            
            for cache_key, cache_info in self.cache_metadata.items():
                cache_age = current_time - cache_info['created_at']
                if cache_age > self.max_cache_age:
                    expired_keys.append(cache_key)
            
            for cache_key in expired_keys:
                self.clear_cache(cache_key)
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            else:
                logger.info("No expired cache entries found")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
    
    def get_cached_logs_with_metadata(self, log_files_path: str) -> Dict[str, Any]:
        """Get cached logs with metadata"""
        cache_key = self._get_cache_key(log_files_path)
        
        result = {
            'logs_df': None,
            'cache_info': None,
            'loaded_from_cache': False
        }
        
        if self._is_cache_valid(cache_key, log_files_path):
            try:
                cache_file = self._get_cache_file(cache_key)
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                result['logs_df'] = cached_data['logs_df']
                result['cache_info'] = self.cache_metadata[cache_key]
                result['loaded_from_cache'] = True
                
                logger.info(f"✓ Loaded {len(cached_data['logs_df'])} logs from cache")
                
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        if result['logs_df'] is None:
            # Load from files
            result['logs_df'] = self._load_and_cache_logs(log_files_path, cache_key)
            result['cache_info'] = self.cache_metadata.get(cache_key)
            result['loaded_from_cache'] = False
        
        return result 