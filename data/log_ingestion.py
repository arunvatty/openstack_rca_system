import os
import pandas as pd
from typing import List, Dict
import logging
from pathlib import Path
from utils.log_parser import OpenStackLogParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogIngestionManager:
    """Manager for ingesting OpenStack log files"""
    
    def __init__(self, data_dir: str = 'logs'):
        self.data_dir = Path(data_dir)
        self.parser = OpenStackLogParser()
        self.supported_extensions = ['.log', '.txt']
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)
    
    def discover_log_files(self, directory: str = None) -> List[Path]:
        """Discover log files in the specified directory"""
        search_dir = Path(directory) if directory else self.data_dir
        
        log_files = []
        for ext in self.supported_extensions:
            log_files.extend(search_dir.glob(f'**/*{ext}'))
        
        logger.info(f"Found {len(log_files)} log files in {search_dir}")
        return log_files
    
    def ingest_single_file(self, file_path: str) -> pd.DataFrame:
        """Ingest a single log file"""
        logger.info(f"Ingesting log file: {file_path}")
        
        try:
            df = self.parser.parse_log_file(file_path)
            if not df.empty:
                # Add metadata
                df['ingestion_time'] = pd.Timestamp.now()
                df['file_size'] = os.path.getsize(file_path)
                logger.info(f"Successfully ingested {len(df)} log entries from {file_path}")
            else:
                logger.warning(f"No data extracted from {file_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def ingest_multiple_files(self, file_paths: List[str] = None) -> pd.DataFrame:
        """Ingest multiple log files"""
        if file_paths is None:
            file_paths = [str(f) for f in self.discover_log_files()]
        
        if not file_paths:
            logger.warning("No log files found to ingest")
            return pd.DataFrame()
        
        logger.info(f"Ingesting {len(file_paths)} log files...")
        
        all_dfs = []
        for file_path in file_paths:
            df = self.ingest_single_file(file_path)
            if not df.empty:
                all_dfs.append(df)
        
        if not all_dfs:
            logger.warning("No data was successfully ingested")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Sort by timestamp if available
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Successfully ingested {len(combined_df)} total log entries")
        return combined_df
    
    def ingest_from_directory(self, directory: str) -> pd.DataFrame:
        """Ingest all log files from a directory"""
        file_paths = [str(f) for f in self.discover_log_files(directory)]
        return self.ingest_multiple_files(file_paths)
    
    def save_ingested_data(self, df: pd.DataFrame, output_path: str):
        """Save ingested data to file"""
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as parquet for efficiency
            if output_path.endswith('.parquet'):
                df.to_parquet(output_path, index=False)
            elif output_path.endswith('.csv'):
                df.to_csv(output_path, index=False)
            else:
                # Default to parquet
                output_path += '.parquet'
                df.to_parquet(output_path, index=False)
            
            logger.info(f"Saved ingested data to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving data to {output_path}: {str(e)}")
    
    def load_saved_data(self, file_path: str) -> pd.DataFrame:
        """Load previously saved ingested data"""
        try:
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Loaded {len(df)} log entries from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def get_ingestion_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics about ingested data"""
        if df.empty:
            return {}
        
        stats = {
            'total_entries': len(df),
            'date_range': {
                'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max() if 'timestamp' in df.columns else None
            },
            'services': df['service_type'].value_counts().to_dict() if 'service_type' in df.columns else {},
            'log_levels': df['level'].value_counts().to_dict() if 'level' in df.columns else {},
            'source_files': df['source_file'].nunique() if 'source_file' in df.columns else 0,
            'unique_instances': df['instance_id'].nunique() if 'instance_id' in df.columns else 0
        }
        
        return stats