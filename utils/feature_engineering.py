import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for OpenStack logs"""
    
    def __init__(self):
        self.scalers = {}
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from timestamps"""
        logger.info("Creating temporal features...")
        
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found")
            return df
        
        df = df.copy()
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['minute'] = df['timestamp'].dt.minute
        df['second'] = df['timestamp'].dt.second
        
        # Time-based categorical features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Time since previous log entry
        df['time_since_prev'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
        
        # Rolling window features - fix deprecated time aliases
        df['logs_in_last_minute'] = df.groupby(df['timestamp'].dt.floor('1min')).cumcount() + 1
        df['logs_in_last_hour'] = df.groupby(df['timestamp'].dt.floor('1h')).cumcount() + 1
        
        logger.info("Temporal features created successfully")
        return df
    
    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from log message text"""
        logger.info("Creating text features...")
        
        if 'message' not in df.columns:
            logger.warning("No message column found")
            return df
        
        df = df.copy()
        
        # Basic text statistics
        df['message_length'] = df['message'].str.len().fillna(0)
        df['word_count'] = df['message'].str.split().str.len().fillna(0)
        df['char_count'] = df['message'].str.len().fillna(0)
        
        # Special character counts
        df['number_count'] = df['message'].str.count(r'\d').fillna(0)
        df['uppercase_count'] = df['message'].str.count(r'[A-Z]').fillna(0)
        df['special_char_count'] = df['message'].str.count(r'[^a-zA-Z0-9\s]').fillna(0)
        
        # URL and IP detection
        df['has_url'] = df['message'].str.contains(r'https?://', case=False, na=False).astype(int)
        df['has_ip'] = df['message'].str.contains(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', na=False).astype(int)
        
        # Error-related keywords
        error_keywords = ['error', 'failed', 'exception', 'timeout', 'denied', 'rejected']
        for keyword in error_keywords:
            df[f'has_{keyword}'] = df['message'].str.contains(keyword, case=False, na=False).astype(int)
        
        # OpenStack-specific keywords
        openstack_keywords = ['instance', 'server', 'image', 'network', 'volume', 'flavor']
        for keyword in openstack_keywords:
            # Use contains with case=False, then sum the boolean results
            df[f'mentions_{keyword}'] = df['message'].str.contains(keyword, case=False, na=False).astype(int)
        
        logger.info("Text features created successfully")
        return df
    
    def create_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on log sequence patterns"""
        logger.info("Creating sequence features...")
        
        df = df.copy()
        
        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Sequential features
        df['log_sequence_id'] = range(len(df))
        
        # Previous and next log level
        df['prev_level'] = df['level'].shift(1)
        df['next_level'] = df['level'].shift(-1)
        
        # Level transition features
        df['level_changed'] = (df['level'] != df['prev_level']).astype(int)
        df['escalated_to_error'] = ((df['prev_level'] == 'INFO') & (df['level'] == 'ERROR')).astype(int)
        
        # Service transition features
        if 'service_type' in df.columns:
            df['prev_service'] = df['service_type'].shift(1)
            df['service_changed'] = (df['service_type'] != df['prev_service']).astype(int)
        
        # Instance continuity features
        if 'instance_id' in df.columns:
            df['prev_instance'] = df['instance_id'].shift(1)
            df['same_instance'] = (df['instance_id'] == df['prev_instance']).astype(int)
        
        # Rolling statistics
        window_sizes = [5, 10, 20]
        for window in window_sizes:
            df[f'error_rate_last_{window}'] = (
                df['level'].eq('ERROR').rolling(window=window, min_periods=1).mean()
            )
            df[f'message_length_avg_last_{window}'] = (
                df['message_length'].rolling(window=window, min_periods=1).mean()
            )
        
        logger.info("Sequence features created successfully")
        return df
    
    def create_instance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to instance behavior"""
        logger.info("Creating instance features...")
        
        if 'instance_id' not in df.columns:
            logger.warning("No instance_id column found")
            return df
        
        df = df.copy()
        
        # Instance-level aggregations
        instance_stats = df.groupby('instance_id').agg({
            'level': lambda x: (x == 'ERROR').sum(),  # Error count per instance
            'message': 'count',  # Total logs per instance
            'timestamp': ['min', 'max'] if 'timestamp' in df.columns else 'count'
        }).fillna(0)
        
        if 'timestamp' in df.columns:
            instance_stats.columns = ['instance_error_count', 'instance_log_count', 'instance_first_seen', 'instance_last_seen']
            # Calculate instance lifespan
            instance_stats['instance_lifespan'] = (
                instance_stats['instance_last_seen'] - instance_stats['instance_first_seen']
            ).dt.total_seconds().fillna(0)
        else:
            instance_stats.columns = ['instance_error_count', 'instance_log_count']
        
        # Instance error rate
        instance_stats['instance_error_rate'] = (
            instance_stats['instance_error_count'] / instance_stats['instance_log_count']
        ).fillna(0)
        
        # Merge back to main dataframe
        df = df.merge(instance_stats, left_on='instance_id', right_index=True, how='left')
        
        # Instance activity patterns
        df['is_new_instance'] = df.groupby('instance_id').cumcount() == 0
        df['instance_log_position'] = df.groupby('instance_id').cumcount() + 1
        
        logger.info("Instance features created successfully")
        return df
    
    def create_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for anomaly detection"""
        logger.info("Creating anomaly features...")
        
        df = df.copy()
        
        # Message length anomalies
        if 'message_length' in df.columns:
            msg_length_mean = df['message_length'].mean()
            msg_length_std = df['message_length'].std()
            df['message_length_zscore'] = abs((df['message_length'] - msg_length_mean) / msg_length_std)
            df['is_long_message'] = (df['message_length'] > msg_length_mean + 2 * msg_length_std).astype(int)
        
        # Time-based anomalies
        if 'time_since_prev' in df.columns:
            time_diff_mean = df['time_since_prev'].mean()
            time_diff_std = df['time_since_prev'].std()
            df['time_diff_zscore'] = abs((df['time_since_prev'] - time_diff_mean) / time_diff_std)
            df['is_time_gap'] = (df['time_since_prev'] > time_diff_mean + 3 * time_diff_std).astype(int)
        
        # Frequency-based anomalies
        if 'timestamp' in df.columns:
            # Logs per minute
            df['minute_bucket'] = df['timestamp'].dt.floor('1min')
            minute_counts = df.groupby('minute_bucket').size()
            df['logs_per_minute'] = df['minute_bucket'].map(minute_counts)
            
            logs_per_minute_mean = df['logs_per_minute'].mean()
            logs_per_minute_std = df['logs_per_minute'].std()
            df['is_high_frequency'] = (
                df['logs_per_minute'] > logs_per_minute_mean + 2 * logs_per_minute_std
            ).astype(int)
        
        # Error burst detection
        if 'level' in df.columns:
            df['is_error'] = (df['level'] == 'ERROR').astype(int)
            df['error_burst'] = df['is_error'].rolling(window=10, min_periods=1).sum()
            df['is_error_burst'] = (df['error_burst'] >= 5).astype(int)
        
        logger.info("Anomaly features created successfully")
        return df
    
    def create_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create contextual features based on surrounding logs"""
        logger.info("Creating context features...")
        
        df = df.copy()
        
        # Context window features
        context_windows = [3, 5, 10]
        
        for window in context_windows:
            # Error context
            df[f'errors_in_context_{window}'] = (
                df['level'].eq('ERROR').rolling(window=window, center=True, min_periods=1).sum()
            )
            
            # Service diversity in context - fix for string columns
            if 'service_type' in df.columns:
                # Convert to categorical codes for rolling operations
                service_codes = pd.Categorical(df['service_type']).codes
                service_series = pd.Series(service_codes, index=df.index)
                
                df[f'services_in_context_{window}'] = (
                    service_series.rolling(window=window, center=True, min_periods=1)
                    .apply(lambda x: len(set(x[x >= 0])), raw=True)  # Exclude -1 (NaN codes)
                )
            
            # Instance diversity in context - fix for string columns
            if 'instance_id' in df.columns:
                # Create a safe version for rolling operations
                instance_filled = df['instance_id'].fillna('__missing__')
                instance_codes = pd.Categorical(instance_filled).codes
                instance_series = pd.Series(instance_codes, index=df.index)
                
                df[f'instances_in_context_{window}'] = (
                    instance_series.rolling(window=window, center=True, min_periods=1)
                    .apply(lambda x: len(set(x)), raw=True)
                )
        
        # Request context features
        if 'request_id' in df.columns:
            # Only process non-null request IDs
            request_mask = df['request_id'].notna()
            if request_mask.any():
                request_stats = df[request_mask].groupby('request_id').agg({
                    'level': lambda x: (x == 'ERROR').sum(),
                    'message': 'count'
                }).fillna(0)
                request_stats.columns = ['request_error_count', 'request_log_count']
                request_stats['request_error_rate'] = (
                    request_stats['request_error_count'] / request_stats['request_log_count']
                ).fillna(0)
                
                df = df.merge(request_stats, left_on='request_id', right_index=True, how='left')
                
                # Fill NaN values for logs without request IDs
                df['request_error_count'] = df['request_error_count'].fillna(0)
                df['request_log_count'] = df['request_log_count'].fillna(1)
                df['request_error_rate'] = df['request_error_rate'].fillna(0)
        
        logger.info("Context features created successfully")
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        logger.info("Starting comprehensive feature engineering...")
        
        if df.empty:
            logger.warning("Empty dataframe provided")
            return df
        
        # Apply all feature engineering steps
        df = self.create_temporal_features(df)
        df = self.create_text_features(df)
        df = self.create_sequence_features(df)
        df = self.create_instance_features(df)
        df = self.create_anomaly_features(df)
        df = self.create_context_features(df)
        
        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        return df
    
    def select_features_for_lstm(self, df: pd.DataFrame) -> List[str]:
        """Select optimal features for LSTM model"""
        lstm_features = [
            # Text features
            'message_length', 'word_count', 'number_count', 'uppercase_count',
            
            # Temporal features
            'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
            'time_since_prev', 'logs_in_last_minute',
            
            # Error-related features
            'has_error', 'has_failed', 'has_exception', 'has_timeout',
            
            # OpenStack features
            'mentions_instance', 'mentions_server', 'mentions_network',
            
            # Sequence features
            'level_changed', 'escalated_to_error', 'error_rate_last_5',
            
            # Anomaly features
            'is_long_message', 'is_time_gap', 'is_high_frequency', 'is_error_burst',
            
            # Context features
            'errors_in_context_5', 'services_in_context_5'
        ]
        
        # Filter features that exist in the dataframe
        available_features = [f for f in lstm_features if f in df.columns]
        
        logger.info(f"Selected {len(available_features)} features for LSTM model")
        return available_features
    
    def scale_features(self, df: pd.DataFrame, feature_columns: List[str], 
                      scaler_type: str = 'standard') -> pd.DataFrame:
        """Scale numerical features"""
        logger.info(f"Scaling features using {scaler_type} scaler...")
        
        df = df.copy()
        
        if scaler_type not in self.scalers:
            if scaler_type == 'standard':
                self.scalers[scaler_type] = StandardScaler()
            elif scaler_type == 'minmax':
                self.scalers[scaler_type] = MinMaxScaler()
            else:
                raise ValueError(f"Unsupported scaler type: {scaler_type}")
        
        # Filter existing columns
        existing_features = [f for f in feature_columns if f in df.columns]
        
        if existing_features:
            scaled_values = self.scalers[scaler_type].fit_transform(df[existing_features])
            df[existing_features] = scaled_values
            
            logger.info(f"Scaled {len(existing_features)} features")
        
        return df
    
    def create_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Create summary of engineered features"""
        summary = {
            'total_features': len(df.columns),
            'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_features': len(df.select_dtypes(include=['datetime64']).columns),
            'missing_values': df.isnull().sum().sum(),
            'feature_categories': {
                'temporal': len([c for c in df.columns if any(t in c for t in ['hour', 'day', 'time', 'minute'])]),
                'text': len([c for c in df.columns if any(t in c for t in ['message', 'word', 'char', 'length'])]),
                'instance': len([c for c in df.columns if 'instance' in c]),
                'error': len([c for c in df.columns if any(t in c for t in ['error', 'failed', 'exception'])]),
                'context': len([c for c in df.columns if 'context' in c]),
                'anomaly': len([c for c in df.columns if any(t in c for t in ['anomaly', 'burst', 'gap', 'zscore'])])
            }
        }
        
        return summary