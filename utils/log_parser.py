import re
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenStackLogParser:
    """Parser for OpenStack log files"""
    
    def __init__(self):
        self.log_pattern = re.compile(
            r'(?P<service>\S+)\s+'
            r'(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+'
            r'(?P<process_id>\d+)\s+'
            r'(?P<level>\w+)\s+'
            r'(?P<module>[\w\.]+)\s+'
            r'(?P<request_info>\[.*?\])\s+'
            r'(?P<message>.*)'
        )
        
        self.service_patterns = {
            'nova-api': r'nova-api',
            'nova-compute': r'nova-compute',
            'nova-scheduler': r'nova-scheduler'
        }
    
    def parse_log_line(self, line: str) -> Optional[Dict]:
        """Parse a single log line into structured data"""
        try:
            match = self.log_pattern.match(line.strip())
            if not match:
                return None
            
            log_data = match.groupdict()
            
            # Parse timestamp
            try:
                log_data['timestamp'] = datetime.strptime(
                    log_data['timestamp'], '%Y-%m-%d %H:%M:%S.%f'
                )
            except ValueError:
                log_data['timestamp'] = None
            
            # Extract service type
            log_data['service_type'] = self._extract_service_type(log_data['service'])
            
            # Extract instance ID if present
            log_data['instance_id'] = self._extract_instance_id(log_data['message'])
            
            # Extract request ID
            log_data['request_id'] = self._extract_request_id(log_data['request_info'])
            
            return log_data
            
        except Exception as e:
            logger.warning(f"Failed to parse log line: {str(e)}")
            return None
    
    def _extract_service_type(self, service: str) -> str:
        """Extract service type from service string"""
        for service_type, pattern in self.service_patterns.items():
            if re.search(pattern, service):
                return service_type
        return 'unknown'
    
    def _extract_instance_id(self, message: str) -> Optional[str]:
        """Extract instance ID from message"""
        instance_pattern = r'\[instance:\s+([a-f0-9-]+)\]'
        match = re.search(instance_pattern, message)
        return match.group(1) if match else None
    
    def _extract_request_id(self, request_info: str) -> Optional[str]:
        """Extract request ID from request info"""
        req_pattern = r'req-([a-f0-9-]+)'
        match = re.search(req_pattern, request_info)
        return match.group(1) if match else None
    
    def parse_log_file(self, file_path: str) -> pd.DataFrame:
        """Parse entire log file into DataFrame"""
        logger.info(f"Parsing log file: {file_path}")
        
        parsed_logs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                for line_num, line in enumerate(file, 1):
                    if line.strip():
                        parsed_data = self.parse_log_line(line)
                        if parsed_data:
                            parsed_data['line_number'] = line_num
                            parsed_data['raw_log'] = line.strip()
                            parsed_logs.append(parsed_data)
        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return pd.DataFrame()
        
        if not parsed_logs:
            logger.warning(f"No valid log entries found in {file_path}")
            return pd.DataFrame()
        
        df = pd.DataFrame(parsed_logs)
        logger.info(f"Parsed {len(df)} log entries from {file_path}")
        
        return df
    
    def parse_multiple_files(self, file_paths: List[str]) -> pd.DataFrame:
        """Parse multiple log files and combine into single DataFrame"""
        all_logs = []
        
        for file_path in file_paths:
            df = self.parse_log_file(file_path)
            if not df.empty:
                df['source_file'] = file_path
                all_logs.append(df)
        
        if not all_logs:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_logs, ignore_index=True)
        
        # Sort by timestamp
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Combined {len(combined_df)} log entries from {len(file_paths)} files")
        return combined_df