import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .ai_client import ClaudeClient
from config.config import Config

# NEW: Import vector database service
try:
    from services.vector_db_service import VectorDBService
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    logging.warning("VectorDBService not available - falling back to TF-IDF only")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RCAAnalyzer:
    """Root Cause Analysis engine using LSTM predictions, Vector DB, and Claude API"""
    
    def __init__(self, anthropic_api_key: str, lstm_model=None):
        self.client = ClaudeClient(anthropic_api_key)
        self.lstm_model = lstm_model
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # NEW: Initialize vector database service
        if VECTOR_DB_AVAILABLE:
            try:
                self.vector_db = VectorDBService()
                logger.info("VectorDBService initialized for RCA analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize VectorDBService: {e}")
        
        # Common OpenStack issue patterns
        self.issue_patterns = {
            'disk_space': [
                'disk space', 'storage', 'disk allocation', 'disk limit',
                'no space', 'disk full', 'insufficient disk'
            ],
            'resource_shortage': [
                'insufficient resources', 'no valid host', 'resource claim',
                'memory limit', 'vcpu limit', 'resource unavailable'
            ],
            'network_issues': [
                'network', 'connection', 'timeout', 'unreachable',
                'network-vif', 'port', 'connectivity'
            ],
            'instance_lifecycle': [
                'spawning', 'terminating', 'building', 'deleting',
                'instance destroyed', 'vm started', 'vm stopped'
            ],
            'authentication': [
                'unauthorized', 'forbidden', 'denied', 'authentication',
                'token', 'credentials'
            ]
        }
    
    def analyze_issue(self, issue_description: str, logs_df: pd.DataFrame, fast_mode: bool = False) -> Dict:
        """Analyze an issue and provide root cause analysis"""
        logger.info(f"Analyzing issue: {issue_description}")
        
        # Step 1: Filter relevant logs using LSTM predictions and Vector DB
        if fast_mode:
            logger.info("Using fast mode: skipping vector DB for speed")
            relevant_logs = self._filter_relevant_logs_fast(logs_df, issue_description)
        else:
            relevant_logs = self._filter_relevant_logs(logs_df, issue_description)
        
        # Step 2: Identify issue category
        issue_category = self._categorize_issue(issue_description)
        
        # Step 3: Extract key events and timeline
        timeline = self._extract_timeline(relevant_logs)
        
        # Step 4: Analyze patterns
        patterns = self._analyze_patterns(relevant_logs, issue_category)
        
        # Step 5: Generate RCA using Claude API with enhanced context
        rca_analysis = self._generate_rca_with_claude(
            issue_description, relevant_logs, timeline, patterns, issue_category
        )
        
        return {
            'issue_description': issue_description,
            'issue_category': issue_category,
            'relevant_logs_count': len(relevant_logs),
            'timeline': timeline,
            'patterns': patterns,
            'root_cause_analysis': rca_analysis,
            'recommendations': self._generate_recommendations(issue_category, patterns),
            'analysis_mode': 'fast' if fast_mode else 'full'
        }
    
    def _filter_relevant_logs(self, logs_df: pd.DataFrame, issue_description: str) -> pd.DataFrame:
        """Filter logs relevant to the issue using LSTM model, Vector DB, and similarity"""
        if logs_df.empty:
            return logs_df
        
        # Step 1: LSTM Filtering (existing approach)
        if self.lstm_model is not None:
            try:
                # Prepare data for LSTM prediction
                from data.preprocessing import LogPreprocessor
                preprocessor = LogPreprocessor()
                X, _ = preprocessor.prepare_lstm_data(logs_df)
                
                # Get importance predictions
                importance_scores = self.lstm_model.predict(X)
                
                # Filter important logs (top 70% by importance)
                threshold = np.percentile(importance_scores, 30)
                important_mask = importance_scores >= threshold
                
                filtered_logs = logs_df[important_mask].copy()
                logger.info(f"LSTM filtered {len(filtered_logs)} important logs from {len(logs_df)}")
                
            except Exception as e:
                logger.warning(f"LSTM filtering failed: {e}. Using fallback method.")
                filtered_logs = logs_df.copy()
        else:
            filtered_logs = logs_df.copy()
        
        # Step 2: Vector DB Similarity (NEW approach)
        if self.vector_db and not filtered_logs.empty:
            try:
                # Get vector similarity scores
                vector_similarities = self.vector_db.get_similarity_scores(
                    filtered_logs, issue_description
                )
                
                # Filter by vector similarity threshold
                similarity_threshold = Config.VECTOR_DB_CONFIG['similarity_threshold']
                vector_mask = np.array(vector_similarities) >= similarity_threshold
                
                if vector_mask.any():
                    vector_filtered_logs = filtered_logs[vector_mask].copy()
                    vector_filtered_logs['vector_similarity'] = np.array(vector_similarities)[vector_mask]
                    vector_filtered_logs = vector_filtered_logs.sort_values('vector_similarity', ascending=False)
                    
                    logger.info(f"Vector DB filtered {len(vector_filtered_logs)} similar logs from {len(filtered_logs)}")
                    filtered_logs = vector_filtered_logs
                else:
                    logger.info("No logs met vector similarity threshold, using LSTM filtered logs")
                    
            except Exception as e:
                logger.warning(f"Vector DB filtering failed: {e}. Using LSTM filtered logs.")
        
        # Step 3: TF-IDF Similarity (fallback/combined approach)
        if not filtered_logs.empty:
            try:
                # Vectorize issue description and log messages
                messages = filtered_logs['message'].fillna('').astype(str).tolist()
                all_texts = [issue_description] + messages
                
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
                
                # Calculate similarity between issue and log messages
                similarities = cosine_similarity(
                    tfidf_matrix[0:1], tfidf_matrix[1:]
                ).flatten()
                
                # Filter logs with similarity above threshold
                similarity_threshold = 0.1
                similar_mask = similarities >= similarity_threshold
                
                if similar_mask.any():
                    final_logs = filtered_logs[similar_mask].copy()
                    final_logs['tfidf_similarity'] = similarities[similar_mask]
                    
                    # Combine similarity scores if vector DB was used
                    if 'vector_similarity' in final_logs.columns:
                        final_logs['combined_similarity'] = (
                            final_logs['vector_similarity'] * 0.7 + 
                            final_logs['tfidf_similarity'] * 0.3
                        )
                        final_logs = final_logs.sort_values('combined_similarity', ascending=False)
                    else:
                        final_logs = final_logs.sort_values('tfidf_similarity', ascending=False)
                    
                    logger.info(f"Final similarity filtering resulted in {len(final_logs)} relevant logs")
                    return final_logs.head(50)  # Limit to top 50 most relevant logs
                    
            except Exception as e:
                logger.warning(f"TF-IDF similarity filtering failed: {e}")
        
        return filtered_logs.head(50)
    
    def _filter_relevant_logs_fast(self, logs_df: pd.DataFrame, issue_description: str) -> pd.DataFrame:
        """Fast filtering using only LSTM and TF-IDF (no vector DB)"""
        if logs_df.empty:
            return logs_df
        
        # Step 1: LSTM Filtering
        if self.lstm_model is not None:
            try:
                from data.preprocessing import LogPreprocessor
                preprocessor = LogPreprocessor()
                X, _ = preprocessor.prepare_lstm_data(logs_df)
                
                importance_scores = self.lstm_model.predict(X)
                threshold = np.percentile(importance_scores, 30)
                important_mask = importance_scores >= threshold
                
                filtered_logs = logs_df[important_mask].copy()
                logger.info(f"LSTM filtered {len(filtered_logs)} important logs from {len(logs_df)}")
                
            except Exception as e:
                logger.warning(f"LSTM filtering failed: {e}. Using all logs.")
                filtered_logs = logs_df.copy()
        else:
            filtered_logs = logs_df.copy()
        
        # Step 2: TF-IDF Similarity (fast)
        try:
            # Prepare texts for TF-IDF
            texts = filtered_logs['message'].fillna('').astype(str).tolist()
            
            # Fit TF-IDF if not already fitted
            if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                self.tfidf_vectorizer.fit(texts)
            
            # Transform query and logs
            query_vector = self.tfidf_vectorizer.transform([issue_description])
            log_vectors = self.tfidf_vectorizer.transform(texts)
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, log_vectors).flatten()
            
            # Filter by similarity threshold
            similarity_threshold = 0.1  # Lower threshold for TF-IDF
            tfidf_mask = similarities >= similarity_threshold
            
            if tfidf_mask.any():
                tfidf_filtered_logs = filtered_logs[tfidf_mask].copy()
                tfidf_filtered_logs['tfidf_similarity'] = similarities[tfidf_mask]
                tfidf_filtered_logs = tfidf_filtered_logs.sort_values('tfidf_similarity', ascending=False)
                
                logger.info(f"TF-IDF filtered {len(tfidf_filtered_logs)} similar logs from {len(filtered_logs)}")
                filtered_logs = tfidf_filtered_logs
            else:
                logger.info("No logs met TF-IDF similarity threshold, using LSTM filtered logs")
                
        except Exception as e:
            logger.warning(f"TF-IDF filtering failed: {e}. Using LSTM filtered logs.")
        
        return filtered_logs
    
    def _get_historical_context(self, issue_description: str) -> str:
        """Get historical context for an issue from vector database"""
        if not self.vector_db:
            return ""
        
        try:
            historical_context = self.vector_db.get_context_for_issue(issue_description)
            if historical_context:
                logger.info("Retrieved historical context from vector database")
            return historical_context
        except Exception as e:
            logger.warning(f"Failed to get historical context: {e}")
            return ""
    
    def _categorize_issue(self, issue_description: str) -> str:
        """Categorize the issue based on description"""
        issue_lower = issue_description.lower()
        
        for category, keywords in self.issue_patterns.items():
            if any(keyword in issue_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _extract_timeline(self, logs_df: pd.DataFrame) -> List[Dict]:
        """Extract timeline of key events"""
        if logs_df.empty or 'timestamp' not in logs_df.columns:
            return []
        
        timeline = []
        
        # Sort logs by timestamp
        sorted_logs = logs_df.sort_values('timestamp')
        
        # Extract key events
        key_event_patterns = [
            (r'instance.*spawned', 'Instance Spawned'),
            (r'instance.*destroyed', 'Instance Destroyed'),
            (r'vm.*started', 'VM Started'),
            (r'vm.*stopped', 'VM Stopped'),
            (r'attempting.*claim', 'Resource Claim Attempt'),
            (r'claim.*successful', 'Resource Claim Successful'),
            (r'no valid host', 'No Valid Host Found'),
            (r'terminating.*instance', 'Instance Termination Started'),
            (r'error.*connection', 'Connection Error'),
            (r'timeout', 'Timeout Occurred')
        ]
        
        for _, row in sorted_logs.iterrows():
            message = str(row.get('message', '')).lower()
            timestamp = row.get('timestamp')
            
            for pattern, event_type in key_event_patterns:
                if re.search(pattern, message):
                    timeline.append({
                        'timestamp': timestamp,
                        'event_type': event_type,
                        'message': row.get('message', ''),
                        'service': row.get('service_type', 'unknown'),
                        'level': row.get('level', 'unknown'),
                        'instance_id': row.get('instance_id')
                    })
                    break
        
        return timeline[-20:]  # Return last 20 events
    
    def _analyze_patterns(self, logs_df: pd.DataFrame, issue_category: str) -> Dict:
        """Analyze patterns in the logs"""
        patterns = {
            'error_frequency': {},
            'service_distribution': {},
            'time_patterns': {},
            'instance_patterns': {},
            'resource_patterns': {}
        }
        
        if logs_df.empty:
            return patterns
        
        # Error frequency analysis
        error_logs = logs_df[logs_df['level'].str.lower().isin(['error', 'critical'])]
        if not error_logs.empty:
            error_messages = error_logs['message'].str.lower()
            
            # Common error patterns
            error_keywords = ['failed', 'error', 'timeout', 'denied', 'unavailable']
            for keyword in error_keywords:
                count = error_messages.str.contains(keyword, na=False).sum()
                if count > 0:
                    patterns['error_frequency'][keyword] = count
        
        # Service distribution
        if 'service_type' in logs_df.columns:
            service_counts = logs_df['service_type'].value_counts().to_dict()
            patterns['service_distribution'] = service_counts
        
        # Time patterns
        if 'timestamp' in logs_df.columns:
            logs_df['hour'] = pd.to_datetime(logs_df['timestamp']).dt.hour
            hourly_counts = logs_df['hour'].value_counts().to_dict()
            patterns['time_patterns'] = hourly_counts
        
        # Instance patterns
        if 'instance_id' in logs_df.columns:
            instance_counts = logs_df['instance_id'].value_counts().head(10).to_dict()
            patterns['instance_patterns'] = {str(k): v for k, v in instance_counts.items() if pd.notna(k)}
        
        # Resource patterns
        resource_keywords = ['memory', 'disk', 'vcpu', 'storage', 'claim']
        resource_messages = logs_df['message'].str.lower()
        for keyword in resource_keywords:
            count = resource_messages.str.contains(keyword, na=False).sum()
            if count > 0:
                patterns['resource_patterns'][keyword] = count
        
        return patterns
    
    def _generate_rca_with_claude(self, issue_description: str, logs_df: pd.DataFrame, 
                                 timeline: List[Dict], patterns: Dict, issue_category: str) -> str:
        """Generate root cause analysis using Claude API with enhanced context"""
        try:
            # Check API key first
            if not self.client.api_key:
                logger.error("No Claude API key configured")
                return self._generate_fallback_rca(issue_category, patterns)
            
            # Prepare context for Claude with historical context
            context = self._prepare_claude_context(
                issue_description, logs_df, timeline, patterns, issue_category
            )
            
            # Enhanced prompt with more specific instructions
            prompt = f"""
You are an expert OpenStack systems administrator performing root cause analysis on real production logs.

ISSUE: {issue_description}
CATEGORY: {issue_category}
RELEVANT LOGS: {len(logs_df)} entries analyzed

LOG ANALYSIS CONTEXT:
{context}

TASK: Provide a detailed technical root cause analysis based on the actual log patterns shown above.

REQUIREMENTS:
1. **Identify the specific technical root cause** from the log evidence
2. **Cite specific log entries or patterns** mentioned in the context
3. **Provide actionable technical solutions** not generic advice
4. **Use OpenStack terminology** and reference specific services (nova-compute, nova-scheduler, etc.)
5. **Be specific about the failure sequence** shown in the timeline
6. **Consider historical patterns** if similar issues were found

FORMAT:
## Root Cause Analysis
**Primary Technical Cause**: [Specific cause based on log evidence]

**Supporting Evidence from Logs**:
- [Specific log pattern or error]
- [Timeline sequence evidence]
- [Resource or service issue evidence]

**Technical Resolution Steps**:
1. [Specific immediate action]
2. [Configuration change needed]
3. [Monitoring/verification step]

Focus on the actual log data provided, not generic OpenStack troubleshooting.
"""
            
            logger.info("Sending request to Claude API...")
            response = self.client.generate_response(
                prompt=prompt,
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.1
            )
            
            logger.info("Successfully received Claude API response")
            return response
            
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            logger.info("Falling back to enhanced pattern-based analysis...")
            return self._generate_enhanced_fallback_rca(issue_category, patterns, logs_df, timeline)
    
    def _prepare_claude_context(self, issue_description: str, logs_df: pd.DataFrame,
                               timeline: List[Dict], patterns: Dict, issue_category: str) -> str:
        """Prepare context for Claude API with historical context"""
        context_parts = []
        
        # NEW: Add historical context from vector database
        historical_context = self._get_historical_context(issue_description)
        if historical_context:
            context_parts.append("## Historical Similar Issues:")
            context_parts.append(historical_context)
            context_parts.append("")
        
        # Add summary statistics
        context_parts.append(f"## Dataset Overview:")
        context_parts.append(f"- Total relevant log entries: {len(logs_df)}")
        context_parts.append(f"- Issue category: {issue_category}")
        
        # NEW: Add similarity information if available
        if 'vector_similarity' in logs_df.columns:
            avg_vector_sim = logs_df['vector_similarity'].mean()
            context_parts.append(f"- Average vector similarity: {avg_vector_sim:.3f}")
        if 'tfidf_similarity' in logs_df.columns:
            avg_tfidf_sim = logs_df['tfidf_similarity'].mean()
            context_parts.append(f"- Average TF-IDF similarity: {avg_tfidf_sim:.3f}")
        if 'combined_similarity' in logs_df.columns:
            avg_combined_sim = logs_df['combined_similarity'].mean()
            context_parts.append(f"- Average combined similarity: {avg_combined_sim:.3f}")
        
        # Timeline summary with more detail
        if timeline:
            context_parts.append("\n## Timeline of Critical Events:")
            for event in timeline[-8:]:  # Show more events
                timestamp = event.get('timestamp', 'Unknown')
                event_type = event.get('event_type', 'Unknown')
                service = event.get('service', 'Unknown')
                message = event.get('message', '')[:100]  # First 100 chars
                context_parts.append(f"- {timestamp}: {event_type} ({service})")
                if message:
                    context_parts.append(f"  Message: {message}...")
        
        # Enhanced patterns analysis
        context_parts.append("\n## Detailed Log Analysis:")
        
        # Error frequency with specific examples
        if patterns.get('error_frequency'):
            context_parts.append("### Critical Error Patterns:")
            for error, count in patterns['error_frequency'].items():
                context_parts.append(f"- '{error}': {count} occurrences")
            
            # Find specific error examples
            if 'message' in logs_df.columns:
                error_logs = logs_df[logs_df['level'].str.upper() == 'ERROR']['message'].head(3)
                if not error_logs.empty:
                    context_parts.append("\n### Sample Error Messages:")
                    for i, error_msg in enumerate(error_logs, 1):
                        context_parts.append(f"- Error {i}: {str(error_msg)[:150]}...")
        
        # Service analysis with specific details
        if patterns.get('service_distribution'):
            context_parts.append("\n### Service Activity Analysis:")
            for service, count in patterns['service_distribution'].items():
                percentage = (count / len(logs_df) * 100) if len(logs_df) > 0 else 0
                context_parts.append(f"- {service}: {count} entries ({percentage:.1f}% of total)")
        
        # Resource patterns with context
        if patterns.get('resource_patterns'):
            context_parts.append("\n### Resource-related Issues:")
            for resource, count in patterns['resource_patterns'].items():
                context_parts.append(f"- {resource} mentioned: {count} times")
        
        # Instance-specific analysis
        if 'instance_id' in logs_df.columns:
            instance_issues = logs_df.groupby('instance_id').size().head(3)
            if len(instance_issues) > 0:
                context_parts.append("\n### Most Active Instances:")
                for instance_id, count in instance_issues.items():
                    if pd.notna(instance_id):
                        context_parts.append(f"- Instance {str(instance_id)[-12:]}: {count} log entries")
        
        # Sample critical log entries with full context
        if not logs_df.empty:
            context_parts.append("\n## Critical Log Entries (Full Context):")
            
            # Get error logs first
            error_logs = logs_df[logs_df['level'].str.upper() == 'ERROR'].head(5)
            if not error_logs.empty:
                context_parts.append("### ERROR Level Entries:")
                for _, log in error_logs.iterrows():
                    timestamp = log.get('timestamp', 'Unknown')
                    service = log.get('service_type', 'Unknown')
                    message = log.get('message', 'Unknown')
                    context_parts.append(f"- [{timestamp}] {service}: {message}")
            
            # Get warning logs if no errors
            elif len(logs_df[logs_df['level'].str.upper() == 'WARNING']) > 0:
                warning_logs = logs_df[logs_df['level'].str.upper() == 'WARNING'].head(3)
                context_parts.append("### WARNING Level Entries:")
                for _, log in warning_logs.iterrows():
                    timestamp = log.get('timestamp', 'Unknown')
                    service = log.get('service_type', 'Unknown')
                    message = log.get('message', 'Unknown')
                    context_parts.append(f"- [{timestamp}] {service}: {message}")
            
            # Get some info logs for context
            else:
                info_logs = logs_df.head(3)
                context_parts.append("### Sample Log Entries:")
                for _, log in info_logs.iterrows():
                    timestamp = log.get('timestamp', 'Unknown')
                    level = log.get('level', 'Unknown')
                    service = log.get('service_type', 'Unknown')
                    message = log.get('message', 'Unknown')[:200]
                    context_parts.append(f"- [{timestamp}] {level} {service}: {message}")
        
        return "\n".join(context_parts)
    
    def _generate_enhanced_fallback_rca(self, issue_category: str, patterns: Dict, 
                                       logs_df: pd.DataFrame, timeline: List[Dict]) -> str:
        """Generate enhanced fallback RCA when Claude API is unavailable"""
        
        # Analyze actual log content for specific issues
        specific_analysis = self._analyze_specific_log_patterns(logs_df, patterns, timeline)
        
        enhanced_templates = {
            'disk_space': f"""
## Root Cause Analysis

**Primary Technical Cause**: Insufficient disk space detected on compute nodes

**Supporting Evidence from Logs**:
{specific_analysis.get('evidence', '- Multiple disk allocation errors in nova-compute logs')}

**Technical Resolution Steps**:
1. Check disk usage: `df -h` on compute nodes
2. Review nova.conf disk_allocation_ratio setting
3. Clean up instance files: `nova-manage cell_v2 delete_host_cell_mappings`
4. Monitor with: `openstack hypervisor stats show`
""",
            'resource_shortage': f"""
## Root Cause Analysis

**Primary Technical Cause**: Compute resource exhaustion (CPU/Memory/Disk limits exceeded)

**Supporting Evidence from Logs**:
{specific_analysis.get('evidence', '- Resource claim failures in scheduler logs')}

**Technical Resolution Steps**:
1. Check resource usage: `openstack hypervisor list --long`
2. Review allocation ratios in nova.conf
3. Add compute capacity or adjust overcommit ratios
4. Verify placement service resource tracking
""",
            'instance_lifecycle': f"""
## Root Cause Analysis

**Primary Technical Cause**: Instance state transition failure in nova-compute service

**Supporting Evidence from Logs**:
{specific_analysis.get('evidence', '- Instance lifecycle state errors detected')}

**Technical Resolution Steps**:
1. Check nova-compute service status: `systemctl status nova-compute`
2. Review instance state: `nova show <instance-id>`
3. Reset instance state if needed: `nova reset-state <instance-id>`
4. Check hypervisor connectivity and libvirt status
""",
            'network_issues': f"""
## Root Cause Analysis

**Primary Technical Cause**: Network configuration or neutron service connectivity issues

**Supporting Evidence from Logs**:
{specific_analysis.get('evidence', '- Network setup failures during instance creation')}

**Technical Resolution Steps**:
1. Verify neutron services: `openstack network agent list`
2. Check security group rules: `openstack security group rule list`
3. Test network connectivity between compute and network nodes
4. Review neutron configuration and restart services if needed
"""
        }
        
        base_analysis = f"""
## Root Cause Analysis

**Primary Technical Cause**: {specific_analysis.get('primary_cause', 'Multiple service interaction failures detected')}

**Supporting Evidence from Logs**:
{specific_analysis.get('evidence', '- Service communication errors identified')}

**Technical Resolution Steps**:
{specific_analysis.get('resolution_steps', '''1. Check service status: `openstack service list`
2. Review configuration files for consistency
3. Restart affected services in proper order
4. Monitor service logs for continued issues''')}
"""
        
        return enhanced_templates.get(issue_category, base_analysis)
    
    def _analyze_specific_log_patterns(self, logs_df: pd.DataFrame, patterns: Dict, timeline: List[Dict]) -> Dict:
        """Analyze actual log patterns to generate specific insights"""
        analysis = {
            'primary_cause': 'OpenStack service operational analysis',
            'evidence': [],
            'resolution_steps': []
        }
        
        if logs_df.empty:
            return analysis
        
        # Analyze patterns in messages, even from INFO logs
        evidence_list = []
        
        # Check for specific OpenStack operational patterns
        if 'message' in logs_df.columns:
            all_messages = logs_df['message'].fillna('')
            
            # OpenStack operational patterns (not just errors)
            operational_patterns = {
                'instance': 'Instance lifecycle operations detected',
                'server': 'Server management activities found',
                'spawn': 'VM spawning/creation processes observed',
                'claim': 'Resource claim operations detected',
                'host': 'Host selection and placement activities',
                'allocation': 'Resource allocation processes found',
                'network': 'Network configuration activities detected',
                'port': 'Network port management operations',
                'volume': 'Volume/storage operations found',
                'image': 'Image management activities detected',
                'flavor': 'Flavor/instance type selections made',
                'quota': 'Quota and resource limit checks',
                'scheduler': 'Scheduler decision-making processes',
                'compute': 'Compute node operations detected',
                'hypervisor': 'Hypervisor management activities'
            }
            
            # Look for operational patterns in all logs (not just errors)
            for pattern, description in operational_patterns.items():
                matches = all_messages.str.contains(pattern, case=False, na=False).sum()
                if matches > 0:
                    evidence_list.append(f"- {description} ({matches} occurrences)")
                    
                    # Set specific primary cause based on dominant patterns
                    if pattern in ['instance', 'spawn', 'server'] and matches > 10:
                        analysis['primary_cause'] = 'Instance lifecycle management operations'
                    elif pattern in ['scheduler', 'host', 'claim'] and matches > 5:
                        analysis['primary_cause'] = 'Resource scheduling and allocation processes'
                    elif pattern in ['network', 'port'] and matches > 5:
                        analysis['primary_cause'] = 'Network service configuration activities'
                    elif pattern in ['compute', 'hypervisor'] and matches > 5:
                        analysis['primary_cause'] = 'Compute node management operations'
            
            # Look for specific OpenStack request patterns
            request_patterns = {
                'GET': 'API GET requests processed',
                'POST': 'API POST operations executed',
                'PUT': 'API PUT updates performed',
                'DELETE': 'API DELETE operations executed',
                'HTTP': 'HTTP API interactions logged'
            }
            
            for pattern, description in request_patterns.items():
                matches = all_messages.str.contains(pattern, case=False, na=False).sum()
                if matches > 0:
                    evidence_list.append(f"- {description} ({matches} requests)")
            
            # Look for WARNING level issues specifically
            warning_logs = logs_df[logs_df['level'].str.upper() == 'WARNING']['message']
            if len(warning_logs) > 0:
                evidence_list.append(f"- {len(warning_logs)} WARNING level events requiring attention")
                for warning_msg in warning_logs:
                    if len(str(warning_msg)) > 20:  # Skip very short messages
                        evidence_list.append(f"  • Warning: {str(warning_msg)[:100]}...")
        
        # Analyze service activity patterns
        if patterns.get('service_distribution'):
            service_analysis = []
            total_logs = sum(patterns['service_distribution'].values())
            
            for service, count in patterns['service_distribution'].items():
                percentage = (count / total_logs * 100) if total_logs > 0 else 0
                if percentage > 20:  # Significant activity
                    service_analysis.append(f"- High activity in {service} ({count} logs, {percentage:.1f}%)")
                elif percentage > 5:
                    service_analysis.append(f"- Moderate activity in {service} ({count} logs, {percentage:.1f}%)")
            
            evidence_list.extend(service_analysis)
        
        # Analyze timeline patterns for operational flow
        if timeline and len(timeline) > 0:
            timeline_events = [event.get('event_type', 'Unknown') for event in timeline[-8:]]
            if timeline_events:
                evidence_list.append(f"- Operation sequence: {' → '.join(timeline_events[-5:])}")
                
                # Look for repeated operations
                from collections import Counter
                event_counts = Counter(timeline_events)
                frequent_ops = [op for op, count in event_counts.items() if count > 2]
                if frequent_ops:
                    evidence_list.append(f"- Frequent operations: {', '.join(frequent_ops)}")
        
        # If no specific patterns found, analyze general activity
        if not evidence_list:
            evidence_list = [
                f"- Normal operational logging from {len(logs_df)} entries",
                f"- Primarily INFO level logs ({logs_df['level'].value_counts().get('INFO', 0)} entries)",
                f"- Services active: {', '.join(patterns.get('service_distribution', {}).keys())}"
            ]
            analysis['primary_cause'] = 'Normal OpenStack operational activity'
        
        analysis['evidence'] = '\n'.join(evidence_list)
        
        return analysis
    
    def _generate_recommendations(self, issue_category: str, patterns: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = {
            'disk_space': [
                "Check disk usage on all compute nodes",
                "Review and adjust disk overcommitment ratios",
                "Implement automated cleanup of unused instance files",
                "Monitor disk space usage with alerts"
            ],
            'resource_shortage': [
                "Add additional compute nodes to the cluster",
                "Review and optimize resource allocation policies",
                "Implement resource monitoring and alerting",
                "Consider migrating instances to balance load"
            ],
            'network_issues': [
                "Verify network connectivity between all components",
                "Check security group rules and firewall settings",
                "Review network service configurations",
                "Test network performance and latency"
            ],
            'instance_lifecycle': [
                "Review instance creation and deletion processes",
                "Check hypervisor and libvirt configurations",
                "Monitor instance state transitions",
                "Verify image and flavor configurations"
            ]
        }
        
        base_recommendations = [
            "Monitor system logs continuously for early detection",
            "Implement proper backup and recovery procedures",
            "Review and update system documentation",
            "Conduct regular system health checks"
        ]
        
        category_recs = recommendations.get(issue_category, [])
        return category_recs + base_recommendations