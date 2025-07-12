import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ai_client import ClaudeClient
from config.config import Config

# Import vector database service
try:
    from services.vector_db_service import VectorDBService
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    logging.warning("VectorDBService not available - falling back to TF-IDF only")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRCAAnalyzer:
    """Hybrid Root Cause Analysis engine combining LSTM importance filtering with Vector DB semantic search"""
    
    def __init__(self, anthropic_api_key: str, lstm_model=None, vector_db=None):
        self.client = ClaudeClient(anthropic_api_key)
        self.lstm_model = lstm_model
        self.vector_db = vector_db or (VectorDBService() if VECTOR_DB_AVAILABLE else None)
        
        # TF-IDF vectorizer for fallback
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Issue categorization patterns
        self.issue_patterns = {
            'resource_shortage': ['resource', 'memory', 'disk', 'cpu', 'allocation', 'insufficient'],
            'network_issues': ['network', 'connection', 'timeout', 'unreachable', 'dns'],
            'authentication': ['auth', 'authentication', 'token', 'credential', 'permission'],
            'service_failure': ['service', 'nova', 'keystone', 'glance', 'neutron', 'failed'],
            'instance_issues': ['instance', 'vm', 'spawn', 'launch', 'terminate', 'destroy'],
            'database': ['database', 'mysql', 'postgresql', 'connection', 'query'],
            'storage': ['storage', 'volume', 'cinder', 'swift', 'object', 'block']
        }
        
        # Performance metrics
        self.performance_metrics = {}
        
        logger.info("HybridRCAAnalyzer initialized successfully")
        if self.lstm_model:
            logger.info("✓ LSTM model loaded")
        if self.vector_db:
            logger.info("✓ Vector DB service available")
    
    def analyze_issue(self, issue_description: str, logs_df: pd.DataFrame, fast_mode: bool = False) -> Dict:
        """Analyze an issue using hybrid LSTM + Vector DB approach"""
        start_time = datetime.now()
        logger.info(f"Starting hybrid RCA analysis: {issue_description}")
        
        try:
            if fast_mode:
                logger.info("Using fast mode: LSTM + TF-IDF only")
                relevant_logs = self._fast_filter_logs(logs_df, issue_description)
                analysis_mode = 'fast'
            else:
                logger.info("Using hybrid mode: LSTM + Vector DB")
                relevant_logs = self._hybrid_filter_logs(logs_df, issue_description)
                analysis_mode = 'hybrid'
            
            # Step 2: Identify issue category
            issue_category = self._categorize_issue(issue_description)
            
            # Step 3: Extract key events and timeline
            timeline = self._extract_timeline(relevant_logs)
            
            # Step 4: Analyze patterns
            patterns = self._analyze_patterns(relevant_logs, issue_category)
            
            # Step 5: Generate RCA using Claude API
            rca_analysis = self._generate_rca_with_claude(
                issue_description, relevant_logs, timeline, patterns, issue_category
            )
            
            # Calculate performance metrics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            self.performance_metrics = {
                'processing_time': processing_time,
                'total_logs': len(logs_df),
                'filtered_logs': len(relevant_logs),
                'analysis_mode': analysis_mode,
                'lstm_available': self.lstm_model is not None,
                'vector_db_available': self.vector_db is not None
            }
            
            return {
                'issue_description': issue_description,
                'issue_category': issue_category,
                'relevant_logs_count': len(relevant_logs),
                'timeline': timeline,
                'patterns': patterns,
                'root_cause_analysis': rca_analysis,
                'recommendations': self._generate_recommendations(issue_category, patterns),
                'analysis_mode': analysis_mode,
                'performance_metrics': self.performance_metrics,
                'filtered_logs': relevant_logs
            }
            
        except Exception as e:
            logger.error(f"Hybrid RCA analysis failed: {e}")
            return self._generate_fallback_analysis(issue_description, logs_df)
    
    def analyze_issue_from_files(self, issue_description: str, log_files_path: str, fast_mode: bool = False) -> Dict:
        """Analyze an issue directly from log files without pre-loading"""
        logger.info(f"Starting RCA analysis from files: {log_files_path}")
        
        try:
            # Load logs from files
            from utils.log_cache import LogCache
            log_cache = LogCache()
            logs_df = log_cache.get_cached_logs(log_files_path)
            
            if logs_df.empty:
                logger.error(f"No valid logs found in {log_files_path}")
                return {
                    'error': f'No valid logs found in {log_files_path}',
                    'issue_description': issue_description
                }
            
            # Perform standard analysis
            return self.analyze_issue(issue_description, logs_df, fast_mode)
            
        except Exception as e:
            logger.error(f"Failed to analyze from files: {e}")
            return {
                'error': f'Analysis failed: {str(e)}',
                'issue_description': issue_description
            }
    
    def analyze_issue_from_text(self, issue_description: str, log_text: str, fast_mode: bool = False) -> Dict:
        """Analyze an issue from raw log text input"""
        logger.info("Starting RCA analysis from text input")
        
        try:
            # Parse log text into DataFrame
            from data.log_ingestion import LogIngestionManager
            import tempfile
            import os
            
            # Create temporary file with log text
            with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as temp_file:
                temp_file.write(log_text)
                temp_file_path = temp_file.name
            
            try:
                # Ingest the temporary file
                ingestion_manager = LogIngestionManager()
                logs_df = ingestion_manager.ingest_multiple_files([temp_file_path])
                
                if logs_df.empty:
                    logger.error("No valid logs found in text input")
                    return {
                        'error': 'No valid logs found in text input',
                        'issue_description': issue_description
                    }
                
                # Apply feature engineering
                from utils.feature_engineering import FeatureEngineer
                feature_engineer = FeatureEngineer()
                logs_df = feature_engineer.engineer_all_features(logs_df)
                
                # Perform standard analysis
                return self.analyze_issue(issue_description, logs_df, fast_mode)
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            
        except Exception as e:
            logger.error(f"Failed to analyze from text: {e}")
            return {
                'error': f'Analysis failed: {str(e)}',
                'issue_description': issue_description
            }
    
    def _hybrid_filter_logs(self, logs_df: pd.DataFrame, issue_description: str) -> pd.DataFrame:
        """Hybrid filtering: LSTM importance + Vector DB semantic search"""
        if logs_df.empty:
            return logs_df
        
        logger.info(f"Starting hybrid filtering on {len(logs_df)} logs")
        
        # Step 1: LSTM Importance Filtering
        lstm_start = datetime.now()
        lstm_filtered_logs = self._lstm_filter_logs(logs_df, issue_description)
        lstm_time = (datetime.now() - lstm_start).total_seconds()
        
        logger.info(f"LSTM filtering: {len(lstm_filtered_logs)} logs in {lstm_time:.2f}s")
        
        # Step 2: Vector DB Semantic Search
        vector_start = datetime.now()
        vector_results = self._vector_db_search(lstm_filtered_logs, issue_description)
        vector_time = (datetime.now() - vector_start).total_seconds()
        
        logger.info(f"Vector DB search: {len(vector_results)} results in {vector_time:.2f}s")
        
        # Step 3: Combine and Rank Results
        if vector_results:
            final_results = self._combine_and_rank_results(lstm_filtered_logs, vector_results)
            logger.info(f"Combined results: {len(final_results)} final logs")
            return final_results
        else:
            logger.warning("Vector DB search failed, using LSTM filtered logs")
            return lstm_filtered_logs.head(50)
    
    def _lstm_filter_logs(self, logs_df: pd.DataFrame, issue_description: str) -> pd.DataFrame:
        """Filter logs by LSTM importance scores"""
        if self.lstm_model is None:
            logger.warning("LSTM model not available, using all logs")
            return logs_df.copy()
        
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
            filtered_logs['lstm_importance'] = importance_scores[important_mask]
            
            logger.info(f"LSTM filtered {len(filtered_logs)} important logs from {len(logs_df)}")
            logger.info(f"Importance score range: {importance_scores.min():.3f} - {importance_scores.max():.3f}")
            
            return filtered_logs
            
        except Exception as e:
            logger.warning(f"LSTM filtering failed: {e}. Using all logs.")
            return logs_df.copy()
    
    def _vector_db_search(self, lstm_filtered_logs: pd.DataFrame, issue_description: str) -> List[Dict]:
        """Search Vector DB for semantically similar logs"""
        if not self.vector_db or lstm_filtered_logs.empty:
            return []
        
        try:
            # Get indices of LSTM-filtered logs
            filtered_indices = lstm_filtered_logs.index.tolist()
            
            # Search Vector DB with metadata filter
            similar_logs = self.vector_db.search_similar_logs(
                query=issue_description,
                filter_metadata={'original_index': filtered_indices},
                top_k=50
            )
            
            logger.info(f"Vector DB found {len(similar_logs)} similar logs")
            return similar_logs
            
        except Exception as e:
            logger.warning(f"Vector DB search failed: {e}")
            return []
    
    def _combine_and_rank_results(self, lstm_filtered_logs: pd.DataFrame, vector_results: List[Dict]) -> pd.DataFrame:
        """Combine LSTM importance and Vector DB similarity scores"""
        combined_results = []
        
        for log in vector_results:
            try:
                original_index = int(log['metadata']['original_index'])
                
                # Get LSTM importance score
                if original_index in lstm_filtered_logs.index:
                    lstm_score = lstm_filtered_logs.loc[original_index, 'lstm_importance']
                else:
                    lstm_score = 0.5  # Default score if not found
                
                vector_score = log['similarity']
                
                # Combined score: 70% LSTM importance + 30% Vector similarity
                combined_score = (lstm_score * 0.7) + (vector_score * 0.3)
                
                # Get original log data
                original_log = lstm_filtered_logs.loc[original_index].to_dict()
                
                # Create enhanced result
                enhanced_log = {
                    'timestamp': original_log.get('timestamp'),
                    'service_type': original_log.get('service_type'),
                    'level': original_log.get('level'),
                    'message': original_log.get('message'),
                    'instance_id': original_log.get('instance_id'),
                    'lstm_importance': lstm_score,
                    'vector_similarity': vector_score,
                    'combined_score': combined_score,
                    'ranking_reason': self._explain_ranking(lstm_score, vector_score)
                }
                
                combined_results.append(enhanced_log)
                
            except Exception as e:
                logger.warning(f"Failed to combine scores for log: {e}")
                continue
        
        # Convert to DataFrame and sort by combined score
        if combined_results:
            result_df = pd.DataFrame(combined_results)
            result_df = result_df.sort_values('combined_score', ascending=False)
            
            logger.info(f"Combined scoring completed: {len(result_df)} results")
            logger.info(f"Score range: {result_df['combined_score'].min():.3f} - {result_df['combined_score'].max():.3f}")
            
            return result_df
        else:
            logger.warning("No combined results, returning LSTM filtered logs")
            return lstm_filtered_logs.head(50)
    
    def _explain_ranking(self, lstm_score: float, vector_score: float) -> str:
        """Explain why a log was ranked as it was"""
        if lstm_score >= 0.8 and vector_score >= 0.8:
            return "High importance + High similarity"
        elif lstm_score >= 0.8:
            return "High importance + Moderate similarity"
        elif vector_score >= 0.8:
            return "Moderate importance + High similarity"
        else:
            return "Moderate importance + Moderate similarity"
    
    def _fast_filter_logs(self, logs_df: pd.DataFrame, issue_description: str) -> pd.DataFrame:
        """Fast filtering using only LSTM and TF-IDF (no Vector DB)"""
        if logs_df.empty:
            return logs_df
        
        logger.info("Using fast filtering mode")
        
        # Step 1: LSTM Filtering
        lstm_filtered_logs = self._lstm_filter_logs(logs_df, issue_description)
        
        # Step 2: TF-IDF Similarity
        try:
            # Prepare texts for TF-IDF
            texts = lstm_filtered_logs['message'].fillna('').astype(str).tolist()
            
            # Fit TF-IDF if not already fitted
            if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                self.tfidf_vectorizer.fit(texts)
            
            # Transform query and logs
            query_vector = self.tfidf_vectorizer.transform([issue_description])
            log_vectors = self.tfidf_vectorizer.transform(texts)
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, log_vectors).flatten()
            
            # Filter by similarity threshold
            similarity_threshold = 0.1
            tfidf_mask = similarities >= similarity_threshold
            
            if tfidf_mask.any():
                tfidf_filtered_logs = lstm_filtered_logs[tfidf_mask].copy()
                tfidf_filtered_logs['tfidf_similarity'] = similarities[tfidf_mask]
                tfidf_filtered_logs = tfidf_filtered_logs.sort_values('tfidf_similarity', ascending=False)
                
                logger.info(f"Fast filtering: {len(tfidf_filtered_logs)} logs")
                return tfidf_filtered_logs.head(50)
            else:
                logger.info("No logs met TF-IDF threshold, using LSTM filtered logs")
                return lstm_filtered_logs.head(50)
                
        except Exception as e:
            logger.warning(f"TF-IDF filtering failed: {e}. Using LSTM filtered logs.")
            return lstm_filtered_logs.head(50)
    
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
        
        for _, log in sorted_logs.iterrows():
            message = str(log.get('message', '')).lower()
            
            for pattern, event_type in key_event_patterns:
                if re.search(pattern, message):
                    timeline.append({
                        'timestamp': log.get('timestamp'),
                        'event': event_type,
                        'message': log.get('message'),
                        'service': log.get('service_type'),
                        'level': log.get('level')
                    })
                    break
        
        return timeline
    
    def _analyze_patterns(self, logs_df: pd.DataFrame, issue_category: str) -> Dict:
        """Analyze patterns in the filtered logs"""
        if logs_df.empty:
            return {}
        
        patterns = {
            'error_count': len(logs_df[logs_df['level'].str.contains('ERROR|CRITICAL', case=False, na=False)]),
            'service_distribution': logs_df['service_type'].value_counts().to_dict(),
            'level_distribution': logs_df['level'].value_counts().to_dict(),
            'time_range': {
                'start': logs_df['timestamp'].min() if 'timestamp' in logs_df.columns else None,
                'end': logs_df['timestamp'].max() if 'timestamp' in logs_df.columns else None
            }
        }
        
        # Add category-specific patterns
        if issue_category == 'resource_shortage':
            patterns['resource_patterns'] = self._analyze_resource_patterns(logs_df)
        elif issue_category == 'network_issues':
            patterns['network_patterns'] = self._analyze_network_patterns(logs_df)
        
        return patterns
    
    def _analyze_resource_patterns(self, logs_df: pd.DataFrame) -> Dict:
        """Analyze resource-related patterns"""
        resource_patterns = {
            'memory_issues': len(logs_df[logs_df['message'].str.contains('memory|ram', case=False, na=False)]),
            'disk_issues': len(logs_df[logs_df['message'].str.contains('disk|storage|space', case=False, na=False)]),
            'cpu_issues': len(logs_df[logs_df['message'].str.contains('cpu|processor', case=False, na=False)]),
            'allocation_failures': len(logs_df[logs_df['message'].str.contains('allocation|claim', case=False, na=False)])
        }
        return resource_patterns
    
    def _analyze_network_patterns(self, logs_df: pd.DataFrame) -> Dict:
        """Analyze network-related patterns"""
        network_patterns = {
            'connection_failures': len(logs_df[logs_df['message'].str.contains('connection|connect', case=False, na=False)]),
            'timeout_issues': len(logs_df[logs_df['message'].str.contains('timeout', case=False, na=False)]),
            'dns_issues': len(logs_df[logs_df['message'].str.contains('dns|hostname', case=False, na=False)]),
            'network_unreachable': len(logs_df[logs_df['message'].str.contains('unreachable|network', case=False, na=False)])
        }
        return network_patterns
    
    def _generate_rca_with_claude(self, issue_description: str, logs_df: pd.DataFrame, 
                                 timeline: List[Dict], patterns: Dict, issue_category: str) -> str:
        """Generate RCA using Claude API with enhanced context"""
        try:
            # Prepare context for Claude
            context = self._prepare_claude_context(issue_description, logs_df, timeline, patterns, issue_category)
            
            # Generate RCA prompt
            prompt = f"""
You are an expert OpenStack system administrator performing root cause analysis.

ISSUE: {issue_description}
CATEGORY: {issue_category}

CONTEXT:
{context}

Please provide a detailed root cause analysis including:
1. Summary of the issue
2. Key events and timeline
3. Root cause identification
4. Contributing factors
5. Impact assessment

Focus on actionable insights and technical details.
"""
            
            # Get response from Claude
            response = self.client.generate_response(prompt)
            
            if response:
                return response
            else:
                return self._generate_fallback_rca(issue_category, patterns, logs_df, timeline)
                
        except Exception as e:
            logger.error(f"Failed to generate RCA with Claude: {e}")
            return self._generate_fallback_rca(issue_category, patterns, logs_df, timeline)
    
    def _prepare_claude_context(self, issue_description: str, logs_df: pd.DataFrame,
                               timeline: List[Dict], patterns: Dict, issue_category: str) -> str:
        """Prepare context for Claude API"""
        context_parts = []
        
        # Add issue description
        context_parts.append(f"Issue Description: {issue_description}")
        context_parts.append(f"Issue Category: {issue_category}")
        context_parts.append("")
        
        # Add patterns analysis
        context_parts.append("Pattern Analysis:")
        context_parts.append(f"- Total relevant logs: {len(logs_df)}")
        context_parts.append(f"- Error count: {patterns.get('error_count', 0)}")
        context_parts.append(f"- Service distribution: {patterns.get('service_distribution', {})}")
        context_parts.append("")
        
        # Add timeline
        if timeline:
            context_parts.append("Key Events Timeline:")
            for event in timeline[:10]:  # Top 10 events
                context_parts.append(f"- {event['timestamp']}: {event['event']} ({event['service']})")
            context_parts.append("")
        
        # Add relevant logs (top 20 by combined score)
        if not logs_df.empty:
            context_parts.append("Most Relevant Logs:")
            top_logs = logs_df.head(20)
            
            for _, log in top_logs.iterrows():
                score = log.get('combined_score', log.get('lstm_importance', 0))
                context_parts.append(f"- [{log.get('level', 'INFO')}] {log.get('service_type', 'unknown')}: {log.get('message', '')} (Score: {score:.3f})")
        
        return "\n".join(context_parts)
    
    def _generate_fallback_rca(self, issue_category: str, patterns: Dict, 
                              logs_df: pd.DataFrame, timeline: List[Dict]) -> str:
        """Generate fallback RCA when Claude API fails"""
        rca_parts = []
        
        rca_parts.append(f"Root Cause Analysis for {issue_category} issue:")
        rca_parts.append("")
        
        # Add pattern-based analysis
        rca_parts.append("Pattern Analysis:")
        rca_parts.append(f"- Found {len(logs_df)} relevant log entries")
        rca_parts.append(f"- {patterns.get('error_count', 0)} error-level messages")
        
        if 'service_distribution' in patterns:
            rca_parts.append("- Most affected services:")
            for service, count in list(patterns['service_distribution'].items())[:3]:
                rca_parts.append(f"  * {service}: {count} events")
        
        # Add timeline analysis
        if timeline:
            rca_parts.append("")
            rca_parts.append("Key Events:")
            for event in timeline[:5]:
                rca_parts.append(f"- {event['timestamp']}: {event['event']}")
        
        # Add recommendations
        rca_parts.append("")
        rca_parts.append("Recommendations:")
        recommendations = self._generate_recommendations(issue_category, patterns)
        for rec in recommendations:
            rca_parts.append(f"- {rec}")
        
        return "\n".join(rca_parts)
    
    def _generate_recommendations(self, issue_category: str, patterns: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if issue_category == 'resource_shortage':
            recommendations.extend([
                "Check available resources on compute nodes",
                "Monitor resource usage patterns",
                "Consider scaling compute resources",
                "Review resource allocation policies"
            ])
        elif issue_category == 'network_issues':
            recommendations.extend([
                "Verify network connectivity between services",
                "Check DNS resolution",
                "Monitor network timeouts",
                "Review network configuration"
            ])
        elif issue_category == 'authentication':
            recommendations.extend([
                "Verify authentication tokens",
                "Check user permissions",
                "Review authentication policies",
                "Monitor authentication failures"
            ])
        else:
            recommendations.extend([
                "Review system logs for additional context",
                "Check service health status",
                "Monitor system performance metrics",
                "Consider system maintenance if needed"
            ])
        
        return recommendations
    
    def _generate_fallback_analysis(self, issue_description: str, logs_df: pd.DataFrame) -> Dict:
        """Generate fallback analysis when main analysis fails"""
        return {
            'issue_description': issue_description,
            'issue_category': 'unknown',
            'relevant_logs_count': len(logs_df),
            'timeline': [],
            'patterns': {},
            'root_cause_analysis': f"Analysis failed for issue: {issue_description}. Please check system logs manually.",
            'recommendations': ["Check system logs manually", "Verify service status", "Contact system administrator"],
            'analysis_mode': 'fallback',
            'performance_metrics': {'processing_time': 0, 'error': 'Analysis failed'},
            'filtered_logs': logs_df.head(10) if not logs_df.empty else pd.DataFrame()
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics from last analysis"""
        return self.performance_metrics 