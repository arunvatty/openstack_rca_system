import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

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

class RCAAnalyzer:
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
            'resource_shortage': ['resource', 'memory', 'disk', 'cpu', 'allocation', 'insufficient', 'space left', 'no space', 'allocation failure', 'memory allocation'],
            'network_issues': ['network', 'connection', 'timeout', 'unreachable', 'dns', 'nova-conductor', 'messaging', 'rpc'],
            'authentication': ['auth', 'authentication', 'token', 'credential', 'permission', 'denied', 'unauthorized'],
            'service_failure': ['service', 'nova', 'keystone', 'glance', 'neutron', 'failed', 'failure', 'exception', 'error'],
            'instance_issues': ['instance', 'vm', 'spawn', 'launch', 'terminate', 'destroy', 'state', 'update instance'],
            'database': ['database', 'mysql', 'postgresql', 'connection', 'query', 'operationalerror', 'db'],
            'storage': ['storage', 'volume', 'cinder', 'swift', 'object', 'block', 'vif', 'plugging'],
            'timeout_issues': ['timeout', 'timed out', 'connection timeout', 'nova-conductor', 'messaging timeout', 'rpc timeout']
        }
        
        # Performance metrics
        self.performance_metrics = {}
        
        logger.info("RCAAnalyzer initialized successfully")
        if self.lstm_model:
            logger.info("✓ LSTM model loaded")
        if self.vector_db:
            logger.info("✓ Vector DB service available")
    
    def analyze_issue(self, issue_description: str, logs_df: pd.DataFrame, fast_mode: bool = False) -> Dict:
        """Analyze an issue using hybrid LSTM + Vector DB approach"""
        start_time = datetime.now()
        logger.info(f"Starting hybrid RCA analysis: {issue_description}")
        
        # Detailed timing breakdown
        timing_breakdown = {}
        
        try:
            # Step 1: Data Loading Check (should be instant if cached)
            data_load_start = datetime.now()
            if logs_df.empty:
                logger.error("No log data provided for analysis")
                return self._generate_fallback_analysis(issue_description, logs_df)
            data_load_time = (datetime.now() - data_load_start).total_seconds()
            timing_breakdown['data_loading'] = data_load_time
            logger.info(f"Data loading: {data_load_time:.3f}s ({len(logs_df)} logs)")
            
            # Step 2: Analysis Mode Selection
            if fast_mode:
                logger.info("Using fast mode: LSTM + TF-IDF only")
                relevant_logs = self._fast_filter_logs(logs_df, issue_description)
                analysis_mode = 'fast'
            else:
                logger.info("Using hybrid mode: LSTM + Vector DB")
                relevant_logs = self._hybrid_filter_logs(logs_df, issue_description)
                analysis_mode = 'hybrid'
            
            # Step 3: Issue Categorization
            category_start = datetime.now()
            issue_category = self._categorize_issue(issue_description)
            category_time = (datetime.now() - category_start).total_seconds()
            timing_breakdown['categorization'] = category_time
            
            # Step 4: Timeline Extraction
            timeline_start = datetime.now()
            timeline = self._extract_timeline(relevant_logs)
            timeline_time = (datetime.now() - timeline_start).total_seconds()
            timing_breakdown['timeline_extraction'] = timeline_time
            
            # Step 5: Pattern Analysis
            pattern_start = datetime.now()
            patterns = self._analyze_patterns(relevant_logs, issue_category)
            pattern_time = (datetime.now() - pattern_start).total_seconds()
            timing_breakdown['pattern_analysis'] = pattern_time
            
            # Step 6: Claude API Analysis
            claude_start = datetime.now()
            rca_analysis, prompt = self._generate_rca_with_claude(
                issue_description, relevant_logs, timeline, patterns, issue_category
            )
            claude_time = (datetime.now() - claude_start).total_seconds()
            timing_breakdown['claude_analysis'] = claude_time
            
            # Print prompt in logs for debugging
            if prompt:
                logger.info("="*80)
                logger.info("FINAL PROMPT SENT TO LLM:")
                logger.info("="*80)
                logger.info(prompt)
                logger.info("="*80)
            
            # Calculate total processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            self.performance_metrics = {
                'processing_time': processing_time,
                'total_logs': len(logs_df),
                'filtered_logs': len(relevant_logs),
                'analysis_mode': analysis_mode,
                'lstm_available': self.lstm_model is not None,
                'vector_db_available': self.vector_db is not None,
                'timing_breakdown': timing_breakdown,
                'query_time_after_response': processing_time  # This is what you asked for
            }
            
            logger.info(f"=== PERFORMANCE BREAKDOWN ===")
            logger.info(f"Total processing time: {processing_time:.3f}s")
            logger.info(f"Data loading: {data_load_time:.3f}s")
            logger.info(f"Categorization: {category_time:.3f}s")
            logger.info(f"Timeline extraction: {timeline_time:.3f}s")
            logger.info(f"Pattern analysis: {pattern_time:.3f}s")
            logger.info(f"Claude analysis: {claude_time:.3f}s")
            logger.info(f"Query time after response: {processing_time:.3f}s")
            
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
                'filtered_logs': relevant_logs,
                'prompt': prompt # Add prompt to results
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
        
        logger.info(f"LSTM filtering: {len(lstm_filtered_logs)} logs in {lstm_time:.3f}s")
        
        # Step 2: Vector DB Semantic Search (CRITICAL: This should use existing data)
        vector_start = datetime.now()
        
        # Check if vector DB has data
        if self.vector_db:
            try:
                # Get vector DB stats to confirm data is available
                stats = self.vector_db.get_collection_stats()
                logger.info(f"Vector DB status: {stats['total_documents']} documents, type: {stats['database_type']}")
                
                if stats['total_documents'] > 0:
                    # Search existing vector database (NO reloading)
                    vector_results = self._vector_db_search(lstm_filtered_logs, issue_description)
                    vector_time = (datetime.now() - vector_start).total_seconds()
                    logger.info(f"Vector DB search: {len(vector_results)} results in {vector_time:.3f}s")
                    
                    # Step 3: Combine and Rank Results
                    if vector_results:
                        combine_start = datetime.now()
                        final_results = self._combine_and_rank_results(lstm_filtered_logs, vector_results)
                        combine_time = (datetime.now() - combine_start).total_seconds()
                        logger.info(f"Combined results: {len(final_results)} final logs in {combine_time:.3f}s")
                        return final_results
                    else:
                        logger.warning("Vector DB search returned no results, using LSTM filtered logs")
                        return lstm_filtered_logs.head(50)
                else:
                    logger.warning("Vector DB has no documents, using LSTM filtered logs")
                    return lstm_filtered_logs.head(50)
                    
            except Exception as e:
                logger.error(f"Vector DB search failed: {e}")
                return lstm_filtered_logs.head(50)
        else:
            logger.warning("Vector DB not available, using LSTM filtered logs")
            return lstm_filtered_logs.head(50)
    
    def get_available_categories(self) -> Dict[str, List[str]]:
        """Get all available issue categories and their keywords"""
        return self.issue_patterns.copy()
    
    def _lstm_filter_logs(self, logs_df: pd.DataFrame, issue_description: str) -> pd.DataFrame:
        """Filter logs by LSTM importance scores with emphasis on ERROR logs"""
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
            
            # Log detailed statistics about filtered results
            logger.info(f"LSTM filtered {len(filtered_logs)} important logs from {len(logs_df)}")
            logger.info(f"Importance score range: {importance_scores.min():.3f} - {importance_scores.max():.3f}")
            
            # Log breakdown by log level
            if 'level' in filtered_logs.columns:
                level_counts = filtered_logs['level'].value_counts()
                logger.info(f"Filtered logs by level: {dict(level_counts)}")
                
                # Check if we have ERROR logs in the filtered results
                error_count = level_counts.get('ERROR', 0)
                if error_count > 0:
                    logger.info(f"✅ LSTM identified {error_count} ERROR logs as important")
                else:
                    logger.warning("⚠️ No ERROR logs identified as important by LSTM")
            
            # Sort by importance score (highest first)
            filtered_logs = filtered_logs.sort_values('lstm_importance', ascending=False)
            
            return filtered_logs
            
        except Exception as e:
            logger.warning(f"LSTM filtering failed: {e}. Using all logs.")
            return logs_df.copy()
    
    def _vector_db_search(self, lstm_filtered_logs: pd.DataFrame, issue_description: str) -> List[Dict]:
        """Search Vector DB for semantically similar logs within LSTM-filtered results"""
        if not self.vector_db or lstm_filtered_logs.empty:
            return []
        
        try:
            # Get the original indices of LSTM-filtered logs to constrain VectorDB search
            lstm_indices = set(lstm_filtered_logs.index.tolist())
            
            # Convert indices to strings for metadata filtering
            lstm_index_strings = [str(idx) for idx in lstm_indices]
            
            # Use correct ChromaDB metadata filtering syntax
            # ChromaDB uses 'where' clause with field names, not nested dictionaries
            where_clause = {'original_index': {'$in': lstm_index_strings}}
            
            similar_logs = self.vector_db.search_similar_logs(
                query=issue_description,
                top_k=50,
                filter_metadata=where_clause
            )
            
            logger.info(f"Vector DB found {len(similar_logs)} similar logs within {len(lstm_filtered_logs)} LSTM-filtered logs")
            
            # Log breakdown of VectorDB results by level
            if similar_logs:
                level_counts = {}
                for log in similar_logs:
                    level = log['metadata'].get('level', 'UNKNOWN')
                    level_counts[level] = level_counts.get(level, 0) + 1
                logger.info(f"VectorDB results by level: {level_counts}")
                
                # Check if ERROR logs are in VectorDB results
                error_count = level_counts.get('ERROR', 0)
                if error_count > 0:
                    logger.info(f"✅ VectorDB found {error_count} ERROR logs in search results")
                else:
                    logger.warning("⚠️ No ERROR logs found in VectorDB search results")
            
            return similar_logs
            
        except Exception as e:
            logger.warning(f"Vector DB search failed: {e}")
            return []
    
    def _combine_and_rank_results(self, lstm_filtered_logs: pd.DataFrame, vector_results: List[Dict]) -> pd.DataFrame:
        """Combine LSTM importance and Vector DB similarity scores"""
        if lstm_filtered_logs.empty:
            return lstm_filtered_logs
        
        try:
            # Create a mapping from VectorDB results to LSTM filtered logs
            # VectorDB results contain original_index in metadata
            vector_logs_by_index = {}
            for vector_log in vector_results:
                original_index = vector_log['metadata'].get('original_index')
                if original_index:
                    try:
                        index = int(original_index)
                        vector_logs_by_index[index] = vector_log
                    except (ValueError, TypeError):
                        continue
            
            # Add VectorDB similarity scores to LSTM filtered logs
            lstm_filtered_logs = lstm_filtered_logs.copy()
            lstm_filtered_logs['vector_similarity'] = 0.0  # Default similarity
            
            for idx in lstm_filtered_logs.index:
                if idx in vector_logs_by_index:
                    vector_log = vector_logs_by_index[idx]
                    distance = vector_log['distance']
                    similarity = vector_log['similarity']
                    
                    # FIX: Clamp similarity to non-negative values for RCA analysis
                    # Negative similarity doesn't make sense - it means the log is semantically opposite
                    # For RCA, we want to treat this as "no semantic match" rather than "negative match"
                    similarity = max(0.0, similarity)
                    
                    lstm_filtered_logs.loc[idx, 'vector_similarity'] = similarity
                    
                    # DEBUG: Show distance and similarity for ERROR logs
                    if 'level' in lstm_filtered_logs.columns and lstm_filtered_logs.loc[idx, 'level'].upper() == 'ERROR':
                        message = lstm_filtered_logs.loc[idx, 'message'][:100]
                        logger.info(f"🔍 DEBUG: ERROR log VectorDB - Distance: {distance:.3f}, Original Similarity: {vector_log['similarity']:.3f}, Clamped: {similarity:.3f} - {message}")
            
            # DEBUG: Show some INFO log scores for comparison
            info_logs = lstm_filtered_logs[lstm_filtered_logs['level'].str.upper() == 'INFO'].head(3)
            for idx, log in info_logs.iterrows():
                if idx in vector_logs_by_index:
                    vector_log = vector_logs_by_index[idx]
                    distance = vector_log['distance']
                    similarity = vector_log['similarity']
                    message = log['message'][:100]
                    logger.info(f"🔍 DEBUG: INFO log VectorDB - Distance: {distance:.3f}, Similarity: {similarity:.3f} - {message}")
            
            # Calculate combined score with special handling for ERROR logs
            if 'lstm_importance' in lstm_filtered_logs.columns:
                # Prioritize LSTM importance for RCA analysis (LSTM learned what's important)
                lstm_filtered_logs['combined_score'] = (
                    lstm_filtered_logs['lstm_importance'] * 0.7 + 
                    lstm_filtered_logs['vector_similarity'] * 0.3
                )
                
                # DEBUG: Print ERROR log scores to understand why they're not ranking high
                if 'level' in lstm_filtered_logs.columns:
                    error_logs = lstm_filtered_logs[lstm_filtered_logs['level'].str.upper() == 'ERROR']
                    if not error_logs.empty:
                        logger.info("🔍 DEBUG: ERROR log scores:")
                        for idx, log in error_logs.iterrows():
                            lstm_score = log.get('lstm_importance', 0)
                            vector_score = log.get('vector_similarity', 0)
                            combined_score = log.get('combined_score', 0)
                            message = log.get('message', '')[:100]
                            logger.info(f"  ERROR: LSTM={lstm_score:.3f}, Vector={vector_score:.3f}, Combined={combined_score:.3f} - {message}")
                
                # Sort by combined score (highest first)
                result_df = lstm_filtered_logs.sort_values('combined_score', ascending=False)
                
                # Log the top results by level
                if 'level' in result_df.columns:
                    top_50 = result_df.head(50)
                    level_counts = top_50['level'].value_counts()
                    logger.info(f"Top 50 results by level: {dict(level_counts)}")
                    
                    # Check if ERROR logs are in top results
                    error_count = level_counts.get('ERROR', 0)
                    if error_count > 0:
                        logger.info(f"✅ {error_count} ERROR logs in top 50 results")
                    else:
                        logger.warning("⚠️ No ERROR logs in top 50 results")
                
                logger.info(f"Combined scoring completed: {len(result_df)} results")
                return result_df.head(50)
            else:
                # Fallback: sort by Vector similarity only
                result_df = lstm_filtered_logs.sort_values('vector_similarity', ascending=False)
                logger.info(f"Vector-only scoring completed: {len(result_df)} results")
                return result_df.head(50)
                
        except Exception as e:
            logger.warning(f"Combined scoring failed: {e}, using LSTM filtered logs")
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
        
        # Priority order for categorization (most specific first)
        priority_categories = [
            'timeout_issues',  # Check timeout issues first
            'resource_shortage',  # Check resource issues before service_failure
            'network_issues',
            'service_failure',
            'authentication',
            'instance_issues',
            'database',
            'storage'
        ]
        
        for category in priority_categories:
            if category in self.issue_patterns:
                keywords = self.issue_patterns[category]
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
            (r'timeout', 'Timeout Occurred'),
            (r'nova-conductor.*timeout', 'Nova-Conductor Timeout'),
            (r'connection.*nova-conductor.*timeout', 'Nova-Conductor Connection Timeout'),
            (r'failed.*update.*instance.*state', 'Instance State Update Failed'),
            (r'messaging.*timeout', 'Messaging Timeout'),
            (r'rpc.*timeout', 'RPC Timeout'),
            (r'connection.*timed.*out', 'Connection Timed Out'),
            (r'update.*instance.*state.*failed', 'Instance State Update Failed')
        ]
        
        for _, log in sorted_logs.iterrows():
            message = str(log.get('message', '')).lower()
            
            for pattern, event_type in key_event_patterns:
                if re.search(pattern, message):
                    timeline.append({
                        'timestamp': log.get('timestamp'),
                        'event_type': event_type,  # Fixed: use 'event_type' consistently
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
        elif issue_category == 'timeout_issues':
            patterns['timeout_patterns'] = self._analyze_network_patterns(logs_df)  # Reuse network patterns for timeouts
        elif issue_category == 'service_failure':
            patterns['service_patterns'] = self._analyze_network_patterns(logs_df)  # Include timeout patterns for service failures
        
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
            'timeout_issues': len(logs_df[logs_df['message'].str.contains('timeout|timed out', case=False, na=False)]),
            'nova_conductor_timeouts': len(logs_df[logs_df['message'].str.contains('nova-conductor.*timeout', case=False, na=False)]),
            'messaging_timeouts': len(logs_df[logs_df['message'].str.contains('messaging.*timeout|rpc.*timeout', case=False, na=False)]),
            'instance_state_update_failures': len(logs_df[logs_df['message'].str.contains('failed.*update.*instance.*state|update.*instance.*state.*failed', case=False, na=False)]),
            'dns_issues': len(logs_df[logs_df['message'].str.contains('dns|hostname', case=False, na=False)]),
            'network_unreachable': len(logs_df[logs_df['message'].str.contains('unreachable|network', case=False, na=False)])
        }
        return network_patterns
    
    def _generate_rca_with_claude(self, issue_description: str, logs_df: pd.DataFrame, 
                                 timeline: List[Dict], patterns: Dict, issue_category: str) -> Tuple[str, str]:
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
                return response, prompt
            else:
                fallback_response = self._generate_fallback_rca(issue_category, patterns, logs_df, timeline)
                return fallback_response, prompt
                
        except Exception as e:
            logger.error(f"Failed to generate RCA with Claude: {e}")
            fallback_response = self._generate_fallback_rca(issue_category, patterns, logs_df, timeline)
            # Return the prompt even when API fails
            try:
                context = self._prepare_claude_context(issue_description, logs_df, timeline, patterns, issue_category)
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
                return fallback_response, prompt
            except:
                return fallback_response, ""
    
    def _prepare_claude_context(self, issue_description: str, logs_df: pd.DataFrame,
                               timeline: List[Dict], patterns: Dict, issue_category: str) -> str:
        """Prepare context for Claude API"""
        context_parts = []
        
        # Add issue description
        context_parts.append(f"Issue Description: {issue_description}")
        context_parts.append(f"Issue Category: {issue_category}")
        context_parts.append("")
        
        # Add historical context from VectorDB (NEW)
        if self.vector_db:
            try:
                logger.info("🔍 Attempting to get historical context from VectorDB...")
                # Get historical context from VectorDB
                historical_context = self.vector_db.get_context_for_issue(
                    issue_description, 
                    top_k=Config.RCA_CONFIG['historical_context_size']
                )
                if historical_context:
                    logger.info(f"✅ Found historical context: {len(historical_context)} characters")
                    context_parts.append("Historical Context (Similar Issues):")
                    context_parts.append(historical_context)
                    context_parts.append("")
                else:
                    logger.warning("⚠️ No historical context found - VectorDB may be empty or no similar issues")
            except Exception as e:
                logger.warning(f"❌ Failed to get historical context: {e}")
        else:
            logger.info("ℹ️ VectorDB not available - skipping historical context")
        
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
                context_parts.append(f"- {event['timestamp']}: {event['event_type']} ({event['service']})")
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
                rca_parts.append(f"- {event['timestamp']}: {event['event_type']}")
        
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
        elif issue_category == 'timeout_issues':
            recommendations.extend([
                "Check nova-conductor service status and connectivity",
                "Verify messaging/RPC timeout configurations",
                "Monitor system load and resource usage",
                "Review network latency between compute and conductor nodes",
                "Check for database connection issues affecting conductor",
                "Consider increasing timeout values if appropriate",
                "Verify message queue (RabbitMQ/ZeroMQ) health"
            ])
        elif issue_category == 'service_failure':
            recommendations.extend([
                "Check service status and restart if necessary",
                "Verify service dependencies and connectivity",
                "Review service logs for additional errors",
                "Monitor system resources for service constraints"
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
            'filtered_logs': logs_df.head(10) if not logs_df.empty else pd.DataFrame(),
            'prompt': f"Analysis failed for issue: {issue_description}. No prompt generated due to analysis failure."
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics from last analysis"""
        return self.performance_metrics 