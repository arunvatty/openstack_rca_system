"""
Simple Working Monitoring Integration for OpenStack RCA System
"""

import os
import time
import logging
import threading
import json
from datetime import datetime
from typing import Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, start_http_server
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class PrometheusMetrics:
    """Prometheus metrics for RCA system"""
    
    def __init__(self):
        # Use default registry for simplicity
        
        # RCA Analysis metrics
        self.rca_analysis_total = Counter(
            'rca_analysis_total',
            'Total number of RCA analyses performed',
            ['status', 'category', 'mode']
        )
        
        self.rca_analysis_duration = Histogram(
            'rca_analysis_duration_seconds',
            'Time spent on RCA analysis',
            ['mode']
        )
        
        # Model metrics
        self.model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
        self.model_precision = Gauge('model_precision', 'Current model precision')
        self.model_recall = Gauge('model_recall', 'Current model recall')
        
        # System metrics
        self.active_users = Gauge('active_users', 'Number of active users')
        self.vector_db_documents = Gauge('vector_db_documents_total', 'Number of documents in vector database')
        
        # Performance metrics
        self.lstm_inference_duration = Histogram('lstm_inference_duration_seconds', 'LSTM inference time')
        self.user_satisfaction_score = Histogram('user_satisfaction_score', 'User satisfaction ratings', buckets=[1, 2, 3, 4, 5])

class SimpleMetricsStorage:
    """Simple in-memory storage for metrics data"""
    
    def __init__(self):
        self.training_runs = []
        self.rca_analyses = []
        self.performance_data = []
    
    def log_training_run(self, config: Dict, metrics: Dict):
        """Store training run data"""
        run_data = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': metrics,
            'run_id': f"run_{len(self.training_runs) + 1}"
        }
        self.training_runs.append(run_data)
        logger.info(f"Training run logged: {run_data['run_id']}")
    
    def log_rca_analysis(self, issue_description: str, results: Dict):
        """Store RCA analysis data"""
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'issue_description': issue_description[:100] if issue_description else 'Unknown',
            'category': results.get('issue_category', 'unknown'),
            'mode': results.get('analysis_mode', 'unknown'),
            'relevant_logs': results.get('relevant_logs_count', 0),
            'processing_time': results.get('performance_metrics', {}).get('processing_time', 0) if isinstance(results.get('performance_metrics'), dict) else 0
        }
        self.rca_analyses.append(analysis_data)
        logger.info(f"RCA analysis logged: {analysis_data['category']}")
    
    def get_training_runs(self):
        """Get all training runs as DataFrame"""
        if not self.training_runs:
            return pd.DataFrame()
        return pd.DataFrame(self.training_runs)
    
    def get_rca_analyses(self):
        """Get all RCA analyses as DataFrame"""
        if not self.rca_analyses:
            return pd.DataFrame()
        return pd.DataFrame(self.rca_analyses)

class MonitoringManager:
    """Main monitoring manager"""
    
    def __init__(self):
        self.metrics = PrometheusMetrics()
        self.storage = SimpleMetricsStorage()
        self.active_users = set()
        self.metrics_server_started = False
        
        # Start metrics server
        self._start_metrics_server()
        
        # Update system metrics periodically
        self._start_system_metrics_updater()
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server"""
        try:
            # Use prometheus_client's built-in server
            start_http_server(8000)
            self.metrics_server_started = True
            logger.info("Prometheus metrics server started on port 8000")
            print("✅ Metrics server started on http://localhost:8000/metrics")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            print(f"❌ Failed to start metrics server: {e}")
            # Try alternative port
            try:
                start_http_server(8001)
                self.metrics_server_started = True
                logger.info("Prometheus metrics server started on port 8001")
                print("✅ Metrics server started on http://localhost:8001/metrics")
            except Exception as e2:
                logger.error(f"Failed to start metrics server on alternative port: {e2}")
    
    def _start_system_metrics_updater(self):
        """Start background thread to update system metrics"""
        def update_metrics():
            while True:
                try:
                    # Update active users count
                    self.metrics.active_users.set(len(self.active_users))
                    
                    # Update vector DB metrics if available
                    try:
                        from services.vector_db_service import VectorDBService
                        vector_db = VectorDBService()
                        stats = vector_db.get_collection_stats()
                        self.metrics.vector_db_documents.set(stats.get('total_documents', 0))
                    except Exception as e:
                        # Set to 0 if VectorDB not available
                        self.metrics.vector_db_documents.set(0)
                    
                    time.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    logger.error(f"Error updating system metrics: {e}")
                    time.sleep(30)
        
        thread = threading.Thread(target=update_metrics, daemon=True)
        thread.start()
    
    def track_user_session(self, user_id: str):
        """Track user session"""
        self.active_users.add(user_id)
        logger.info(f"User session started: {user_id}")
    
    def end_user_session(self, user_id: str):
        """End user session"""
        self.active_users.discard(user_id)
        logger.info(f"User session ended: {user_id}")
    
    def record_user_satisfaction(self, score: int):
        """Record user satisfaction score (1-5)"""
        if 1 <= score <= 5:
            self.metrics.user_satisfaction_score.observe(score)
            logger.info(f"User satisfaction recorded: {score}/5")
    
    def log_training_metrics(self, results: Dict):
        """Log training metrics"""
        try:
            self.metrics.model_accuracy.set(results.get('val_accuracy', 0))
            self.metrics.model_precision.set(results.get('val_precision', 0))
            self.metrics.model_recall.set(results.get('val_recall', 0))
            logger.info("Training metrics updated")
        except Exception as e:
            logger.error(f"Failed to log training metrics: {e}")
    
    def log_rca_metrics(self, issue_description: str, results: Dict, processing_time: float, mode: str = 'hybrid'):
        """Log RCA analysis metrics"""
        try:
            category = results.get('issue_category', 'unknown')
            status = 'success' if results.get('root_cause_analysis') else 'error'
            
            # Record Prometheus metrics
            self.metrics.rca_analysis_total.labels(
                status=status, category=category, mode=mode
            ).inc()
            
            self.metrics.rca_analysis_duration.labels(mode=mode).observe(processing_time)
            
            # Store in local storage
            self.storage.log_rca_analysis(issue_description, results)
            
            logger.info(f"RCA metrics logged: {status}, {category}, {processing_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to log RCA metrics: {e}")

# Global monitoring manager instance
monitoring_manager = None

def get_monitoring_manager():
    """Get or create monitoring manager"""
    global monitoring_manager
    if monitoring_manager is None:
        monitoring_manager = MonitoringManager()
    return monitoring_manager

def integrate_monitoring_with_main():
    """Integration function for main.py"""
    return get_monitoring_manager()

def integrate_monitoring_with_streamlit_app(app_instance):
    """Integration function for streamlit app"""
    import streamlit as st
    
    manager = get_monitoring_manager()
    
    # Track user session
    if 'user_session_id' not in st.session_state:
        import uuid
        st.session_state.user_session_id = str(uuid.uuid4())
        manager.track_user_session(st.session_state.user_session_id)
    
    # Add evaluation dashboard to app
    app_instance.evaluation_dashboard = EvaluationDashboard(manager)
    
    return manager

class EvaluationDashboard:
    """Evaluation metrics dashboard for Streamlit"""
    
    def __init__(self, monitoring_manager: MonitoringManager):
        self.monitoring_manager = monitoring_manager
    
    def render_evaluation_dashboard(self):
        """Render the evaluation metrics dashboard"""
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        
        st.header("🎯 Model Evaluation & Performance Metrics")
        
        # Show metrics server status
        if self.monitoring_manager.metrics_server_started:
            st.success("✅ Metrics server is running")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 View Prometheus Metrics"):
                    try:
                        import requests
                        response = requests.get("http://localhost:8000/metrics", timeout=2)
                        if response.status_code == 200:
                            st.code(response.text[:2000] + "..." if len(response.text) > 2000 else response.text)
                        else:
                            st.error("Failed to fetch metrics")
                    except Exception as e:
                        st.error(f"Error fetching metrics: {e}")
            with col2:
                st.info("Metrics available at: http://localhost:8000/metrics")
        else:
            st.error("❌ Metrics server failed to start")
        
        # Training Runs Section
        st.subheader("📊 Training History")
        
        training_df = self.monitoring_manager.storage.get_training_runs()
        if not training_df.empty:
            st.write("**Recent Training Runs:**")
            st.dataframe(training_df, use_container_width=True)
            
            # Plot training metrics if available
            if not training_df.empty and 'metrics' in training_df.columns:
                try:
                    # Extract accuracy from metrics
                    accuracies = []
                    timestamps = []
                    for idx, row in training_df.iterrows():
                        if isinstance(row['metrics'], dict) and 'val_accuracy' in row['metrics']:
                            accuracies.append(row['metrics']['val_accuracy'])
                            timestamps.append(row['timestamp'])
                    
                    if accuracies:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=timestamps, y=accuracies, mode='lines+markers', name='Validation Accuracy'))
                        fig.update_layout(title="Model Accuracy Over Time", xaxis_title="Time", yaxis_title="Accuracy")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot training metrics: {e}")
        else:
            st.info("No training runs found. Train a model to see metrics here.")
        
        # RCA Analysis Section
        st.subheader("🔍 RCA Analysis History")
        
        rca_df = self.monitoring_manager.storage.get_rca_analyses()
        if not rca_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Category distribution
                if 'category' in rca_df.columns:
                    category_counts = rca_df['category'].value_counts()
                    fig = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title="Analysis Categories"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Processing time distribution
                if 'processing_time' in rca_df.columns:
                    fig = px.histogram(
                        rca_df,
                        x='processing_time',
                        title="Processing Time Distribution",
                        labels={'processing_time': 'Processing Time (seconds)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Recent analyses table
            st.write("**Recent Analyses:**")
            st.dataframe(rca_df.tail(10), use_container_width=True)
        else:
            st.info("No RCA analyses found. Run some analyses to see metrics here.")
        
        # Real-time Metrics Section
        st.subheader("⚡ Real-time Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Users", len(self.monitoring_manager.active_users))
        
        with col2:
            try:
                accuracy = self.monitoring_manager.metrics.model_accuracy._value._value
                st.metric("Model Accuracy", f"{accuracy:.3f}")
            except:
                st.metric("Model Accuracy", "N/A")
        
        with col3:
            try:
                precision = self.monitoring_manager.metrics.model_precision._value._value
                st.metric("Model Precision", f"{precision:.3f}")
            except:
                st.metric("Model Precision", "N/A")
        
        with col4:
            try:
                recall = self.monitoring_manager.metrics.model_recall._value._value
                st.metric("Model Recall", f"{recall:.3f}")
            except:
                st.metric("Model Recall", "N/A")
        
        # User Satisfaction Section
        st.subheader("😊 User Satisfaction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Rate this analysis session:**")
            satisfaction_score = st.select_slider(
                "How satisfied are you with the RCA results?",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x
            )
            
            if st.button("Submit Rating"):
                self.monitoring_manager.record_user_satisfaction(satisfaction_score)
                st.success("Thank you for your feedback!")
        
        with col2:
            st.write("**System Status:**")
            
            # Vector DB Status
            try:
                from services.vector_db_service import VectorDBService
                vector_db = VectorDBService()
                stats = vector_db.get_collection_stats()
                
                st.metric("Vector DB Documents", stats.get('total_documents', 0))
                
                if stats.get('total_documents', 0) > 0:
                    st.success("✅ Vector DB is healthy")
                else:
                    st.warning("⚠️ Vector DB is empty")
                    
            except Exception as e:
                st.error(f"❌ Vector DB unavailable: {e}")
        
        # Export functionality
        st.subheader("📤 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Training Data"):
                training_df = self.monitoring_manager.storage.get_training_runs()
                if not training_df.empty:
                    csv = training_df.to_csv(index=False)
                    st.download_button(
                        label="Download Training Data CSV",
                        data=csv,
                        file_name=f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No training data to export")
        
        with col2:
            if st.button("📈 Export RCA Data"):
                rca_df = self.monitoring_manager.storage.get_rca_analyses()
                if not rca_df.empty:
                    csv = rca_df.to_csv(index=False)
                    st.download_button(
                        label="Download RCA Data CSV",
                        data=csv,
                        file_name=f"rca_analyses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No RCA data to export")

# Monkey patch functions for existing code
def enhance_train_model_pipeline(original_train_function):
    """Enhance training pipeline with monitoring"""
    def monitored_train(*args, **kwargs):
        start_time = time.time()
        try:
            result = original_train_function(*args, **kwargs)
            
            # Log metrics if training was successful
            if result and hasattr(result, 'history'):
                # Extract final metrics
                history = result.history.history if hasattr(result.history, 'history') else {}
                final_metrics = {}
                
                if 'val_accuracy' in history:
                    final_metrics['val_accuracy'] = history['val_accuracy'][-1]
                if 'val_precision' in history:
                    final_metrics['val_precision'] = history['val_precision'][-1]
                if 'val_recall' in history:
                    final_metrics['val_recall'] = history['val_recall'][-1]
                
                # Log to monitoring
                manager = get_monitoring_manager()
                manager.log_training_metrics(final_metrics)
                manager.storage.log_training_run({}, final_metrics)
            
            return result
        except Exception as e:
            logger.error(f"Training monitoring failed: {e}")
            return original_train_function(*args, **kwargs)
    
    return monitored_train

def enhance_rca_analysis(original_analyze_function):
    """Enhance RCA analysis with monitoring"""
    def monitored_analyze(issue_description, logs_df, fast_mode=False):
        start_time = time.time()
        mode = 'fast' if fast_mode else 'hybrid'
        
        try:
            result = original_analyze_function(issue_description, logs_df, fast_mode)
            processing_time = time.time() - start_time
            
            # Log metrics
            manager = get_monitoring_manager()
            manager.log_rca_metrics(issue_description, result, processing_time, mode)
            
            return result
        except Exception as e:
            # Log error
            manager = get_monitoring_manager()
            manager.metrics.rca_analysis_total.labels(
                status='error', category='unknown', mode=mode
            ).inc()
            
            logger.error(f"RCA analysis monitoring failed: {e}")
            return original_analyze_function(issue_description, logs_df, fast_mode)
    
    return monitored_analyze