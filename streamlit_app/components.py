import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re

class LogVisualizationComponents:
    """Reusable visualization components for log analysis"""
    
    @staticmethod
    def render_log_metrics_cards(df: pd.DataFrame):
        """Render key metrics cards"""
        if df.empty:
            st.info("No data available for metrics")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_logs = len(df)
            st.metric(
                label="📝 Total Logs",
                value=f"{total_logs:,}",
                help="Total number of log entries"
            )
        
        with col2:
            error_count = len(df[df['level'].str.upper() == 'ERROR']) if 'level' in df.columns else 0
            error_rate = (error_count / total_logs * 100) if total_logs > 0 else 0
            st.metric(
                label="🚨 Errors",
                value=error_count,
                delta=f"{error_rate:.1f}% of total",
                help="Number and percentage of error-level logs"
            )
        
        with col3:
            unique_instances = df['instance_id'].nunique() if 'instance_id' in df.columns else 0
            st.metric(
                label="🖥️ Instances",
                value=unique_instances,
                help="Number of unique instances mentioned in logs"
            )
        
        with col4:
            unique_services = df['service_type'].nunique() if 'service_type' in df.columns else 0
            st.metric(
                label="⚙️ Services",
                value=unique_services,
                help="Number of different OpenStack services"
            )
    
    @staticmethod
    def render_log_level_distribution(df: pd.DataFrame):
        """Render log level distribution chart"""
        if df.empty or 'level' not in df.columns:
            st.info("No log level data available")
            return
        
        level_counts = df['level'].value_counts()
        
        # Create color mapping for log levels
        color_map = {
            'ERROR': '#ff4444',
            'CRITICAL': '#cc0000',
            'WARNING': '#ff8800',
            'INFO': '#0088cc',
            'DEBUG': '#888888'
        }
        
        colors = [color_map.get(level, '#888888') for level in level_counts.index]
        
        fig = px.pie(
            values=level_counts.values,
            names=level_counts.index,
            title="Log Level Distribution",
            color_discrete_sequence=colors
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_service_distribution(df: pd.DataFrame):
        """Render service distribution chart"""
        if df.empty or 'service_type' not in df.columns:
            st.info("No service data available")
            return
        
        service_counts = df['service_type'].value_counts().head(10)
        
        fig = px.bar(
            x=service_counts.values,
            y=service_counts.index,
            orientation='h',
            title="Top 10 Services by Log Count",
            labels={'x': 'Number of Logs', 'y': 'Service Type'}
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_timeline_chart(df: pd.DataFrame):
        """Render log timeline chart"""
        if df.empty or 'timestamp' not in df.columns:
            st.info("No timestamp data available for timeline")
            return
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by hour and log level
        df['hour'] = df['timestamp'].dt.floor('1h')
        timeline_data = df.groupby(['hour', 'level']).size().reset_index(name='count')
        
        fig = px.line(
            timeline_data,
            x='hour',
            y='count',
            color='level',
            title="Log Activity Timeline",
            labels={'hour': 'Time', 'count': 'Number of Logs'}
        )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Number of Logs",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_instance_activity_heatmap(df: pd.DataFrame):
        """Render instance activity heatmap"""
        if df.empty or 'instance_id' not in df.columns or 'timestamp' not in df.columns:
            st.info("No instance activity data available")
            return
        
        # Filter out null instance IDs and limit to top 20 instances
        instance_df = df[df['instance_id'].notna()].copy()
        if instance_df.empty:
            st.info("No instance activity data available")
            return
        
        top_instances = instance_df['instance_id'].value_counts().head(20).index
        instance_df = instance_df[instance_df['instance_id'].isin(top_instances)]
        
        # Convert timestamp and create hour bins
        instance_df['timestamp'] = pd.to_datetime(instance_df['timestamp'])
        instance_df['hour'] = instance_df['timestamp'].dt.hour
        
        # Create pivot table for heatmap
        heatmap_data = instance_df.groupby(['instance_id', 'hour']).size().reset_index(name='activity')
        heatmap_pivot = heatmap_data.pivot(index='instance_id', columns='hour', values='activity').fillna(0)
        
        fig = px.imshow(
            heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=[f"...{id[-8:]}" for id in heatmap_pivot.index],  # Show last 8 chars of instance ID
            title="Instance Activity Heatmap (by Hour)",
            labels={'x': 'Hour of Day', 'y': 'Instance ID', 'color': 'Log Count'},
            aspect='auto'
        )
        
        fig.update_layout(height=max(300, len(top_instances) * 20))
        st.plotly_chart(fig, use_container_width=True)

class LogFilterComponents:
    """Reusable filter components for log analysis"""
    
    @staticmethod
    def render_time_filter(df: pd.DataFrame) -> Optional[tuple]:
        """Render time range filter"""
        if df.empty or 'timestamp' not in df.columns:
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        return (start_date, end_date)
    
    @staticmethod
    def render_level_filter(df: pd.DataFrame) -> List[str]:
        """Render log level filter"""
        if df.empty or 'level' not in df.columns:
            return []
        
        available_levels = sorted(df['level'].unique())
        selected_levels = st.multiselect(
            "Select Log Levels",
            options=available_levels,
            default=available_levels,
            help="Filter logs by severity level"
        )
        
        return selected_levels
    
    @staticmethod
    def render_service_filter(df: pd.DataFrame) -> List[str]:
        """Render service type filter"""
        if df.empty or 'service_type' not in df.columns:
            return []
        
        available_services = sorted(df['service_type'].unique())
        selected_services = st.multiselect(
            "Select Services",
            options=available_services,
            default=available_services,
            help="Filter logs by OpenStack service"
        )
        
        return selected_services
    
    @staticmethod
    def render_text_search(placeholder: str = "Search in log messages...") -> str:
        """Render text search filter"""
        search_term = st.text_input(
            "Search Messages",
            placeholder=placeholder,
            help="Search for specific text in log messages (case-insensitive)"
        )
        
        return search_term
    
    @staticmethod
    def render_instance_filter(df: pd.DataFrame) -> List[str]:
        """Render instance ID filter"""
        if df.empty or 'instance_id' not in df.columns:
            return []
        
        # Get instances with non-null IDs
        instance_df = df[df['instance_id'].notna()]
        if instance_df.empty:
            return []
        
        available_instances = sorted(instance_df['instance_id'].unique())
        
        # Show abbreviated instance IDs for better UX
        instance_options = {f"...{id[-12:]}": id for id in available_instances}
        
        selected_display = st.multiselect(
            "Select Instances",
            options=list(instance_options.keys()),
            help="Filter logs by specific instance IDs"
        )
        
        # Convert back to full instance IDs
        selected_instances = [instance_options[display_id] for display_id in selected_display]
        
        return selected_instances

class RCADisplayComponents:
    """Components for displaying RCA analysis results"""
    
    @staticmethod
    def render_analysis_summary(results: Dict):
        """Render RCA analysis summary"""
        st.subheader("🎯 Analysis Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Issue Category",
                results.get('issue_category', 'Unknown').title(),
                help="Automatically categorized issue type"
            )
        
        with col2:
            st.metric(
                "Relevant Logs",
                results.get('relevant_logs_count', 0),
                help="Number of logs identified as relevant to the issue"
            )
        
        with col3:
            confidence_score = results.get('confidence_score', 0.5)
            st.metric(
                "Confidence",
                f"{confidence_score:.1%}",
                help="Confidence level of the analysis"
            )
    
    @staticmethod
    def render_root_cause_analysis(analysis_text: str):
        """Render the main RCA text"""
        st.subheader("🔍 Root Cause Analysis")
        
        # Clean and format the analysis text
        formatted_text = RCADisplayComponents._format_analysis_text(analysis_text)
        st.markdown(formatted_text)
    
    @staticmethod
    def render_timeline(timeline: List[Dict]):
        """Render event timeline"""
        if not timeline:
            return
        
        st.subheader("⏱️ Event Timeline")
        
        # Create DataFrame for better display
        timeline_df = pd.DataFrame(timeline)
        
        # Format timestamp for display
        if 'timestamp' in timeline_df.columns:
            timeline_df['formatted_time'] = pd.to_datetime(timeline_df['timestamp']).dt.strftime('%H:%M:%S')
        
        # Display in an interactive table
        st.dataframe(
            timeline_df[['formatted_time', 'event_type', 'service', 'level']],
            column_config={
                'formatted_time': 'Time',
                'event_type': 'Event',
                'service': 'Service',
                'level': 'Level'
            },
            hide_index=True,
            use_container_width=True
        )
    
    @staticmethod
    def render_patterns_analysis(patterns: Dict):
        """Render patterns analysis"""
        if not patterns:
            return
        
        st.subheader("📊 Detected Patterns")
        
        # Create tabs for different pattern types
        pattern_tabs = st.tabs(["Errors", "Services", "Resources", "Timeline"])
        
        with pattern_tabs[0]:
            if patterns.get('error_frequency'):
                st.write("**Error Frequency Analysis:**")
                error_df = pd.DataFrame(
                    list(patterns['error_frequency'].items()),
                    columns=['Error Type', 'Frequency']
                ).sort_values('Frequency', ascending=False)
                
                fig = px.bar(error_df, x='Error Type', y='Frequency', title="Error Types by Frequency")
                st.plotly_chart(fig, use_container_width=True)
        
        with pattern_tabs[1]:
            if patterns.get('service_distribution'):
                st.write("**Service Distribution:**")
                service_df = pd.DataFrame(
                    list(patterns['service_distribution'].items()),
                    columns=['Service', 'Log Count']
                ).sort_values('Log Count', ascending=False)
                
                fig = px.pie(service_df, values='Log Count', names='Service', title="Logs by Service")
                st.plotly_chart(fig, use_container_width=True)
        
        with pattern_tabs[2]:
            if patterns.get('resource_patterns'):
                st.write("**Resource-related Mentions:**")
                resource_df = pd.DataFrame(
                    list(patterns['resource_patterns'].items()),
                    columns=['Resource Type', 'Mentions']
                ).sort_values('Mentions', ascending=False)
                
                fig = px.bar(resource_df, x='Resource Type', y='Mentions', title="Resource Mentions")
                st.plotly_chart(fig, use_container_width=True)
        
        with pattern_tabs[3]:
            if patterns.get('time_patterns'):
                st.write("**Time-based Patterns:**")
                time_df = pd.DataFrame(
                    list(patterns['time_patterns'].items()),
                    columns=['Hour', 'Activity']
                ).sort_values('Hour')
                
                fig = px.line(time_df, x='Hour', y='Activity', title="Activity by Hour of Day")
                st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_recommendations(recommendations: List[str]):
        """Render actionable recommendations"""
        if not recommendations:
            return
        
        st.subheader("💡 Recommendations")
        
        # Categorize recommendations
        immediate_actions = []
        short_term_fixes = []
        long_term_solutions = []
        
        for rec in recommendations:
            if any(word in rec.lower() for word in ['immediate', 'urgent', 'now', 'asap']):
                immediate_actions.append(rec)
            elif any(word in rec.lower() for word in ['short', 'quick', 'temporary']):
                short_term_fixes.append(rec)
            else:
                long_term_solutions.append(rec)
        
        # If categorization didn't work well, just show all as general recommendations
        if not immediate_actions and not short_term_fixes and not long_term_solutions:
            st.write("**Recommended Actions:**")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            # Show categorized recommendations
            if immediate_actions:
                st.write("🚨 **Immediate Actions:**")
                for i, rec in enumerate(immediate_actions, 1):
                    st.write(f"{i}. {rec}")
            
            if short_term_fixes:
                st.write("⚡ **Short-term Fixes:**")
                for i, rec in enumerate(short_term_fixes, 1):
                    st.write(f"{i}. {rec}")
            
            if long_term_solutions:
                st.write("🏗️ **Long-term Solutions:**")
                for i, rec in enumerate(long_term_solutions, 1):
                    st.write(f"{i}. {rec}")
    
    @staticmethod
    def _format_analysis_text(text: str) -> str:
        """Format RCA analysis text for better display"""
        if not text:
            return "No analysis available."
        
        # Clean up the text
        formatted = text.strip()
        
        # Add proper markdown formatting for headers
        formatted = re.sub(r'\*\*(.*?)\*\*', r'**\1**', formatted)
        formatted = re.sub(r'#{1,3}\s*(.*)', r'### \1', formatted)
        
        # Ensure proper line breaks
        formatted = re.sub(r'\n\n+', '\n\n', formatted)
        
        return formatted

class ModelTrainingComponents:
    """Components for model training interface"""
    
    @staticmethod
    def render_training_config():
        """Render training configuration interface"""
        st.subheader("🔧 Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider(
                "Training Epochs",
                min_value=10,
                max_value=100,
                value=50,
                step=5,
                help="Number of training epochs"
            )
            
            batch_size = st.selectbox(
                "Batch Size",
                options=[16, 32, 64, 128],
                index=1,
                help="Training batch size"
            )
        
        with col2:
            lstm_units = st.slider(
                "LSTM Units",
                min_value=32,
                max_value=256,
                value=64,
                step=16,
                help="Number of LSTM units in each layer"
            )
            
            dropout_rate = st.slider(
                "Dropout Rate",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Dropout rate for regularization"
            )
        
        return {
            'epochs': epochs,
            'batch_size': batch_size,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate
        }
    
    @staticmethod
    def render_training_progress(progress_value: float, status_text: str):
        """Render training progress bar"""
        progress_bar = st.progress(progress_value)
        status_placeholder = st.empty()
        status_placeholder.text(status_text)
        
        return progress_bar, status_placeholder
    
    @staticmethod
    def render_training_results(results: Dict):
        """Render training results and metrics"""
        st.subheader("📈 Training Results")
        
        # Metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Training Accuracy",
                f"{results.get('train_accuracy', 0):.3f}",
                help="Accuracy on training set"
            )
        
        with col2:
            st.metric(
                "Validation Accuracy",
                f"{results.get('val_accuracy', 0):.3f}",
                help="Accuracy on validation set"
            )
        
        with col3:
            st.metric(
                "Training Precision",
                f"{results.get('train_precision', 0):.3f}",
                help="Precision on training set"
            )
        
        with col4:
            st.metric(
                "Validation Precision",
                f"{results.get('val_precision', 0):.3f}",
                help="Precision on validation set"
            )
        
        # Classification report
        if 'classification_report' in results:
            st.subheader("📊 Detailed Classification Report")
            st.text(results['classification_report'])
        
        # Confusion matrix
        if 'confusion_matrix' in results:
            st.subheader("🔄 Confusion Matrix")
            cm = results['confusion_matrix']
            
            fig = px.imshow(
                cm,
                text_auto=True,
                title="Confusion Matrix",
                labels={'x': 'Predicted', 'y': 'Actual'},
                x=['Not Important', 'Important'],
                y=['Not Important', 'Important']
            )
            
            st.plotly_chart(fig, use_container_width=True)