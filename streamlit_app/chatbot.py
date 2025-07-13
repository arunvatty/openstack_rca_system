import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from data.log_ingestion import LogIngestionManager
from data.preprocessing import LogPreprocessor
from models.lstm_classifier import LSTMLogClassifier
from models.rca_analyzer import RCAAnalyzer
from utils.feature_engineering import FeatureEngineer

# Configure Streamlit page
st.set_page_config(
    page_title=Config.STREAMLIT_CONFIG['page_title'],
    page_icon=Config.STREAMLIT_CONFIG['page_icon'],
    layout=Config.STREAMLIT_CONFIG['layout']
)

class OpenStackRCAAssistant:
    """Main Streamlit application for OpenStack RCA"""
    
    def __init__(self):
        self.config = Config()
        self.ingestion_manager = LogIngestionManager(self.config.DATA_DIR)
        self.preprocessor = LogPreprocessor()
        self.feature_engineer = FeatureEngineer()
        
        # Initialize session state
        if 'logs_df' not in st.session_state:
            st.session_state.logs_df = pd.DataFrame()
        if 'lstm_model' not in st.session_state:
            st.session_state.lstm_model = None
        if 'rca_analyzer' not in st.session_state:
            st.session_state.rca_analyzer = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'vector_db' not in st.session_state:
            st.session_state.vector_db = None
    
    def run(self):
        """Main application runner"""
        st.title("🔍 CloudTracer RCA Assistant")
        st.markdown("*Intelligent Root Cause Analysis for OpenStack Environments*")
        
        # Sidebar for configuration and data management
        self.render_sidebar()
        
        # Main content area
        if st.session_state.logs_df.empty:
            self.render_data_upload_section()
        else:
            # Show tabs for different functionalities
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Dashboard", 
                "🤖 RCA Chat", 
                "📈 Log Analysis", 
                "⚙️ Model Training"
            ])
            
            with tab1:
                self.render_dashboard()
            
            with tab2:
                self.render_chat_interface()
            
            with tab3:
                self.render_log_analysis()
            
            with tab4:
                self.render_model_training()
    
    def render_sidebar(self):
        """Render sidebar with configuration options"""
        st.sidebar.header("Configuration")
        
        # API Key configuration
        api_key = st.sidebar.text_input(
            "Anthropic API Key", 
            type="password",
            value=os.getenv('ANTHROPIC_API_KEY', ''),
            help="Enter your Anthropic API key for Claude integration"
        )
        
        if api_key:
            os.environ['ANTHROPIC_API_KEY'] = api_key
            if st.session_state.rca_analyzer is None:
                try:
                    st.session_state.rca_analyzer = RCAAnalyzer(
                        api_key, 
                        st.session_state.lstm_model
                    )
                    st.sidebar.success("✅ Claude API connected")
                except Exception as e:
                    st.sidebar.error(f"❌ Failed to connect: {e}")
        
        # Global prompt display toggle
        st.sidebar.subheader("Display Options")
        st.session_state.show_prompt_global = st.sidebar.checkbox(
            "Always show LLM prompts",
            value=st.session_state.get('show_prompt_global', False),
            help="Show the final prompt sent to the AI model for all analyses"
        )
        
        # Data source information
        st.sidebar.subheader("Data Source")
        
        # Check if VectorDB is available and has data
        vector_db_available = hasattr(st.session_state, 'vector_db') and st.session_state.vector_db is not None
        if vector_db_available:
            try:
                vector_db_count = st.session_state.vector_db.collection.count()
                st.sidebar.success(f"✅ VectorDB: {vector_db_count} documents")
                st.sidebar.info("RCA analysis uses VectorDB data")
            except:
                vector_db_available = False
        
        # Show file system data count
        if not st.session_state.logs_df.empty:
            file_count = len(st.session_state.logs_df)
            if vector_db_available:
                st.sidebar.warning(f"⚠️ File System: {file_count} logs")
                st.sidebar.info("UI shows file data, RCA uses VectorDB")
                
                # Add sync button
                if st.sidebar.button("🔄 Sync with VectorDB"):
                    self._sync_with_vector_db()
            else:
                st.sidebar.metric("Total Log Entries", file_count)
            
            # Data statistics
            if 'timestamp' in st.session_state.logs_df.columns:
                date_range = st.session_state.logs_df['timestamp'].agg(['min', 'max'])
                st.sidebar.metric("Date Range", f"{date_range['min'].date()} to {date_range['max'].date()}")
            
            # Clear data button
            if st.sidebar.button("🗑️ Clear Data"):
                st.session_state.logs_df = pd.DataFrame()
                st.session_state.lstm_model = None
                st.rerun()
    
    def render_data_upload_section(self):
        """Render data upload and ingestion section"""
        st.header("📁 Data Ingestion")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Log Files")
            uploaded_files = st.file_uploader(
                "Choose log files",
                accept_multiple_files=True,
                type=['log', 'txt'],
                help="Upload OpenStack log files for analysis"
            )
            
            if uploaded_files and st.button("Process Uploaded Files"):
                self.process_uploaded_files(uploaded_files)
        
        with col2:
            st.subheader("Load All Logs")
            st.info("Load all OpenStack log files for analysis (one-time setup)")
            
            # Check if logs are already loaded
            if not st.session_state.logs_df.empty:
                st.success(f"✅ {len(st.session_state.logs_df)} logs already loaded")
                if st.button("🔄 Reload All Logs"):
                    self.load_all_logs()
            else:
                if st.button("📊 Load All Logs"):
                    self.load_all_logs()
        
        # Show sample log format
        st.subheader("Expected Log Format")
        sample_log = """
nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:00:00.008 25746 INFO nova.osapi_compute.wsgi.server [req-38101a0b-2096-447d-96ea-a692162415ae] 10.11.10.1 "GET /v2/servers/detail HTTP/1.1" status: 200
nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:00:04.500 2931 INFO nova.compute.manager [req-3ea4052c-895d-4b64-9e2d-04d64c4d94ab] [instance: b9000564-fe1a-409b-b8cc-1e88b294cd1d] VM Started (Lifecycle Event)
        """
        st.code(sample_log, language="text")
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded log files"""
        with st.spinner("Processing uploaded files..."):
            try:
                # Save uploaded files temporarily
                temp_files = []
                for uploaded_file in uploaded_files:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_files.append(temp_path)
                
                # Ingest files
                df = self.ingestion_manager.ingest_multiple_files(temp_files)
                
                # Clean up temporary files
                for temp_file in temp_files:
                    os.remove(temp_file)
                
                if not df.empty:
                    # Apply feature engineering
                    df = self.feature_engineer.engineer_all_features(df)
                    st.session_state.logs_df = df
                    
                    st.success(f"✅ Successfully processed {len(df)} log entries from {len(uploaded_files)} files")
                    st.rerun()
                else:
                    st.error("❌ No valid log entries found in uploaded files")
                    
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")
    
    def load_all_logs(self):
        """Load all available OpenStack log files (optimized one-time process)"""
        with st.spinner("Loading all OpenStack logs (this may take a few minutes)..."):
            try:
                # First, check if VectorDB already has data
                if hasattr(st.session_state, 'vector_db') and st.session_state.vector_db:
                    try:
                        vector_db_count = st.session_state.vector_db.collection.count()
                        if vector_db_count > 0:
                            st.info(f"📊 VectorDB already has {vector_db_count} documents loaded")
                            
                            # Load data from VectorDB instead of file system
                            df = self._load_logs_from_vector_db()
                            
                            if not df.empty:
                                # Apply feature engineering
                                with st.spinner("Applying feature engineering..."):
                                    df = self.feature_engineer.engineer_all_features(df)
                                
                                # Store in session state
                                st.session_state.logs_df = df
                                
                                # Show success message
                                st.success(f"✅ Successfully loaded {len(df)} log entries from VectorDB")
                                st.info("🎯 UI now uses the same data source as RCA analysis")
                                
                                # Show detailed statistics
                                self._show_loading_success(df, [f"VectorDB ({vector_db_count} documents)"])
                                
                                st.rerun()
                                return
                    except Exception as e:
                        st.warning(f"⚠️ Could not load from VectorDB: {e}")
                
                # Fallback to file system loading
                # Step 1: Discover all log files
                log_files = self._discover_all_log_files()
                
                if not log_files:
                    st.error("❌ No log files found. Please ensure log files are in the project directory.")
                    st.info("Expected locations: logs/ directory or root directory")
                    return
                
                st.info(f"📁 Found {len(log_files)} log files to process")
                
                # Step 2: Load and parse logs (without VectorDB storage)
                df = self._load_logs_without_vector_db(log_files)
                
                if not df.empty:
                    # Step 3: Apply feature engineering
                    with st.spinner("Applying feature engineering..."):
                        df = self.feature_engineer.engineer_all_features(df)
                    
                    # Step 4: Store in session state
                    st.session_state.logs_df = df
                    
                    # Step 5: Show success message with detailed stats
                    self._show_loading_success(df, log_files)
                    
                    # Step 6: Offer VectorDB setup (one-time)
                    self._offer_vector_db_setup(df)
                    
                    st.rerun()
                else:
                    st.error("❌ No valid log entries found in the discovered files")
                    
            except Exception as e:
                st.error(f"❌ Error loading logs: {str(e)}")
                st.info("Please check that the log files exist and are readable.")
    
    def _discover_all_log_files(self):
        """Discover all available log files"""
        log_files = []
        
        # Search in multiple locations
        search_paths = [
            'logs/',
            './',
            '../',
            'data/',
            'sample_data/'
        ]
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if file.endswith(('.log', '.txt')) and 'openstack' in file.lower():
                            log_files.append(os.path.join(root, file))
        
        # Also look for specific known files
        specific_files = [
            'OpenStack_2k.log',
            'OpenStack.log',
            'nova.log',
            'openstack.log'
        ]
        
        for file in specific_files:
            if os.path.exists(file):
                log_files.append(file)
        
        return list(set(log_files))  # Remove duplicates
    
    def _load_logs_without_vector_db(self, log_files):
        """Load logs without storing in VectorDB (for performance)"""
        all_dfs = []
        
        # Create a temporary ingestion manager without VectorDB
        from data.log_ingestion import LogIngestionManager
        temp_ingestion_manager = LogIngestionManager()
        temp_ingestion_manager.vector_db = None  # Disable VectorDB
        
        for i, log_file in enumerate(log_files):
            st.text(f"Processing {i+1}/{len(log_files)}: {os.path.basename(log_file)}")
            
            try:
                df = temp_ingestion_manager.ingest_single_file(log_file)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                st.warning(f"⚠️ Failed to process {log_file}: {e}")
                continue
        
        if not all_dfs:
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Sort by timestamp if available
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        return combined_df
    
    def _sync_with_vector_db(self):
        """Sync UI data with VectorDB data"""
        with st.spinner("Syncing with VectorDB..."):
            try:
                df = self._load_logs_from_vector_db()
                
                if not df.empty:
                    # Apply feature engineering
                    with st.spinner("Applying feature engineering..."):
                        df = self.feature_engineer.engineer_all_features(df)
                    
                    # Store in session state
                    st.session_state.logs_df = df
                    
                    st.success(f"✅ Successfully synced with VectorDB! {len(df)} logs loaded")
                    st.info("🎯 UI now uses the same data source as RCA analysis")
                    st.rerun()
                else:
                    st.error("❌ No data found in VectorDB")
                    
            except Exception as e:
                st.error(f"❌ Sync failed: {e}")
    
    def _load_logs_from_vector_db(self):
        """Load logs from VectorDB to sync UI with RCA analysis"""
        try:
            if not st.session_state.vector_db:
                return pd.DataFrame()
            
            # Get all documents from VectorDB
            results = st.session_state.vector_db.collection.get()
            
            if not results['documents']:
                return pd.DataFrame()
            
            # Convert VectorDB results to DataFrame
            logs_data = []
            for i, doc in enumerate(results['documents']):
                metadata = results['metadatas'][i] if results['metadatas'] else {}
                
                log_entry = {
                    'message': doc,
                    'timestamp': metadata.get('timestamp'),
                    'level': metadata.get('level', 'INFO'),
                    'service_type': metadata.get('service_type', 'unknown'),
                    'instance_id': metadata.get('instance_id'),
                    'request_id': metadata.get('request_id'),
                    'source_file': metadata.get('source_file', 'vector_db')
                }
                logs_data.append(log_entry)
            
            df = pd.DataFrame(logs_data)
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Sort by timestamp if available
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            st.info(f"📊 Loaded {len(df)} logs from VectorDB")
            return df
            
        except Exception as e:
            st.error(f"❌ Failed to load from VectorDB: {e}")
            return pd.DataFrame()
    
    def _show_loading_success(self, df, log_files):
        """Show detailed success message with statistics"""
        st.success(f"✅ Successfully loaded {len(df)} log entries from {len(log_files)} files")
        
        # Show detailed statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Log Entries", len(df))
        
        with col2:
            error_count = len(df[df['level'].str.upper() == 'ERROR']) if 'level' in df.columns else 0
            st.metric("Error Entries", error_count)
        
        with col3:
            unique_instances = df['instance_id'].nunique() if 'instance_id' in df.columns else 0
            st.metric("Unique Instances", unique_instances)
        
        with col4:
            unique_services = df['service_type'].nunique() if 'service_type' in df.columns else 0
            st.metric("Services", unique_services)
        
        # Show file breakdown
        if 'source_file' in df.columns:
            st.subheader("📊 Files Processed:")
            file_stats = df['source_file'].value_counts().head(10)
            for file, count in file_stats.items():
                st.text(f"  • {os.path.basename(file)}: {count} entries")
    
    def _offer_vector_db_setup(self, df):
        """Offer to set up VectorDB for historical context (one-time)"""
        st.subheader("🔍 Vector Database Setup (Optional)")
        st.info("VectorDB enables historical context and similarity search for better RCA analysis.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Setup VectorDB (Recommended)"):
                self._setup_vector_db(df)
        
        with col2:
            if st.button("⏭️ Skip VectorDB Setup"):
                st.success("✅ VectorDB setup skipped. You can enable it later for enhanced analysis.")
    
    def _setup_vector_db(self, df):
        """Set up VectorDB with loaded logs (one-time process)"""
        with st.spinner("Setting up VectorDB for historical context..."):
            try:
                # Initialize VectorDB service
                from services.vector_db_service import VectorDBService
                vector_db = VectorDBService()
                
                # Add logs to VectorDB
                logs_added = vector_db.add_logs(df, enable_chunking=False)
                
                st.success(f"✅ VectorDB setup complete! {logs_added} logs stored for historical context.")
                st.info("🎯 You can now use enhanced RCA analysis with historical context.")
                
                # Store VectorDB reference in session state
                st.session_state.vector_db = vector_db
                
            except Exception as e:
                st.error(f"❌ VectorDB setup failed: {e}")
                st.info("You can still use RCA analysis without VectorDB (fast mode).")
    
    def load_sample_data(self):
        """Legacy method - redirect to new load_all_logs"""
        self.load_all_logs()
    
    def create_sample_log_data(self):
        """Create sample log data for demonstration"""
        # This would typically read from the uploaded OpenStack log file
        # For now, we'll create a representative sample
        
        sample_logs = [
            {
                'timestamp': datetime.now() - timedelta(minutes=30),
                'service_type': 'nova-api',
                'level': 'INFO',
                'message': 'GET /v2/servers/detail HTTP/1.1 status: 200',
                'instance_id': None,
                'request_id': 'req-38101a0b-2096-447d-96ea-a692162415ae'
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=25),
                'service_type': 'nova-compute',
                'level': 'INFO',
                'message': '[instance: b9000564-fe1a-409b-b8cc-1e88b294cd1d] VM Started (Lifecycle Event)',
                'instance_id': 'b9000564-fe1a-409b-b8cc-1e88b294cd1d',
                'request_id': 'req-3ea4052c-895d-4b64-9e2d-04d64c4d94ab'
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=20),
                'service_type': 'nova-compute',
                'level': 'ERROR',
                'message': 'No valid host was found. There are not enough hosts available.',
                'instance_id': 'b9000564-fe1a-409b-b8cc-1e88b294cd1d',
                'request_id': 'req-3ea4052c-895d-4b64-9e2d-04d64c4d94ab'
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=15),
                'service_type': 'nova-compute',
                'level': 'INFO',
                'message': '[instance: b9000564-fe1a-409b-b8cc-1e88b294cd1d] Attempting claim: memory 2048 MB, disk 20 GB, vcpus 1 CPU',
                'instance_id': 'b9000564-fe1a-409b-b8cc-1e88b294cd1d',
                'request_id': 'req-caeb3818-dab6-4e8d-9ea6-aceb23905ebc'
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=10),
                'service_type': 'nova-compute',
                'level': 'INFO',
                'message': '[instance: b9000564-fe1a-409b-b8cc-1e88b294cd1d] Instance spawned successfully.',
                'instance_id': 'b9000564-fe1a-409b-b8cc-1e88b294cd1d',
                'request_id': 'req-caeb3818-dab6-4e8d-9ea6-aceb23905ebc'
            }
        ]
        
        return pd.DataFrame(sample_logs)
    
    def render_dashboard(self):
        """Render main dashboard with log statistics"""
        st.header("📊 OpenStack Log Dashboard")
        
        df = st.session_state.logs_df
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Logs", len(df))
        
        with col2:
            error_count = len(df[df['level'] == 'ERROR']) if 'level' in df.columns else 0
            st.metric("Errors", error_count)
        
        with col3:
            unique_instances = df['instance_id'].nunique() if 'instance_id' in df.columns else 0
            st.metric("Unique Instances", unique_instances)
        
        with col4:
            unique_services = df['service_type'].nunique() if 'service_type' in df.columns else 0
            st.metric("Services", unique_services)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if 'level' in df.columns:
                st.subheader("Log Level Distribution")
                level_counts = df['level'].value_counts()
                fig = px.pie(values=level_counts.values, names=level_counts.index, 
                           title="Distribution of Log Levels")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'service_type' in df.columns:
                st.subheader("Service Distribution")
                service_counts = df['service_type'].value_counts()
                fig = px.bar(x=service_counts.index, y=service_counts.values,
                           title="Logs by Service Type")
                st.plotly_chart(fig, use_container_width=True)
        
        # Timeline view
        if 'timestamp' in df.columns:
            st.subheader("Log Timeline")
            
            # Create hourly aggregation
            df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('h')
            timeline_data = df.groupby(['hour', 'level']).size().reset_index(name='count')
            
            fig = px.line(timeline_data, x='hour', y='count', color='level',
                         title="Log Activity Over Time")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_chat_interface(self):
        """Render the RCA chat interface"""
        st.header("🤖 Root Cause Analysis Chat")
        
        if st.session_state.rca_analyzer is None:
            st.warning("⚠️ Please configure your Anthropic API key in the sidebar to use the RCA chat.")
            return
        
        # Chat interface
        st.subheader("Describe your issue:")
        
        # Issue input
        issue_description = st.text_area(
            "What problem are you experiencing?",
            placeholder="e.g., I'm trying to launch a VM through Horizon, but it keeps failing. The error message says 'No valid host was found.'",
            height=100
        )
        
        # Add toggle for showing prompt (default to global setting)
        show_prompt = st.checkbox(
            "🔍 Show final prompt sent to LLM", 
            value=st.session_state.get('show_prompt_global', False), 
            help="Display the complete prompt that was sent to the AI model"
        )
        
        if st.button("🔍 Analyze Issue") and issue_description:
            self.perform_rca_analysis(issue_description, fast_mode=False, show_prompt=show_prompt)
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Analysis Results:")
            for i, chat in enumerate(st.session_state.chat_history):
                with st.expander(f"Analysis {i+1}: {chat['issue'][:50]}...", expanded=(i == len(st.session_state.chat_history)-1)):
                    # Use local setting if available, otherwise use global setting
                    local_show_prompt = chat.get('show_prompt', st.session_state.get('show_prompt_global', False))
                    self.display_rca_results(chat['results'], local_show_prompt)
    
    def perform_rca_analysis(self, issue_description, fast_mode=False, show_prompt=False):
        """Perform RCA analysis on the described issue"""
        with st.spinner("Analyzing logs and generating root cause analysis..."):
            try:
                # Use pre-loaded VectorDB if available, otherwise create new one
                if hasattr(st.session_state, 'vector_db') and st.session_state.vector_db:
                    # Use existing VectorDB for better performance
                    rca_analyzer = RCAAnalyzer(
                        Config.ANTHROPIC_API_KEY, 
                        st.session_state.lstm_model,
                        st.session_state.vector_db  # Use pre-loaded VectorDB
                    )
                else:
                    # Create new RCA analyzer (will initialize VectorDB if needed)
                    rca_analyzer = RCAAnalyzer(
                        Config.ANTHROPIC_API_KEY, 
                        st.session_state.lstm_model
                    )
                
                results = rca_analyzer.analyze_issue(
                    issue_description, 
                    st.session_state.logs_df,
                    fast_mode=fast_mode
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'timestamp': datetime.now(),
                    'issue': issue_description,
                    'results': results,
                    'fast_mode': fast_mode,
                    'show_prompt': show_prompt
                })
                
                st.success("✅ Analysis complete!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
    
    def display_rca_results(self, results, show_prompt=False):
        """Display RCA analysis results"""
        # Issue summary
        st.write(f"**Issue Category:** {results.get('issue_category', 'Unknown')}")
        st.write(f"**Relevant Logs:** {results.get('relevant_logs_count', 0)}")
        
        # Analysis mode indicator
        analysis_mode = results.get('analysis_mode', 'full')
        if analysis_mode == 'hybrid':
            st.info("🔍 Analysis performed in Hybrid Mode (LSTM + Vector DB + TF-IDF)")
        elif analysis_mode == 'fast':
            st.info("⚡ Analysis performed in Fast Mode (LSTM + TF-IDF only)")
        else:
            st.info("🔍 Analysis performed in Full Mode")
        
        # Historical Context (NEW)
        if hasattr(st.session_state, 'vector_db') and st.session_state.vector_db:
            try:
                # Extract issue description from results or use a placeholder
                issue_description = results.get('issue_description', 'Unknown issue')
                
                # Get historical context
                historical_context = st.session_state.vector_db.get_context_for_issue(issue_description, top_k=3)
                
                if historical_context:
                    st.subheader("📚 Historical Context")
                    st.info("Similar issues found in historical data:")
                    st.text(historical_context)
            except Exception as e:
                st.warning(f"⚠️ Could not retrieve historical context: {e}")
        
        # Root cause analysis
        if 'root_cause_analysis' in results:
            st.subheader("🎯 Root Cause Analysis")
            st.markdown(results['root_cause_analysis'])
        
        # Timeline
        if results.get('timeline'):
            st.subheader("⏱️ Event Timeline")
            timeline_df = pd.DataFrame(results['timeline'])
            
            # Check which columns are available and display them
            available_columns = []
            expected_columns = ['timestamp', 'event_type', 'service', 'level']
            
            for col in expected_columns:
                if col in timeline_df.columns:
                    available_columns.append(col)
            
            if available_columns:
                st.dataframe(timeline_df[available_columns])
            else:
                st.dataframe(timeline_df)  # Display all available columns
        
        # Patterns
        if results.get('patterns'):
            st.subheader("📊 Detected Patterns")
            patterns = results['patterns']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if patterns.get('error_frequency'):
                    st.write("**Error Frequency:**")
                    for error, count in patterns['error_frequency'].items():
                        st.write(f"- {error}: {count}")
            
            with col2:
                if patterns.get('service_distribution'):
                    st.write("**Service Distribution:**")
                    for service, count in patterns['service_distribution'].items():
                        st.write(f"- {service}: {count}")
        
        # Recommendations
        if results.get('recommendations'):
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(results['recommendations'], 1):
                st.write(f"{i}. {rec}")
        
        if show_prompt:
            st.subheader("🔍 Prompt Sent to LLM:")
            st.code(results.get('prompt', 'N/A'), language="text")
    
    def render_log_analysis(self):
        """Render detailed log analysis section"""
        st.header("📈 Detailed Log Analysis")
        
        df = st.session_state.logs_df
        
        if df.empty:
            st.info("No log data available for analysis.")
            return
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'level' in df.columns:
                selected_levels = st.multiselect(
                    "Filter by Log Level",
                    options=df['level'].unique(),
                    default=df['level'].unique()
                )
                df = df[df['level'].isin(selected_levels)]
        
        with col2:
            if 'service_type' in df.columns:
                selected_services = st.multiselect(
                    "Filter by Service",
                    options=df['service_type'].unique(),
                    default=df['service_type'].unique()
                )
                df = df[df['service_type'].isin(selected_services)]
        
        with col3:
            if 'timestamp' in df.columns:
                date_range = st.date_input(
                    "Date Range",
                    value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
                    min_value=df['timestamp'].min().date(),
                    max_value=df['timestamp'].max().date()
                )
        
        # Display filtered data
        st.subheader("Filtered Log Entries")
        
        # Show important columns
        display_columns = ['timestamp', 'level', 'service_type', 'message']
        if 'instance_id' in df.columns:
            display_columns.append('instance_id')
        
        available_columns = [col for col in display_columns if col in df.columns]
        st.dataframe(df[available_columns].head(100))
        
        # Export functionality
        if st.button("📥 Export Filtered Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"openstack_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def render_model_training(self):
        """Render model training section"""
        st.header("⚙️ LSTM Model Training")
        
        df = st.session_state.logs_df
        
        if df.empty:
            st.info("No log data available for training.")
            return
        
        # Model configuration
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider("Epochs", min_value=10, max_value=100, value=50)
            batch_size = st.slider("Batch Size", min_value=16, max_value=128, value=32)
        
        with col2:
            lstm_units = st.slider("LSTM Units", min_value=32, max_value=128, value=64)
            dropout_rate = st.slider("Dropout Rate", min_value=0.1, max_value=0.5, value=0.2)
        
        # Training button
        if st.button("🚀 Train LSTM Model"):
            self.train_lstm_model(epochs, batch_size, lstm_units, dropout_rate)
        
        # Model status
        if st.session_state.lstm_model is not None:
            st.success("✅ LSTM model is trained and ready!")
            
            # Model evaluation metrics would go here
            st.subheader("Model Performance")
            st.info("Model performance metrics will be displayed here after training.")
    
    def train_lstm_model(self, epochs, batch_size, lstm_units, dropout_rate):
        """Train the LSTM model"""
        with st.spinner("Training LSTM model..."):
            try:
                # Prepare training configuration
                config = {
                    'max_sequence_length': 100,
                    'embedding_dim': 128,
                    'lstm_units': lstm_units,
                    'dropout_rate': dropout_rate,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'validation_split': 0.2
                }
                
                # Initialize LSTM classifier
                lstm_classifier = LSTMLogClassifier(config)
                
                # Prepare data for training
                X, y = self.preprocessor.prepare_lstm_data(st.session_state.logs_df)
                
                # Train model
                results = lstm_classifier.train(X, y)
                
                # Save model to session state
                st.session_state.lstm_model = lstm_classifier
                
                # Update RCA analyzer with trained model
                if st.session_state.rca_analyzer:
                    st.session_state.rca_analyzer.lstm_model = lstm_classifier
                
                st.success("✅ Model training completed!")
                
                # Display training results
                st.subheader("Training Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Training Accuracy", f"{results['train_accuracy']:.3f}")
                    st.metric("Training Precision", f"{results['train_precision']:.3f}")
                
                with col2:
                    st.metric("Validation Accuracy", f"{results['val_accuracy']:.3f}")
                    st.metric("Validation Precision", f"{results['val_precision']:.3f}")
                
                # Show classification report
                st.text("Classification Report:")
                st.code(results['classification_report'])
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

def main():
    """Main function to run the Streamlit app"""
    app = OpenStackRCAAssistant()
    app.run()

if __name__ == "__main__":
    main()